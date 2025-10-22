"""
3D Pointmap Decoder

This module decodes object-centric slots into 3D pointmaps:
1. Embeds camera pose into slots
2. Decodes each slot to a per-object 3D pointmap
3. Aggregates slot outputs into a scene pointmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class PoseEmbedding(nn.Module):
    """Embed camera pose to add to slots."""
    
    def __init__(self, pose_dim: int = 7, embed_dim: int = 64):
        """
        Initialize pose embedding.
        
        Args:
            pose_dim: Dimension of pose (default: 7 for [tx, ty, tz, qx, qy, qz, qw])
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        
        # MLP to embed pose
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Embed camera pose.
        
        Args:
            pose: Camera pose (B, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            
        Returns:
            Pose embedding (B, embed_dim)
        """
        B = pose.shape[0]
        
        # Pose is already in the correct format (B, 7)
        if pose.shape[-1] != 7:
            raise ValueError(f"Expected pose shape (B, 7), got {pose.shape}")
        
        # Embed
        pose_embed = self.mlp(pose)
        
        return pose_embed


class Decoder3D(nn.Module):
    """
    3D Pointmap Decoder using Spatial Broadcast.
    
    This decoder:
    1. Takes slots and camera pose as input
    2. Uses spatial broadcast to expand slots to spatial grid
    3. Decodes each slot to a 3D pointmap with alpha mask
    4. Aggregates all slots into final scene pointmap
    """
    
    def __init__(
        self,
        slot_dim: int,
        pose_dim: int = 7,
        pose_embed_dim: int = 64,
        resolution: Tuple[int, int] = (64, 64),
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        output_type: str = 'pointmap',  # 'pointmap' or 'depth_rgb'
    ):
        """
        Initialize 3D decoder.
        
        Args:
            slot_dim: Dimension of each slot
            pose_dim: Dimension of pose (7 for [tx, ty, tz, qx, qy, qz, qw])
            pose_embed_dim: Dimension of pose embedding
            resolution: Output resolution (H, W)
            hidden_dims: Hidden dimensions for decoder CNN
            output_type: 'pointmap' for XYZ or 'depth_rgb' for depth+RGB
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.resolution = resolution
        self.output_type = output_type
        
        # Pose embedding
        self.pose_embedding = PoseEmbedding(pose_dim, pose_embed_dim)
        
        # Slot dimension with pose
        slot_with_pose_dim = slot_dim + pose_embed_dim
        
        # Spatial coordinates for broadcast
        H, W = resolution
        y_coords = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
        self.register_buffer('coords', coords)
        
        # Input dimension: slot + pose + spatial coords
        input_dim = slot_with_pose_dim + 2
        
        # CNN decoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        if output_type == 'pointmap':
            # Output: XYZ coordinates + alpha mask
            output_dim = 4  # (X, Y, Z, alpha)
        elif output_type == 'depth_rgb':
            # Output: Depth + RGB + alpha mask
            output_dim = 5  # (depth, R, G, B, alpha)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        layers.append(nn.Conv2d(prev_dim, output_dim, 1))
        
        self.decoder = nn.Sequential(*layers)
        
    def _spatial_broadcast(
        self, 
        slots: torch.Tensor,
        pose_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spatial broadcast: expand slots to spatial grid.
        
        Args:
            slots: Slot representations (B, K, D_slot)
            pose_embed: Pose embedding (B, D_pose)
            
        Returns:
            Broadcasted features (B, K, D_slot + D_pose + 2, H, W)
        """
        B, K, _ = slots.shape
        H, W = self.resolution
        
        # Add pose to slots
        pose_expand = pose_embed.unsqueeze(1).expand(B, K, -1)  # (B, K, D_pose)
        slots_with_pose = torch.cat([slots, pose_expand], dim=-1)  # (B, K, D_slot + D_pose)
        
        # Broadcast to spatial dimensions
        slots_broadcast = slots_with_pose.view(B, K, -1, 1, 1).expand(B, K, -1, H, W)
        # (B, K, D_slot + D_pose, H, W)
        
        # Add spatial coordinates
        coords = self.coords.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1, -1)
        # (B, K, H, W, 2) -> (B, K, 2, H, W)
        coords = coords.permute(0, 1, 4, 2, 3)
        
        # Concatenate
        features = torch.cat([slots_broadcast, coords], dim=2)
        # (B, K, D_slot + D_pose + 2, H, W)
        
        return features
    
    def forward(
        self, 
        slots: torch.Tensor,
        pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode slots to 3D pointmap.
        
        Args:
            slots: Object slots (B, K, D_slot)
            pose: Camera pose (B, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            
        Returns:
            Dictionary with:
            - 'pointmap': Scene pointmap (B, H, W, 3) in world coordinates
            - 'masks': Soft masks (B, K, H, W) normalized across slots
        """
        B, K, _ = slots.shape
        H, W = self.resolution
        
        # Embed pose
        pose_embed = self.pose_embedding(pose)  # (B, D_pose)
        
        # Spatial broadcast
        features = self._spatial_broadcast(slots, pose_embed)
        # (B, K, D_slot + D_pose + 2, H, W)
        
        # Decode each slot independently
        # Reshape to process all slots in batch
        features_flat = features.view(B * K, -1, H, W)
        outputs_flat = self.decoder(features_flat)  # (B*K, output_dim, H, W)
        outputs = outputs_flat.view(B, K, -1, H, W)
        
        # Split into pointmaps and alpha masks
        if self.output_type == 'pointmap':
            pointmaps = outputs[:, :, :3, :, :]  # (B, K, 3, H, W)
            alphas = outputs[:, :, 3:4, :, :]  # (B, K, 1, H, W)
        elif self.output_type == 'depth_rgb':
            depth = outputs[:, :, 0:1, :, :]  # (B, K, 1, H, W)
            rgb = outputs[:, :, 1:4, :, :]  # (B, K, 3, H, W)
            alphas = outputs[:, :, 4:5, :, :]  # (B, K, 1, H, W)
            # Convert to pointmaps (simplified - would need camera intrinsics)
            pointmaps = depth.expand(-1, -1, 3, -1, -1)  # Placeholder
        
        # Normalize alpha masks (softmax over slots) - this becomes the segmentation mask
        alphas = F.softmax(alphas, dim=1)  # (B, K, 1, H, W)
        masks = alphas.squeeze(2)  # (B, K, H, W) - SlotContrast format
        
        # Weighted sum aggregation
        # pointmaps: (B, K, 3, H, W)
        # alphas: (B, K, 1, H, W)
        scene_pointmap = (pointmaps * alphas).sum(dim=1)  # (B, 3, H, W)
        
        # Permute to (B, H, W, 3)
        scene_pointmap = scene_pointmap.permute(0, 2, 3, 1)
        
        return {
            'pointmap': scene_pointmap,
            'masks': masks,  # (B, K, H, W) - soft segmentation masks
        }
    
    def get_slot_pointmaps(
        self,
        slots: torch.Tensor,
        pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-slot pointmaps and masks (for visualization).
        
        Args:
            slots: Object slots (B, K, D_slot)
            pose: Camera pose (B, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            
        Returns:
            Tuple of:
            - Per-slot pointmaps (B, K, H, W, 3)
            - Per-slot alpha masks (B, K, H, W)
        """
        B, K, _ = slots.shape
        H, W = self.resolution
        
        # Embed pose
        pose_embed = self.pose_embedding(pose)
        
        # Spatial broadcast and decode
        features = self._spatial_broadcast(slots, pose_embed)
        features_flat = features.view(B * K, -1, H, W)
        outputs_flat = self.decoder(features_flat)
        outputs = outputs_flat.view(B, K, -1, H, W)
        
        # Extract pointmaps and alphas
        pointmaps = outputs[:, :, :3, :, :].permute(0, 1, 3, 4, 2)  # (B, K, H, W, 3)
        alphas = outputs[:, :, 3, :, :]  # (B, K, H, W)
        alphas = F.softmax(alphas, dim=1)
        
        return pointmaps, alphas
    
    def __repr__(self):
        return (
            f"Decoder3D(\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  resolution={self.resolution},\n"
            f"  output_type='{self.output_type}'\n"
            f")"
        )


# Test function
def test_decoder():
    """Test the 3D decoder module."""
    print("Testing 3D Decoder...")
    
    # Parameters
    batch_size = 2
    num_slots = 8
    slot_dim = 128
    resolution = (64, 64)
    
    # Create decoder
    decoder = Decoder3D(
        slot_dim=slot_dim,
        resolution=resolution,
        output_type='pointmap'
    )
    
    # Test inputs
    slots = torch.randn(batch_size, num_slots, slot_dim)
    pose = torch.randn(batch_size, 7)  # [tx, ty, tz, qx, qy, qz, qw]
    
    # Forward pass
    pointmap = decoder(slots, pose)
    print(f"✓ Scene pointmap shape: {pointmap.shape}")
    assert pointmap.shape == (batch_size, *resolution, 3)
    
    # Test per-slot outputs
    slot_pointmaps, slot_alphas = decoder.get_slot_pointmaps(slots, pose)
    print(f"✓ Per-slot pointmaps shape: {slot_pointmaps.shape}")
    print(f"✓ Per-slot alphas shape: {slot_alphas.shape}")
    assert slot_pointmaps.shape == (batch_size, num_slots, *resolution, 3)
    assert slot_alphas.shape == (batch_size, num_slots, *resolution)
    
    # Check alpha masks sum to 1
    alpha_sum = slot_alphas.sum(dim=1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
    print("✓ Alpha masks sum to 1")
    
    print("\n3D Decoder test passed! ✓")


if __name__ == "__main__":
    test_decoder()
