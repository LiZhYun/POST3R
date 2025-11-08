"""
3D Pointmap Decoder

This module decodes object-centric slots into 3D pointmaps and features:
1. Embeds camera pose into slots
2. Decodes each slot to a per-object 3D pointmap
3. Decodes each slot to per-object features
4. Aggregates slot outputs into scene pointmap and features
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math

# Add SlotContrast to path
SLOTCONTRAST_PATH = Path(__file__).parent.parent.parent / "submodules" / "slotcontrast"
if str(SLOTCONTRAST_PATH) not in sys.path:
    sys.path.insert(0, str(SLOTCONTRAST_PATH))


class PoseEmbedding(nn.Module):
    """Embed camera pose to add to slots."""
    
    def __init__(self, pose_dim: int = 7, embed_dim: int = 64):
        """
        Initialize pose embedding.
        
        Args:
            pose_dim: Dimension of pose (default: 7 for [tx, ty, tz, qx, q        slot_dim: int = 128,
        feature_dim: int = 1024,
        pose_dim: int = 7,
        pose_embed_dim: int = 64,
        pointmap_resolution: Tuple[int, int] = (512, 512),  # Default output resolution
        pointmap_hidden_dims: Tuple[int, ...] = (256, 256, 128),  # Legacy, kept for compatibility qw])
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


class FeatureHead(nn.Module):
    """
    Feature decoder head following SlotContrast's MLPDecoder pattern.
    
    Decodes slots into feature maps that reconstruct the encoder features.
    Uses MLP with positional embeddings (like SlotContrast), without pose conditioning.
    """
    
    def __init__(
        self,
        slot_dim: int,
        output_dim: int = 1024,
        n_patches: int = 1024,  # Number of patches from TTT3R encoder (32x32 for 512x512 image with patch_size=16)
        hidden_dims: Tuple[int, ...] = (512,),
        activation: str = "relu",
    ):
        """
        Initialize feature head.
        
        Args:
            slot_dim: Dimension of each slot
            output_dim: Output feature dimension (should match encoder output, e.g., 1024)
            n_patches: Number of patches from encoder (e.g., 1024 for 32x32)
            hidden_dims: Hidden dimensions for MLP (tuple of ints)
            activation: Activation function name
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.output_dim = output_dim
        self.n_patches = n_patches
        
        # Input dimension: slot only (no pose)
        input_dim = slot_dim
        
        # Positional embedding (SlotContrast-style)
        # Shape: (1, 1, n_patches, input_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, input_dim) * input_dim**-0.5)
        
        # MLP decoder (SlotContrast-style)
        # Input: slot + pos_emb -> Output: features + alpha mask
        from slotcontrast.modules.networks import MLP
        self.mlp = MLP(
            input_dim, 
            output_dim + 1,  # +1 for alpha mask
            list(hidden_dims), 
            activation=activation
        )
    
    def forward(
        self,
        slots: torch.Tensor,
        target_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode slots to feature map.
        
        Args:
            slots: Object slots (B, K, D_slot)
            target_shape: Target spatial shape (H, W) - used for reshaping output
            
        Returns:
            Dictionary with:
            - 'features': Scene features (B, H, W, D_feat)
            - 'masks': Soft masks (B, K, H, W)
        """
        B, K, _ = slots.shape
        H, W = target_shape
        
        # Expand slots to all patches: (B, K, D) -> (B, K, N, D)
        slots_expanded = slots.unsqueeze(2).expand(B, K, self.n_patches, -1)
        
        # Add positional embeddings (SlotContrast-style)
        # pos_emb: (1, 1, N, D) -> broadcast to (B, K, N, D)
        slots_with_pos = slots_expanded + self.pos_emb
        
        # Apply MLP: (B, K, N, D) -> (B, K, N, output_dim + 1)
        mlp_out = self.mlp(slots_with_pos)
        
        # Split into features and alpha
        recons = mlp_out[..., :self.output_dim]  # (B, K, N, output_dim)
        alpha = mlp_out[..., self.output_dim:]    # (B, K, N, 1)
        
        # Compute masks via softmax over slots
        masks = torch.softmax(alpha, dim=1)  # (B, K, N, 1)
        
        # Aggregate features: sum over slots weighted by masks
        recon = torch.sum(recons * masks, dim=1)  # (B, N, output_dim)
        
        # Reshape to spatial: (B, N, D) -> (B, H, W, D)
        # Assuming N = H * W
        assert self.n_patches == H * W, f"n_patches ({self.n_patches}) must equal H*W ({H*W})"
        recon = recon.view(B, H, W, self.output_dim)
        
        # Reshape masks: (B, K, N, 1) -> (B, K, H, W)
        masks = masks.squeeze(-1).view(B, K, H, W)
        
        return {
            'features': recon,
            'masks': masks
        }
        # It operates on tokens of shape (B, S, D) where S = H_low * W_low
        # and outputs (D_out + conf) * patch_size^2 per token
        
        # For our case, we have K slots, and we want to produce H x W features
        # Let's assume slots correspond to spatial locations at low resolution
        
        # Simpler approach: use a small CNN decoder per slot
        # But that's what Decoder3D does...
        
        # Let me use a broadcast-based approach:
        # Each slot produces features uniformly, then we use alpha masks to aggregate
        
        # Reshape features: (B, K, D*P^2) -> (B*K, D, P, P)
        features_per_slot = features_flat.view(B, K, self.output_dim, self.patch_size, self.patch_size)
        features_per_slot = features_per_slot.view(B * K, self.output_dim, self.patch_size, self.patch_size)
        
        # Upsample via pixel shuffle or interpolation to target size
        # Pixel shuffle requires specific format, so let's use interpolation
        features_per_slot = F.interpolate(
            features_per_slot,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B*K, D, H, W)
        
        features_per_slot = features_per_slot.view(B, K, self.output_dim, H, W)
        
        # Alphas: (B, K, P^2) -> (B, K, P, P) -> (B, K, H, W)
        alphas_per_slot = alphas_flat.view(B, K, self.patch_size, self.patch_size)
        alphas_per_slot = F.interpolate(
            alphas_per_slot,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B, K, H, W)
        
        # Normalize alpha masks (softmax over slots)
        masks = F.softmax(alphas_per_slot, dim=1)  # (B, K, H, W)
        
        # Weighted aggregation
        # features_per_slot: (B, K, D, H, W)
        # masks: (B, K, H, W)
        masks_expanded = masks.unsqueeze(2)  # (B, K, 1, H, W)
        scene_features = (features_per_slot * masks_expanded).sum(dim=1)  # (B, D, H, W)
        
        # Permute to (B, H, W, D)
        scene_features = scene_features.permute(0, 2, 3, 1)
        
        return {
            'features': scene_features,
            'masks': masks,
        }


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


class SlotToDPTAdapter(nn.Module):
    """
    Adapter to convert slots into multi-scale token representations for DPT.
    
    DPT expects multi-scale features from transformer decoder layers.
    We simulate this by:
    1. Adding pose embedding to slots
    2. Spatially broadcasting slots to create token grids
    3. Using multi-layer transformations to create multi-scale representations
    """
    
    def __init__(
        self,
        slot_dim: int,
        pose_dim: int = 7,
        pose_embed_dim: int = 64,
        target_dim: int = 768,  # Decoder embed dim for DPT
        num_scales: int = 4,
        spatial_size: Tuple[int, int] = (18, 32),  # (H, W) tokens, e.g., 288/16=18, 512/16=32
    ):
        """
        Initialize slot to DPT adapter.
        
        Args:
            slot_dim: Dimension of input slots
            pose_dim: Dimension of pose (7 for quaternion + translation)
            pose_embed_dim: Dimension of pose embedding
            target_dim: Target dimension for DPT tokens (dec_embed_dim)
            num_scales: Number of scale levels (typically 4 for DPT)
            spatial_size: Number of tokens (H, W)
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.pose_embed_dim = pose_embed_dim
        self.target_dim = target_dim
        self.num_scales = num_scales
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size[0] * spatial_size[1]
        
        # Pose embedding
        self.pose_embedding = PoseEmbedding(pose_dim, pose_embed_dim)
        
        # Projection from slots + pose to target dimension
        self.slot_proj = nn.Linear(slot_dim + pose_embed_dim, target_dim)
        
        # Learnable positional embeddings for spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, target_dim) * 0.02)
        
        # Multi-scale transformations (simple MLPs for each scale)
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, target_dim),
                nn.GELU(),
                nn.Linear(target_dim, target_dim),
            )
            for _ in range(num_scales)
        ])
    
    def forward(self, slots: torch.Tensor, pose: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert slots to multi-scale token representations.
        
        Args:
            slots: Input slots (B, K, D_slot)
            pose: Camera pose (B, 7)
            
        Returns:
            List of token representations at different scales:
            [tokens_scale0, tokens_scale1, tokens_scale2, tokens_scale3]
            Each has shape (B, num_tokens, target_dim)
        """
        B, K, _ = slots.shape
        
        # Embed pose (B, 7) -> (B, D_pose)
        pose_embed = self.pose_embedding(pose)  # (B, pose_embed_dim)
        
        # Add pose to each slot: (B, K, D_slot) + (B, 1, D_pose) -> (B, K, D_slot + D_pose)
        pose_expand = pose_embed.unsqueeze(1).expand(B, K, -1)
        slots_with_pose = torch.cat([slots, pose_expand], dim=-1)  # (B, K, D_slot + D_pose)
        
        # Project slots + pose to target dimension
        slots_proj = self.slot_proj(slots_with_pose)  # (B, K, target_dim)
        
        # Average pool slots to get global representation
        global_feat = slots_proj.mean(dim=1, keepdim=True)  # (B, 1, target_dim)
        
        # Broadcast to spatial tokens
        global_feat = global_feat.expand(B, self.num_tokens, -1)  # (B, num_tokens, target_dim)
        
        # Add positional embeddings
        tokens_base = global_feat + self.pos_embed  # (B, num_tokens, target_dim)
        
        # Generate multi-scale representations
        multi_scale_tokens = []
        for scale_idx, transform in enumerate(self.scale_transforms):
            tokens = transform(tokens_base)
            multi_scale_tokens.append(tokens)
        
        return multi_scale_tokens


class DPTPointmapHead(nn.Module):
    """
    DPT-based pointmap head following TTT3R's DPTPts3dPose architecture.
    
    Adapted to work with slots instead of transformer decoder outputs.
    Uses DPT's multi-scale architecture for robust 3D pointmap prediction.
    """
    
    def __init__(
        self,
        slot_dim: int,
        pose_dim: int = 7,
        dec_embed_dim: int = 768,
        spatial_size: Tuple[int, int] = (18, 32),
        output_channels: int = 4,  # 3 for XYZ + 1 for confidence
        feature_dim: int = 256,
        last_dim: int = 128,
    ):
        """
        Initialize DPT pointmap head.
        
        Args:
            slot_dim: Dimension of input slots
            pose_dim: Dimension of pose vector
            dec_embed_dim: Decoder embedding dimension (for DPT)
            spatial_size: Spatial size of token grid (H, W)
            output_channels: Number of output channels (3 for XYZ + 1 for conf)
            feature_dim: Feature dimension for DPT
            last_dim: Last layer dimension for DPT
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.dec_embed_dim = dec_embed_dim
        self.spatial_size = spatial_size
        self.output_channels = output_channels
        
        # Slot to DPT adapter (now includes pose embedding)
        self.slot_adapter = SlotToDPTAdapter(
            slot_dim=slot_dim,
            pose_dim=pose_dim,
            pose_embed_dim=64,  # Internal pose embedding dimension
            target_dim=dec_embed_dim,
            num_scales=4,
            spatial_size=spatial_size,
        )
        
        # # Import DPT components from TTT3R
        # # We'll create a simplified DPT adapter
        # try:
        import sys
        from pathlib import Path
        TTT3R_PATH = Path(__file__).parent.parent.parent / "submodules" / "ttt3r" / "src"
        if str(TTT3R_PATH) not in sys.path:
            sys.path.insert(0, str(TTT3R_PATH))
        
        from models.dpt_block import DPTOutputAdapter
        
        # DPT output adapter configuration
        hooks_idx = [0, 1, 2, 3]  # Multi-scale hooks
        dim_tokens = [dec_embed_dim] * 4  # Same dim for all scales
        
        dpt_args = dict(
            output_width_ratio=1,
            num_channels=output_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks=hooks_idx,
            head_type="regression",
        )
        
        self.dpt = DPTOutputAdapter(**dpt_args)
        self.dpt.init(dim_tokens_enc=dim_tokens)
        self.has_dpt = True
            
        # except Exception as e:
        #     print(f"Warning: Could not load DPT components: {e}")
        #     print("Falling back to simple CNN decoder")
        #     self.has_dpt = False
            
        #     # Fallback: simple CNN decoder
        #     self.fallback_decoder = nn.Sequential(
        #         nn.Linear(slot_dim, 256),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(256, spatial_size[0] * spatial_size[1] * output_channels),
        #     )
    
    def forward(
        self,
        slots: torch.Tensor,
        pose: torch.Tensor,
        target_size: Tuple[int, int] = (288, 512),
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DPT pointmap head.
        
        Args:
            slots: Object slots (B, K, D_slot)
            pose: Camera pose (B, 7)
            target_size: Target output size (H, W)
            
        Returns:
            Dictionary with:
            - 'pointmap': 3D pointmap (B, H, W, 3)
            - 'confidence': Confidence map (B, H, W)
            - 'masks': Per-slot masks (B, K, H, W)
        """
        B, K, _ = slots.shape
        H, W = target_size
        
        if self.has_dpt:
            # Process each slot independently to get per-slot pointmaps and alphas
            # Following the same style as feature head's alpha composition
            
            all_pointmaps = []
            all_alphas = []
            
            for k in range(K):
                # Get single slot
                slot_k = slots[:, k:k+1, :]  # (B, 1, D_slot)
                
                # Convert to multi-scale tokens (now with pose)
                multi_scale_tokens = self.slot_adapter(slot_k, pose)  # List of 4 tensors
                
                # Apply DPT to get per-slot output
                dpt_output = self.dpt(multi_scale_tokens, image_size=(H, W))  # (B, C, H, W)
                
                # Split into pointmap and alpha
                if self.output_channels == 4:
                    pointmap_k = dpt_output[:, :3, :, :]  # (B, 3, H, W)
                    alpha_k = dpt_output[:, 3:4, :, :]  # (B, 1, H, W)
                else:
                    pointmap_k = dpt_output[:, :3, :, :]
                    alpha_k = torch.zeros(B, 1, H, W, device=pointmap_k.device)
                
                all_pointmaps.append(pointmap_k)
                all_alphas.append(alpha_k)
            
            # Stack per-slot outputs
            pointmaps_per_slot = torch.stack(all_pointmaps, dim=1)  # (B, K, 3, H, W)
            alphas_per_slot = torch.stack(all_alphas, dim=1)  # (B, K, 1, H, W)
            
            # Normalize alpha masks (softmax over slots) - following feature head style
            masks = F.softmax(alphas_per_slot.squeeze(2), dim=1)  # (B, K, H, W)
            
            # Weighted aggregation - same as feature head
            masks_expanded = masks.unsqueeze(2)  # (B, K, 1, H, W)
            scene_pointmap = (pointmaps_per_slot * masks_expanded).sum(dim=1)  # (B, 3, H, W)
            
            # Permute to (B, H, W, 3)
            scene_pointmap = scene_pointmap.permute(0, 2, 3, 1)
            
            # Confidence can be derived from max mask value or kept separate
            confidence = masks.max(dim=1)[0]  # (B, H, W) - max slot confidence per pixel
            
        else:
            # Fallback path - also follow per-slot style
            all_pointmaps = []
            all_alphas = []
            
            for k in range(K):
                slot_k = slots[:, k:k+1, :]  # (B, 1, D_slot)
                slots_pooled = slot_k.squeeze(1)  # (B, D_slot)
                
                # Decode
                output = self.fallback_decoder(slots_pooled)  # (B, H*W*C)
                output = output.view(B, self.spatial_size[0], self.spatial_size[1], self.output_channels)
                
                # Upsample to target size
                output = F.interpolate(
                    output.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )  # (B, C, H, W)
                
                pointmap_k = output[:, :3, :, :]  # (B, 3, H, W)
                alpha_k = output[:, 3:4, :, :] if self.output_channels == 4 else torch.zeros(B, 1, H, W, device=output.device)
                
                all_pointmaps.append(pointmap_k)
                all_alphas.append(alpha_k)
            
            # Stack and aggregate
            pointmaps_per_slot = torch.stack(all_pointmaps, dim=1)  # (B, K, 3, H, W)
            alphas_per_slot = torch.stack(all_alphas, dim=1)  # (B, K, 1, H, W)
            
            # Normalize masks
            masks = F.softmax(alphas_per_slot.squeeze(2), dim=1)  # (B, K, H, W)
            masks_expanded = masks.unsqueeze(2)  # (B, K, 1, H, W)
            
            # Weighted aggregation
            scene_pointmap = (pointmaps_per_slot * masks_expanded).sum(dim=1)  # (B, 3, H, W)
            scene_pointmap = scene_pointmap.permute(0, 2, 3, 1)  # (B, H, W, 3)
            
            confidence = masks.max(dim=1)[0]  # (B, H, W)
        
        return {
            'pointmap': scene_pointmap,
            'confidence': confidence,
            'masks': masks,  # (B, K, H, W) - per-slot masks, same style as feature head
        }
    
    def __repr__(self):
        return (
            f"DPTPointmapHead(\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  dec_embed_dim={self.dec_embed_dim},\n"
            f"  spatial_size={self.spatial_size},\n"
            f"  has_dpt={self.has_dpt}\n"
            f")"
        )


class POST3RDecoder(nn.Module):
    """
    Dual-head decoder for POST3R.
    Outputs both pointmap and features from slots.
    
    Architecture:
    - Feature head: Follows SlotContrast's MLPDecoder (MLP + pos_emb + pose_emb)
    - Pointmap head: Follows TTT3R's DPTPts3dPose style (DPT-based multi-scale architecture)
    """
    
    def __init__(
        self,
        slot_dim: int,
        feature_dim: int = 1024,
        n_patches: int = 1024,  # Number of patches from encoder (32x32 for 512x512)
        pose_dim: int = 7,
        pose_embed_dim: int = 64,
        pointmap_resolution: Tuple[int, int] = (288, 512),  # Default to match typical image size
        pointmap_hidden_dims: Tuple[int, ...] = (256, 256, 128),  # Legacy, kept for compatibility
        feature_hidden_dims: Tuple[int, ...] = (512,),  # Hidden dims for feature MLP
        use_dpt_head: bool = True,  # Use DPT-based head by default
        dec_embed_dim: int = 768,  # For DPT
    ):
        """
        Initialize dual-head decoder.
        
        Args:
            slot_dim: Dimension of each slot
            feature_dim: Dimension of encoder features (for reconstruction)
            n_patches: Number of patches from encoder (e.g., 1024 for 32x32)
            pose_dim: Dimension of pose
            pose_embed_dim: Dimension of pose embedding
            pointmap_resolution: Output resolution for pointmap
            pointmap_hidden_dims: Hidden dimensions for pointmap CNN (legacy)
            feature_hidden_dims: Hidden dimensions for feature MLP (SlotContrast-style)
            use_dpt_head: Whether to use DPT-based pointmap head (default: True)
            dec_embed_dim: Decoder embedding dimension for DPT
        """
        super().__init__()
        
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.n_patches = n_patches
        self.use_dpt_head = use_dpt_head
        
        # Store pointmap resolution for use in forward
        self.pointmap_resolution = pointmap_resolution
        
        # Pointmap head - DPT-based (following DPTPts3dPose) or CNN-based fallback
        if use_dpt_head:
            # Calculate spatial size for DPT tokens (assuming patch_size=16)
            # Use pointmap_resolution to calculate spatial size
            # This ensures consistency between initialization and forward pass
            H_tokens = pointmap_resolution[0] // 16
            W_tokens = pointmap_resolution[1] // 16
            
            self.pointmap_head = DPTPointmapHead(
                slot_dim=slot_dim,
                pose_dim=pose_dim,
                dec_embed_dim=dec_embed_dim,
                spatial_size=(H_tokens, W_tokens),
                output_channels=4,  # XYZ + confidence
                feature_dim=256,
                last_dim=128,
            )
        else:
            # Fallback to simple CNN-based decoder
            self.pointmap_head = Decoder3D(
                slot_dim=slot_dim,
                pose_dim=pose_dim,
                pose_embed_dim=pose_embed_dim,
                resolution=pointmap_resolution,
                hidden_dims=pointmap_hidden_dims,
                output_type='pointmap'
            )
        
        # Feature head (SlotContrast-style MLP with pos_emb, NO pose)
        self.feature_head = FeatureHead(
            slot_dim=slot_dim,
            output_dim=feature_dim,
            n_patches=n_patches,
            hidden_dims=feature_hidden_dims,
            activation='relu',
        )
    
    def forward(
        self,
        slots: torch.Tensor,
        pose: torch.Tensor,
        feature_target_shape: Optional[Tuple[int, int]] = None,
        pointmap_target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode slots to pointmap and features.
        
        Args:
            slots: Object slots (B, K, D_slot)
            pose: Camera pose (B, 7)
            feature_target_shape: Target shape for features (H, W). If None, uses default
            pointmap_target_size: Target size for pointmap (H, W). Only for DPT head.
            
        Returns:
            Dictionary with:
            - 'pointmap': Scene pointmap (B, H_p, W_p, 3)
            - 'pointmap_masks': Per-slot masks from pointmap head (B, K, H_p, W_p)
            - 'features': Scene features (B, H_f, W_f, D_feat)
            - 'feature_masks': Per-slot masks from feature head (B, K, H_f, W_f)
        """
        # Decode pointmap
        if self.use_dpt_head:
            # DPT-based head - pass target_size
            if pointmap_target_size is None:
                pointmap_target_size = self.pointmap_resolution  # Use configured resolution
            pointmap_out = self.pointmap_head(slots, pose, target_size=pointmap_target_size)
        else:
            # CNN-based head
            pointmap_out = self.pointmap_head(slots, pose)
        
        # Decode features (no pose)
        if feature_target_shape is None:
            # Use a default feature resolution (typically smaller than pointmap)
            feature_target_shape = (32, 32)  # Default for 512x512 input with patch_size=16
        
        feature_out = self.feature_head(slots, feature_target_shape)
        
        result = {
            'pointmap': pointmap_out['pointmap'],
            'features': feature_out['features'],
            'feature_masks': feature_out['masks'],  # (B, K, H_feat, W_feat)
            'pointmap_masks': pointmap_out['masks'],  # (B, K, H_pointmap, W_pointmap) - now always present
        }
        
        return result
    
    def __repr__(self):
        head_type = "DPT-based" if self.use_dpt_head else "CNN-based"
        return (
            f"POST3RDecoder(\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  pointmap_head_type={head_type},\n"
            f"  use_dpt={self.use_dpt_head}\n"
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
    
    # Test POST3RDecoder
    print("\nTesting POST3RDecoder...")
    
    # Create POST3R decoder
    n_patches = (resolution[0] // 16) * (resolution[1] // 16)
    post3r_decoder = POST3RDecoder(
        slot_dim=slot_dim,
        feature_dim=1024,
        n_patches=n_patches,
        pose_dim=7,
        pose_embed_dim=64,
        pointmap_resolution=resolution,
        pointmap_hidden_dims=(256, 256, 128),
        feature_hidden_dims=(512,),
    )
    
    # Forward pass
    outputs = post3r_decoder(slots, pose)
    print(f"✓ Pointmap shape: {outputs['pointmap'].shape}")
    print(f"✓ Features shape: {outputs['features'].shape}")
    assert outputs['pointmap'].shape == (batch_size, *resolution, 3)
    assert outputs['features'].shape == (batch_size, *resolution, 1024)
    
    print("POST3RDecoder test passed! ✓")


if __name__ == "__main__":
    test_decoder()
