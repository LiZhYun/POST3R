"""
POST3R: Object-Centric 3D Reconstruction with Slot Attention

This is the main model that integrates:
1. TTT3R Backbone (frozen) - extracts features, poses, and 3D pointmaps
2. Recurrent Slot Attention - decomposes scene into object slots
3. 3D Decoder - reconstructs 3D pointmap from slots
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone import TTT3RBackbone
from .slot_attention import RecurrentSlotAttention
from .decoder_3d import Decoder3D


class POST3R(nn.Module):
    """
    POST3R: Object-Centric 3D Reconstruction Model
    
    Architecture:
    - TTT3R Backbone (frozen): Extracts geometric features
    - Slot Attention: Decomposes features into object-centric slots
    - 3D Decoder: Reconstructs 3D pointmap from slots
    """
    
    def __init__(
        self,
        # Backbone config
        ttt3r_checkpoint: str,
        freeze_backbone: bool = True,
        
        # Slot attention config
        num_slots: int = 8,
        slot_dim: int = 128,
        num_iterations: int = 3,
        mlp_hidden_dim: int = 256,
        use_ttt3r_update: bool = True,
        
        # Decoder config
        decoder_resolution: Tuple[int, int] = (64, 64),
        decoder_hidden_dims: Tuple[int, ...] = (256, 256, 128),
        
        # Feature projection
        feature_dim: int = 1024,  # TTT3R encoder output dimension (enc_embed_dim)
    ):
        """
        Initialize POST3R model.
        
        Args:
            ttt3r_checkpoint: Path to pretrained TTT3R checkpoint
            freeze_backbone: Whether to freeze TTT3R backbone
            num_slots: Number of object slots
            slot_dim: Dimension of each slot
            num_iterations: Number of slot attention iterations
            mlp_hidden_dim: Hidden dimension for slot attention MLP
            use_ttt3r_update: Use TTT3R-style confidence weighting
            decoder_resolution: Output resolution for decoder
            decoder_hidden_dims: Hidden dimensions for decoder CNN
            feature_dim: Dimension of TTT3R features
        """
        super().__init__()
        
        # Configuration
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        
        # 1. TTT3R Backbone
        self.backbone = TTT3RBackbone(
            model_path=ttt3r_checkpoint,
            frozen=freeze_backbone
        )
        
        # 2. Recurrent Slot Attention
        # Let SlotAttention handle feature projection internally (1024 → 128)
        # This is cleaner and avoids redundant projections
        self.slot_attention = RecurrentSlotAttention(
            num_slots=num_slots,
            feature_dim=feature_dim,  # 1024 (TTT3R encoder output)
            slot_dim=slot_dim,         # 128 (desired slot dimension)
            num_iterations=num_iterations,
            mlp_hidden_dim=mlp_hidden_dim,
            use_ttt3r_update=use_ttt3r_update
        )
        
        # 4. 3D Decoder
        self.decoder = Decoder3D(
            slot_dim=slot_dim,
            resolution=decoder_resolution,
            hidden_dims=decoder_hidden_dims,
            output_type='pointmap'
        )
        
        # Memory for recurrent processing
        self.prev_slots = None
    
    def reset_memory(self):
        """Reset recurrent memory (call at start of new sequence)."""
        self.prev_slots = None
        self.backbone.reset_memory()
    
    def forward(
        self,
        images: torch.Tensor,
        reset_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single frame.
        
        Args:
            images: Input images (B, 3, H, W)
            reset_memory: Whether to reset recurrent memory
            
        Returns:
            Dictionary with:
            - 'slots': Object slots (B, K, D)
            - 'recon_pointmap': Reconstructed 3D pointmap (B, H_out, W_out, 3)
            - 'gt_pointmap': Ground truth pointmap from TTT3R (B, H, W, 3)
            - 'pose': Camera pose (B, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            - 'features': Backbone features (B, H_feat, W_feat, D_feat)
            - 'confidence': TTT3R confidence (B, H, W) if available
        """
        if reset_memory:
            self.reset_memory()
        
        # 1. Extract features from TTT3R backbone
        backbone_output = self.backbone(images)
        
        features = backbone_output['features']  # (B, H_feat, W_feat, D_feat)
        pose = backbone_output['pose']  # (B, 7) [tx, ty, tz, qx, qy, qz, qw]
        gt_pointmap = backbone_output['pointmap']  # (B, H, W, 3)
        confidence = backbone_output.get('confidence', None)
        
        B, H_feat, W_feat, D_feat = features.shape
        
        # 2. Flatten features for slot attention
        # Shape: (B, H_feat, W_feat, D_feat) → (B, N, D_feat) where N = H_feat * W_feat
        features_flat = features.view(B, H_feat * W_feat, D_feat)  # (B, N, 1024)
        
        # 3. Slot attention decomposition
        # SlotAttention will handle projection from 1024 → 128 internally
        slots_output = self.slot_attention(
            features_flat,
            prev_slots=self.prev_slots,
            confidence=confidence
        )
        
        slots = slots_output['slots']  # (B, K, D_slot)
        attn_weights = slots_output.get('attn_weights', None)
        grouping_masks = slots_output.get('masks', None)  # (B, K, N) - attention masks
        
        # Update memory
        self.prev_slots = slots.detach()
        
        # 4. Decode slots to 3D pointmap
        decoder_output = self.decoder(slots, pose)
        recon_pointmap = decoder_output['pointmap']  # (B, H_out, W_out, 3)
        decoder_masks = decoder_output['masks']  # (B, K, H_out, W_out) - decoder masks
        
        return {
            'slots': slots,
            'recon_pointmap': recon_pointmap,
            'gt_pointmap': gt_pointmap,
            'pose': pose,
            'features': features,
            'confidence': confidence,
            'attn_weights': attn_weights,
            'grouping_masks': grouping_masks,  # (B, K, N) - from slot attention
            'decoder_masks': decoder_masks,  # (B, K, H, W) - from decoder
        }
    
    def forward_sequence(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for video sequence using TTT3R's batched API.
        
        IMPORTANT: This now processes the ENTIRE sequence in one TTT3R call,
        which is the correct way to use forward_recurrent_lighter.
        
        Args:
            images: Input video (B, T, 3, H, W)
            
        Returns:
            Dictionary with:
            - 'slots': Object slots (B, T, K, D)
            - 'recon_pointmap': Reconstructed pointmaps (B, T, H_out, W_out, 3)
            - 'gt_pointmap': Ground truth pointmaps (B, T, H, W, 3)
            - 'poses': Camera poses (B, T, 7) in [tx, ty, tz, qx, qy, qz, qw] format
        """
        B, T = images.shape[:2]
        
        # Reset memory at start of sequence
        self.reset_memory()
        
        # 1. Process ENTIRE sequence through TTT3R in one call (CORRECT)
        # This is much more efficient and how TTT3R is designed to work
        backbone_output = self.backbone.forward_sequence(images)
        
        # Extract outputs: all have shape (B, T, ...)
        features = backbone_output['features']  # (B, T, H_feat, W_feat, D_feat)
        poses = backbone_output['pose']  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
        gt_pointmaps = backbone_output['pointmap']  # (B, T, H, W, 3)
        confidence = backbone_output.get('confidence', None)  # (B, T, H, W) or None
        
        # 2. Process each frame through slot attention and decoder
        # Note: Slot attention is recurrent, so we still need frame-by-frame processing
        all_slots = []
        all_recon_pointmaps = []
        all_grouping_masks = []
        all_decoder_masks = []
        
        for t in range(T):
            # Extract features for this timestep
            features_t = features[:, t]  # (B, H_feat, W_feat, D_feat)
            pose_t = poses[:, t]  # (B, 7) [tx, ty, tz, qx, qy, qz, qw]
            confidence_t = confidence[:, t] if confidence is not None else None  # (B, H, W)
            
            B_t, H_feat, W_feat, D_feat = features_t.shape
            
            # Flatten features
            features_flat = features_t.view(B_t, H_feat * W_feat, D_feat)  # (B, N, 1024)
            
            # Slot attention decomposition
            slots_output = self.slot_attention(
                features_flat,
                prev_slots=self.prev_slots,
                confidence=confidence_t
            )
            
            slots_t = slots_output['slots']  # (B, K, D_slot)
            grouping_masks_t = slots_output.get('masks', None)  # (B, K, N)
            
            # Update memory
            self.prev_slots = slots_t.detach()
            
            # Decode slots to 3D pointmap
            decoder_output_t = self.decoder(slots_t, pose_t)
            recon_pointmap_t = decoder_output_t['pointmap']  # (B, H_out, W_out, 3)
            decoder_masks_t = decoder_output_t['masks']  # (B, K, H_out, W_out)
            
            all_slots.append(slots_t)
            all_recon_pointmaps.append(recon_pointmap_t)
            all_grouping_masks.append(grouping_masks_t)
            all_decoder_masks.append(decoder_masks_t)
        
        # Stack along time dimension
        return {
            'slots': torch.stack(all_slots, dim=1),  # (B, T, K, D)
            'recon_pointmap': torch.stack(all_recon_pointmaps, dim=1),  # (B, T, H_out, W_out, 3)
            'gt_pointmap': gt_pointmaps,  # (B, T, H, W, 3) - already stacked from backbone
            'poses': poses,  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
            'grouping_masks': torch.stack(all_grouping_masks, dim=1) if all_grouping_masks[0] is not None else None,  # (B, T, K, N)
            'decoder_masks': torch.stack(all_decoder_masks, dim=1),  # (B, T, K, H, W)
        }
    
    def get_slot_decomposition(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed slot-wise decomposition (for visualization).
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Dictionary with per-slot information:
            - 'slots': Object slots (B, K, D)
            - 'slot_pointmaps': Per-slot pointmaps (B, K, H, W, 3)
            - 'slot_masks': Per-slot alpha masks (B, K, H, W)
            - 'attn_maps': Attention maps (B, K, H_feat, W_feat)
        """
        # Forward pass
        output = self.forward(images, reset_memory=False)
        
        slots = output['slots']
        pose = output['pose']
        attn_weights = output.get('attn_weights', None)
        
        # Get per-slot pointmaps and masks
        slot_pointmaps, slot_masks = self.decoder.get_slot_pointmaps(slots, pose)
        
        # Reshape attention weights to spatial
        if attn_weights is not None:
            B, K, HW = attn_weights.shape
            H_feat = W_feat = int(HW ** 0.5)
            attn_maps = attn_weights.view(B, K, H_feat, W_feat)
        else:
            attn_maps = None
        
        return {
            'slots': slots,
            'slot_pointmaps': slot_pointmaps,
            'slot_masks': slot_masks,
            'attn_maps': attn_maps,
        }
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (
            f"POST3R(\n"
            f"  num_slots={self.num_slots},\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )


# Test function
def test_post3r():
    """Test the full POST3R model."""
    print("Testing POST3R Model...")
    
    # Note: This test requires a real TTT3R checkpoint
    # Update the path to your checkpoint before running
    
    print("WARNING: This test requires a real TTT3R checkpoint.")
    print("Please update the checkpoint path and ensure the model is available.")
    print("Skipping test...")
    return
    
    # Parameters
    batch_size = 2
    time_steps = 4
    height, width = 224, 224
    
    # Create model with real checkpoint path
    checkpoint_path = 'submodules/ttt3r/src/cut3r_512_dpt_4_64.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please download TTT3R checkpoint first.")
        return
    
    model = POST3R(
        ttt3r_checkpoint=checkpoint_path,
        num_slots=8,
        slot_dim=128,
        decoder_resolution=(64, 64)
    )
    
    print(f"\n{model}")
    
    # Test single frame
    print("\nTesting single frame...")
    images = torch.randn(batch_size, 3, height, width)
    
    output = model(images, reset_memory=True)
    
    print(f"✓ Slots shape: {output['slots'].shape}")
    print(f"✓ Reconstructed pointmap shape: {output['recon_pointmap'].shape}")
    print(f"✓ GT pointmap shape: {output['gt_pointmap'].shape}")
    print(f"✓ Pose shape: {output['pose'].shape}")
    
    # Test sequence
    print("\nTesting sequence...")
    video = torch.randn(batch_size, time_steps, 3, height, width)
    
    seq_output = model.forward_sequence(video)
    
    print(f"✓ Sequence slots shape: {seq_output['slots'].shape}")
    print(f"✓ Sequence pointmaps shape: {seq_output['recon_pointmap'].shape}")
    
    assert seq_output['slots'].shape == (batch_size, time_steps, 8, 128)
    
    # Test slot decomposition
    print("\nTesting slot decomposition...")
    decomp = model.get_slot_decomposition(images)
    
    print(f"✓ Per-slot pointmaps shape: {decomp['slot_pointmaps'].shape}")
    print(f"✓ Per-slot masks shape: {decomp['slot_masks'].shape}")
    
    print("\nPOST3R model test passed! ✓")


if __name__ == "__main__":
    test_post3r()
