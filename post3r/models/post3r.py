"""
POST3R: Object-Centric 3D Reconstruction with Slot Attention

This is the main model that integrates:
1. TTT3R Backbone (frozen) - extracts features, poses, and 3D pointmaps
2. Recurrent Slot Attention - decomposes scene into object slots
3. 3D Decoder - reconstructs 3D pointmap fr            'gt_features': features,  # (B, T, H_feat, W_feat, D_feat)
            'poses': poses,  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
            'grouping_masks': torch.stack(all_grouping_masks, dim=1) if all_grouping_masks[0] is not None else None,  # (B, T, K, N)
            'pointmap_masks': torch.stack(all_pointmap_masks, dim=1),  # (B, T, K, H, W) - always present now
            'feature_masks': torch.stack(all_feature_masks, dim=1),  # (B, T, K, H, W)
            'confidence': confidence,  # (B, T, H, W) if availablets
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone import TTT3RBackbone
from .slot_attention import RecurrentSlotAttention
from .decoder_3d import POST3RDecoder


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
        use_dinov2: bool = False,  # Use DINOv2 encoder instead of TTT3R encoder
        dinov2_model: str = "vit_base_patch14_dinov2",  # DINOv2 model name
        
        # Slot attention config
        num_slots: int = 8,
        slot_dim: int = 128,
        num_iterations: int = 3,
        mlp_hidden_dim: int = 256,
        
        # Decoder config
        decoder_resolution: Tuple[int, int] = (512, 512),  # Default output resolution
        decoder_hidden_dims: Tuple[int, ...] = (256, 256, 128),
        n_patches: int = 1024,  # Number of patches from encoder (32x32 for 512x512)
        
        # Feature projection
        feature_dim: int = 1024,  # Encoder output dimension (1024 for TTT3R, 768 for DINOv2-base)
    ):
        """
        Initialize POST3R model.
        
        Args:
            ttt3r_checkpoint: Path to pretrained TTT3R checkpoint
            freeze_backbone: Whether to freeze TTT3R backbone
            use_dinov2: Whether to use DINOv2 encoder instead of TTT3R encoder (for ablation)
            dinov2_model: DINOv2 model name (e.g., vit_base_patch14_dinov2)
            num_slots: Number of object slots
            slot_dim: Dimension of each slot
            num_iterations: Number of slot attention iterations
            mlp_hidden_dim: Hidden dimension for slot attention MLP
            decoder_resolution: Output resolution for decoder (default: 512x512)
            decoder_hidden_dims: Hidden dimensions for decoder CNN
            n_patches: Number of patches from encoder (e.g., 1024 for 32x32)
            feature_dim: Dimension of encoder features (1024 for TTT3R, 768 for DINOv2-base)
        """
        super().__init__()
        
        # Configuration
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.use_dinov2 = use_dinov2
        
        # 1. TTT3R Backbone (with optional DINOv2 encoder)
        self.backbone = TTT3RBackbone(
            model_path=ttt3r_checkpoint,
            frozen=freeze_backbone,
            use_dinov2=use_dinov2,
            dinov2_model=dinov2_model,
            feature_dim=feature_dim
        )
        
        # 2. Recurrent Slot Attention (TTT3R-style)
        # Follows TTT3R pattern: register_tokens, _encode_state, _decoder, model_update_type="ttt3r"
        # Now with ROPE support from TTT3R
        self.slot_attention = RecurrentSlotAttention(
            num_slots=num_slots,
            feature_dim=feature_dim,  # 1024 for TTT3R, 768 for DINOv2-base
            slot_dim=slot_dim,         # 128 (desired slot dimension)
            num_iterations=num_iterations,
            mlp_hidden_dim=mlp_hidden_dim,
            model_update_type="ttt3r",  # TTT3R-style confidence-weighted updates
            rope=self.backbone.rope,    # Use TTT3R's ROPE for attention
            slot_pos_type="2d"          # 2D grid positions for slots
        )
        
        # 4. 3D Decoder
        # Dual-head decoder: pointmap head + feature head
        self.decoder = POST3RDecoder(
            slot_dim=slot_dim,
            feature_dim=feature_dim,  # 1024 for TTT3R, 768 for DINOv2-base
            n_patches=n_patches,  # Number of patches from encoder (from config)
            pointmap_resolution=decoder_resolution,
            pointmap_hidden_dims=decoder_hidden_dims,
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
        feature_pos = backbone_output['feature_pos']  # (B, N, 2) for ROPE
        pose = backbone_output['pose']  # (B, 7) [tx, ty, tz, qx, qy, qz, qw]
        gt_pointmap = backbone_output['pointmap']  # (B, H, W, 3)
        confidence = backbone_output.get('confidence', None)
        
        B, H_feat, W_feat, D_feat = features.shape
        
        # 2. Flatten features for slot attention
        # Shape: (B, H_feat, W_feat, D_feat) → (B, N, D_feat) where N = H_feat * W_feat
        features_flat = features.view(B, H_feat * W_feat, D_feat)  # (B, N, 1024)
        
        # 3. Slot attention decomposition with ROPE
        # SlotAttention will handle projection from 1024 → 128 internally
        # feature_pos is (B, N, 2) from TTT3R's patch_embed.position_getter
        slots_output = self.slot_attention(
            features_flat,
            feature_pos=feature_pos,  # Pass positions for ROPE
            prev_slots=self.prev_slots,
            confidence=confidence
        )
        
        slots = slots_output['slots']  # (B, K, D_slot)
        slot_pos = slots_output['slot_pos']  # (B, K, 2) slot positions for ROPE
        attn_weights = slots_output.get('attn_weights', None)
        grouping_masks = slots_output.get('masks', None)  # (B, K, N) - attention masks
        
        # Update memory
        self.prev_slots = slots.detach()
        
        # Get target size for pointmap (use GT pointmap size)
        _, H_gt, W_gt, _ = gt_pointmap.shape
        
        # 4. Decode slots to 3D pointmap and features
        decoder_output = self.decoder(
            slots, 
            pose,
            feature_target_shape=(H_feat, W_feat),  # Match encoder feature resolution
            pointmap_target_size=(H_gt, W_gt)  # Match GT pointmap size
        )
        recon_pointmap = decoder_output['pointmap']  # (B, H_out, W_out, 3)
        recon_features = decoder_output['features']  # (B, H_feat, W_feat, D_feat)
        pointmap_masks = decoder_output['pointmap_masks']  # (B, K, H_out, W_out) - upsampled from feature masks if DPT
        feature_masks = decoder_output['feature_masks']  # (B, K, H_feat, W_feat)
        
        return {
            'slots': slots,
            'recon_pointmap': recon_pointmap,
            'recon_features': recon_features,
            'gt_pointmap': gt_pointmap,
            'gt_features': features,
            'pose': pose,
            'confidence': confidence,
            'attn_weights': attn_weights,
            'grouping_masks': grouping_masks,  # (B, K, N) - from slot attention
            'pointmap_masks': pointmap_masks,  # (B, K, H, W) - from pointmap decoder
            'feature_masks': feature_masks,  # (B, K, H, W) - from feature decoder
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
        feature_pos = backbone_output['feature_pos']  # (B, T, N, 2) for ROPE
        poses = backbone_output['pose']  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
        gt_pointmaps = backbone_output['pointmap']  # (B, T, H, W, 3)
        confidence = backbone_output.get('confidence', None)  # (B, T, H, W) or None
        
        # 2. Process each frame through slot attention and decoder
        # Note: Slot attention is recurrent, so we still need frame-by-frame processing
        all_slots = []
        all_recon_pointmaps = []
        all_recon_features = []
        all_grouping_masks = []
        all_pointmap_masks = []
        all_feature_masks = []
        
        for t in range(T):
            # Extract features for this timestep
            features_t = features[:, t]  # (B, H_feat, W_feat, D_feat)
            feature_pos_t = feature_pos[:, t]  # (B, N, 2) for ROPE
            pose_t = poses[:, t]  # (B, 7) [tx, ty, tz, qx, qy, qz, qw]
            confidence_t = confidence[:, t] if confidence is not None else None  # (B, H, W)
            
            B_t, H_feat, W_feat, D_feat = features_t.shape
            
            # Flatten features
            features_flat = features_t.view(B_t, H_feat * W_feat, D_feat)  # (B, N, 1024)
            
            # Slot attention decomposition with ROPE
            slots_output = self.slot_attention(
                features_flat,
                feature_pos=feature_pos_t,  # Pass positions for ROPE
                prev_slots=self.prev_slots,
                confidence=confidence_t
            )
            
            slots_t = slots_output['slots']  # (B, K, D_slot)
            grouping_masks_t = slots_output.get('masks', None)  # (B, K, N)
            
            # Update memory
            self.prev_slots = slots_t.detach()
            
            # Decode slots to 3D pointmap and features
            decoder_output_t = self.decoder(
                slots_t, 
                pose_t,
                feature_target_shape=(H_feat, W_feat)
            )
            recon_pointmap_t = decoder_output_t['pointmap']  # (B, H_out, W_out, 3)
            recon_features_t = decoder_output_t['features']  # (B, H_feat, W_feat, D_feat)
            pointmap_masks_t = decoder_output_t['pointmap_masks']  # (B, K, H_out, W_out) - upsampled from feature masks if DPT
            feature_masks_t = decoder_output_t['feature_masks']  # (B, K, H_feat, W_feat)
            
            all_slots.append(slots_t)
            all_recon_pointmaps.append(recon_pointmap_t)
            all_recon_features.append(recon_features_t)
            all_grouping_masks.append(grouping_masks_t)
            all_pointmap_masks.append(pointmap_masks_t)
            all_feature_masks.append(feature_masks_t)
        
        # Stack along time dimension
        return {
            'slots': torch.stack(all_slots, dim=1),  # (B, T, K, D)
            'recon_pointmap': torch.stack(all_recon_pointmaps, dim=1),  # (B, T, H_out, W_out, 3)
            'recon_features': torch.stack(all_recon_features, dim=1),  # (B, T, H_feat, W_feat, D_feat)
            'gt_pointmap': gt_pointmaps,  # (B, T, H, W, 3) - already stacked from backbone
            'gt_features': features,  # (B, T, H_feat, W_feat, D_feat)
            'poses': poses,  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
            'grouping_masks': torch.stack(all_grouping_masks, dim=1) if all_grouping_masks[0] is not None else None,  # (B, T, K, N)
            'pointmap_masks': torch.stack(all_pointmap_masks, dim=1) if all_pointmap_masks[0] is not None else None,  # (B, T, K, H, W) - may not exist for DPT
            'feature_masks': torch.stack(all_feature_masks, dim=1),  # (B, T, K, H, W)
            'confidence': confidence,  # (B, T, H, W) if available
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
        
        # Get per-slot pointmaps and masks (using pointmap head's method)
        slot_pointmaps, slot_masks = self.decoder.pointmap_head.get_slot_pointmaps(slots, pose)
        
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
        decoder_resolution=(512, 512)
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
