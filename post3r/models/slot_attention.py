"""
Recurrent Slot Attention following TTT3R State Management Pattern

This module implements object-centric slot attention with TTT3R-style state management:
1. register_tokens: Learnable slot embeddings (like TTT3R's state tokens)
2. _encode_state: Initialize slot states with positional encoding (slot_pos for ROPE)
3. _decoder: Attend to features and decode slots (iterative attention with ROPE)
4. model_update_type="ttt3r": Confidence-weighted slot updates
5. ROPE: Rotary Position Embedding for attention (reuses TTT3R's Attention/CrossAttention)

Integration with TTT3R:
-----------------------
The slot attention receives features directly from TTT3R's _encode_image:

    # In POST3R model:
    feat, pos, _ = ttt3r_backbone._encode_image(img, true_shape)
    # feat: (B, N, D) - encoded image features
    # pos: (B, N, 2) - 2D positions from patch_embed for ROPE
    
    # Apply slot attention with ROPE
    output = slot_attention(feat, feature_pos=pos, prev_slots=prev_slots)
    slots = output['slots']         # (B, K, D_slot)
    slot_pos = output['slot_pos']   # (B, K, 2) for ROPE

Key differences from standard slot attention:
- Uses TTT3R's CrossAttention with ROPE instead of dot-product attention
- Slot positions (slot_pos) generated like TTT3R's state_pos (1D or 2D grid)
- Feature positions (feature_pos) come from TTT3R's patch_embed.position_getter
- Confidence-weighted updates for recurrent processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import sys
import os

# Import TTT3R's Attention and CrossAttention blocks
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/ttt3r/src'))
from dust3r.blocks import Attention, CrossAttention


class RecurrentSlotAttention(nn.Module):
    """
    Recurrent Slot Attention following TTT3R's state management pattern.
    
    Architecture mirrors TTT3R's approach:
    - register_tokens: Learnable slot embeddings
    - _encode_state: Initialize slots with position encoding (slot_pos for ROPE)
    - _decoder: Iterative attention with ROPE (reuses TTT3R's Attention/CrossAttention)
    - TTT3R-style confidence-weighted updates
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        num_iterations: int = 3,
        hidden_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        model_update_type: str = "ttt3r",
        epsilon: float = 1e-8,
        rope=None,  # ROPE from TTT3R (e.g., RoPE2D)
        slot_pos_type: str = "2d",  # "1d", "2d", or "none"
    ):
        """
        Initialize recurrent slot attention.
        
        Args:
            num_slots: Number of object slots (like state_size in TTT3R)
            slot_dim: Dimension of each slot (like enc_embed_dim in TTT3R)
            feature_dim: Dimension of input features
            num_iterations: Number of attention iterations
            hidden_dim: Hidden dimension for attention (default: slot_dim)
            mlp_hidden_dim: Hidden dimension for MLP (default: slot_dim)
            model_update_type: "ttt3r" for confidence-weighted updates
            epsilon: Small constant for numerical stability
            rope: ROPE module from TTT3R (e.g., RoPE2D(100))
            slot_pos_type: Position encoding type for slots ("1d", "2d", or "none")
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.model_update_type = model_update_type
        self.epsilon = epsilon
        self.slot_pos_type = slot_pos_type
        
        hidden_dim = hidden_dim or slot_dim
        mlp_hidden_dim = mlp_hidden_dim or slot_dim
        
        # TTT3R-style: register_tokens for slot initialization
        self.register_tokens = nn.Embedding(num_slots, slot_dim)
        
        # Layer normalization
        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        # Use TTT3R's CrossAttention for slot-to-feature attention with ROPE
        self.rope = rope.float() if rope is not None else None
        self.cross_attn = CrossAttention(
            dim=slot_dim,
            rope=self.rope,
            num_heads=slot_dim // 64,  # Follow TTT3R's convention
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0
        )
        
        # Project features to slot_dim if needed
        if feature_dim != slot_dim:
            self.feature_proj = nn.Linear(feature_dim, slot_dim)
        else:
            self.feature_proj = nn.Identity()
        
        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )
        
        # For TTT3R-style confidence-weighted updates
        if model_update_type == "ttt3r":
            self.confidence_proj = nn.Linear(slot_dim, 1)
    
    def _encode_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize slot states with positional encoding (following TTT3R's _encode_state pattern).
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            slots: Initial slot states (batch_size, num_slots, slot_dim)
            slot_pos: Slot positions for ROPE (batch_size, num_slots, 2)
        """
        # Use register_tokens to initialize slots (like TTT3R's state initialization)
        slot_indices = torch.arange(self.num_slots, device=device)
        slots = self.register_tokens(slot_indices)  # (num_slots, slot_dim)
        slots = slots.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_slots, slot_dim)
        
        # Add small noise for diversity
        slots = slots + torch.randn_like(slots) * 0.01
        
        # Generate slot_pos for ROPE (following TTT3R's state_pos pattern)
        # NOTE: ROPE kernel expects positions as Long (int64) type, not Float
        if self.slot_pos_type == "1d":
            # 1D position: [0, 0], [1, 1], [2, 2], ...
            slot_pos = torch.tensor(
                [[i, i] for i in range(self.num_slots)],
                dtype=torch.long,  # ROPE expects Long type
                device=device
            )[None].expand(batch_size, -1, -1).contiguous()  # (B, num_slots, 2)
        elif self.slot_pos_type == "2d":
            # 2D grid position: arrange slots in a grid
            width = int(self.num_slots ** 0.5)
            width = width + 1 if width % 2 == 1 else width
            slot_pos = torch.tensor(
                [[i // width, i % width] for i in range(self.num_slots)],
                dtype=torch.long,  # ROPE expects Long type
                device=device
            )[None].expand(batch_size, -1, -1).contiguous()  # (B, num_slots, 2)
        else:  # "none"
            slot_pos = None
        
        return slots, slot_pos
    
    def _compute_attention(
        self, 
        slots: torch.Tensor,
        slot_pos: torch.Tensor,
        features: torch.Tensor,
        feature_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using TTT3R's CrossAttention with ROPE.
        
        Args:
            slots: Slot representations (B, K, D_slot)
            slot_pos: Slot positions for ROPE (B, K, 2)
            features: Input features (B, N, D_feat)
            feature_pos: Feature positions for ROPE (B, N, 2)
            
        Returns:
            updates: Aggregated features (B, K, D_slot)
            attn_weights: Not returned by CrossAttention, return None
        """
        B, K, _ = slots.shape
        _, N, _ = features.shape
        
        # Normalize
        slots_norm = self.norm_slots(slots)
        features_norm = self.norm_inputs(features)
        
        # Project features to slot_dim if needed
        features_proj = self.feature_proj(features_norm)  # (B, N, D_slot)
        
        # Use TTT3R's CrossAttention with ROPE
        # query=slots, key=features, value=features, qpos=slot_pos, kpos=feature_pos
        updates = self.cross_attn(
            query=slots_norm,
            key=features_proj,
            value=features_proj,
            qpos=slot_pos,
            kpos=feature_pos
        )  # (B, K, D_slot)
        
        return updates, None  # CrossAttention doesn't return attn weights by default
    
    def _slot_attention_iteration(
        self,
        slots: torch.Tensor,
        slot_pos: torch.Tensor,
        features: torch.Tensor,
        feature_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single iteration of slot attention with ROPE.
        
        Args:
            slots: Current slot states (B, K, D_slot)
            slot_pos: Slot positions for ROPE (B, K, 2)
            features: Input features (B, N, D_feat)
            feature_pos: Feature positions for ROPE (B, N, 2)
            
        Returns:
            updated_slots: Updated slot states (B, K, D_slot)
            attn_weights: Attention weights (None for now)
        """
        B, K, D_slot = slots.shape
        
        # Compute attention using TTT3R's CrossAttention with ROPE
        updates, attn_weights = self._compute_attention(
            slots, slot_pos, features, feature_pos
        )
        
        # Update slots using GRU
        slots_flat = slots.view(B * K, D_slot)
        updates_flat = updates.view(B * K, D_slot)
        
        slots_updated = self.gru(updates_flat, slots_flat)
        slots_updated = slots_updated.view(B, K, D_slot)
        
        # Apply MLP
        slots_refined = slots_updated + self.mlp(self.norm_mlp(slots_updated))
        
        return slots_refined, attn_weights
    
    def _decoder(
        self,
        slots: torch.Tensor,
        slot_pos: torch.Tensor,
        features: torch.Tensor,
        feature_pos: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decode slots from features using iterative attention with ROPE (TTT3R's _decoder pattern).
        
        Args:
            slots: Initial slot states (B, K, D_slot)
            slot_pos: Slot positions for ROPE (B, K, 2)
            features: Encoded features from TTT3R's _encode_image (B, N, D_feat)
            feature_pos: Feature positions from TTT3R's patch_embed (B, N, 2)
            prev_slots: Previous slot states for recurrent updates
            confidence: Optional confidence map for TTT3R-style updates
            
        Returns:
            final_slots: Final slot states (B, K, D_slot)
            attn_weights: Final attention weights (None for now)
            slot_confidence: Per-slot confidence (B, K) if using TTT3R update
        """
        # Iterative attention refinement with ROPE
        current_slots = slots
        attn_weights = None
        for i in range(self.num_iterations):
            current_slots, attn_weights = self._slot_attention_iteration(
                current_slots, slot_pos, features, feature_pos
            )
        
        # TTT3R-style confidence-weighted update with previous slots
        slot_confidence = None
        if prev_slots is not None and self.model_update_type == "ttt3r":
            # Compute slot confidence from slot features
            slot_confidence = torch.sigmoid(
                self.confidence_proj(current_slots).squeeze(-1)
            )  # (B, K)
            
            # Update with confidence weighting
            update_weight = slot_confidence.unsqueeze(-1)  # (B, K, 1)
            current_slots = current_slots * update_weight + prev_slots * (1 - update_weight)
        
        return current_slots, attn_weights, slot_confidence
    
    def forward(
        self,
        features: torch.Tensor,
        feature_pos: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through slot attention with ROPE.
        
        This follows TTT3R's pattern where features come from _encode_image:
            feat, pos = backbone._encode_image(img, true_shape)
            # feat: (B, N, D) - encoded features
            # pos: (B, N, 2) - positions from patch_embed for ROPE
        
        Args:
            features: Encoded features from TTT3R's _encode_image (B, N, D)
            feature_pos: Feature positions from TTT3R's patch_embed (B, N, 2)
            prev_slots: Previous slots for recurrent processing (B, K, D_slot)
            confidence: Optional confidence map for TTT3R-style updates
            
        Returns:
            Dictionary with:
                - slots: Slot representations (B, K, D_slot)
                - slot_pos: Slot positions (B, K, 2)
                - attn_weights: Attention weights (None for now)
                - slot_confidence: Per-slot confidence (B, K) if using TTT3R
        """
        B = features.shape[0]
        device = features.device
        
        # Initialize slots and slot_pos using _encode_state (TTT3R pattern)
        if prev_slots is None:
            slots, slot_pos = self._encode_state(B, device)
        else:
            # Use previous slots as initialization, regenerate slot_pos
            slots = prev_slots
            _, slot_pos = self._encode_state(B, device)
        
        # Decode slots from features using _decoder with ROPE (TTT3R pattern)
        final_slots, attn_weights, slot_confidence = self._decoder(
            slots, slot_pos, features, feature_pos, prev_slots, confidence
        )
        
        output = {
            'slots': final_slots,
            'slot_pos': slot_pos,
            'attn_weights': attn_weights,
        }
        
        if slot_confidence is not None:
            output['slot_confidence'] = slot_confidence
        
        return output
    
    def __repr__(self):
        return (
            f"RecurrentSlotAttention(\n"
            f"  num_slots={self.num_slots},\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  num_iterations={self.num_iterations},\n"
            f"  model_update_type='{self.model_update_type}',\n"
            f"  rope={'RoPE' if self.rope is not None else 'None'},\n"
            f"  slot_pos_type='{self.slot_pos_type}'\n"
            f")"
        )


# Test function
def test_slot_attention():
    """Test the recurrent slot attention module with ROPE."""
    print("Testing Recurrent Slot Attention (TTT3R-style with ROPE)...")
    
    # Parameters
    batch_size = 2
    num_slots = 8
    slot_dim = 768  # Match TTT3R's enc_embed_dim
    feature_dim = 768
    num_features = 576  # 24x24 patches
    
    # Create ROPE (import from TTT3R)
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/ttt3r/src'))
    from models.pos_embed import RoPE2D
    rope = RoPE2D(100)
    
    # Create module with TTT3R-style updates and ROPE
    slot_attn = RecurrentSlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        feature_dim=feature_dim,
        num_iterations=3,
        model_update_type="ttt3r",
        rope=rope,
        slot_pos_type="2d"
    )
    
    print(f"✓ Created slot attention module:\n{slot_attn}")
    
    # Simulate TTT3R's _encode_image output
    # In TTT3R: feat, pos = backbone._encode_image(img, true_shape)
    features = torch.randn(batch_size, num_features, feature_dim)  # (B, N, D)
    
    # Create feature positions (from TTT3R's patch_embed)
    # This simulates what patch_embed.position_getter returns
    # NOTE: ROPE expects Long (int64) type positions
    H = W = 24
    y_pos = torch.arange(H)
    x_pos = torch.arange(W)
    yy, xx = torch.meshgrid(y_pos, x_pos, indexing='ij')
    feature_pos = torch.stack([yy, xx], dim=-1).view(1, num_features, 2)
    feature_pos = feature_pos.expand(batch_size, -1, -1)  # (B, N, 2) - Long type
    
    # First frame (no previous slots) - uses _encode_state
    output_t0 = slot_attn(features, feature_pos=feature_pos, prev_slots=None)
    slots_t0 = output_t0['slots']
    slot_pos = output_t0['slot_pos']
    print(f"✓ First frame slots shape: {slots_t0.shape}")
    print(f"✓ Slot positions shape: {slot_pos.shape}")
    assert slots_t0.shape == (batch_size, num_slots, slot_dim)
    assert slot_pos.shape == (batch_size, num_slots, 2)
    
    # Second frame (with previous slots) - uses _decoder with TTT3R updates
    features_t1 = torch.randn(batch_size, num_features, feature_dim)
    output_t1 = slot_attn(features_t1, feature_pos=feature_pos, prev_slots=slots_t0)
    slots_t1 = output_t1['slots']
    print(f"✓ Second frame slots shape: {slots_t1.shape}")
    if 'slot_confidence' in output_t1:
        print(f"✓ TTT3R slot confidence shape: {output_t1['slot_confidence'].shape}")
    assert slots_t1.shape == (batch_size, num_slots, slot_dim)
    
    print("\nRecurrent Slot Attention (TTT3R-style with ROPE) test passed! ✓")
    print("\nUsage example:")
    print("  # Get features from TTT3R backbone")
    print("  feat, pos, _ = backbone._encode_image(img, true_shape)")
    print("  # Apply slot attention")
    print("  output = slot_attn(feat, feature_pos=pos, prev_slots=prev_slots)")
    print("  slots = output['slots']  # (B, K, D)")


if __name__ == "__main__":
    test_slot_attention()
