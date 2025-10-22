"""
Recurrent Slot Attention with TTT3R-inspired Updates

This module implements object-centric slot attention with:
1. Iterative attention mechanism (from Slot Attention paper)
2. TTT3R-inspired confidence-weighted state updates
3. Recurrent processing for temporal consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlotInitializer(nn.Module):
    """Initialize slots for slot attention."""
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        method: str = 'learned',
    ):
        """
        Initialize slot initializer.
        
        Args:
            num_slots: Number of slots
            slot_dim: Dimension of each slot
            method: Initialization method ('learned' or 'random')
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.method = method
        
        if method == 'learned':
            # Learnable slot initialization
            self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
            self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        elif method == 'random':
            # Random Gaussian initialization
            pass
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Initialize slots for a batch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial slots (batch_size, num_slots, slot_dim)
        """
        if self.method == 'learned':
            # Sample from learned Gaussian
            mu = self.slots_mu.expand(batch_size, -1, -1)
            sigma = self.slots_log_sigma.exp().expand(batch_size, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:  # random
            slots = torch.randn(batch_size, self.num_slots, self.slot_dim)
            slots = slots.to(self.slots_mu.device if hasattr(self, 'slots_mu') else 'cpu')
        
        return slots


class RecurrentSlotAttention(nn.Module):
    """
    Recurrent Slot Attention with TTT3R-inspired updates.
    
    This implements:
    1. Standard slot attention mechanism
    2. Confidence-weighted learning rate (from TTT3R)
    3. Recurrent state management using previous slots
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        num_iterations: int = 3,
        hidden_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        use_ttt3r_update: bool = True,
        epsilon: float = 1e-8,
    ):
        """
        Initialize recurrent slot attention.
        
        Args:
            num_slots: Number of object slots
            slot_dim: Dimension of each slot
            feature_dim: Dimension of input features
            num_iterations: Number of attention iterations
            hidden_dim: Hidden dimension for attention (default: slot_dim)
            mlp_hidden_dim: Hidden dimension for MLP (default: slot_dim)
            use_ttt3r_update: Whether to use TTT3R-style updates
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.use_ttt3r_update = use_ttt3r_update
        self.epsilon = epsilon
        
        hidden_dim = hidden_dim or slot_dim
        mlp_hidden_dim = mlp_hidden_dim or slot_dim
        
        # Slot initializer
        self.slot_initializer = SlotInitializer(num_slots, slot_dim, method='learned')
        
        # Layer normalization
        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        # Attention parameters
        # Linear projections for keys, queries, values
        self.project_q = nn.Linear(slot_dim, hidden_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)
        
        # Attention scaling
        self.scale = hidden_dim ** -0.5
        
        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )
    
    def _compute_attention(
        self, 
        slots: torch.Tensor, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights between slots and features.
        
        Args:
            slots: Slot representations (B, K, D_slot)
            features: Input features (B, N, D_feat)
            
        Returns:
            Tuple of:
            - attn_weights: Attention weights (B, K, N)
            - confidence: Confidence scores per slot (B, K)
        """
        B, K, _ = slots.shape
        B, N, _ = features.shape
        
        # Normalize
        slots_norm = self.norm_slots(slots)
        features_norm = self.norm_inputs(features)
        
        # Project to queries, keys, values
        q = self.project_q(slots_norm)  # (B, K, H)
        k = self.project_k(features_norm)  # (B, N, H)
        v = self.project_v(features_norm)  # (B, N, D_slot)
        
        # Compute attention logits
        attn_logits = torch.einsum('bkh,bnh->bkn', q, k) * self.scale  # (B, K, N)
        
        # Softmax over slots (each feature attends to all slots)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, K, N)
        
        # Compute confidence as max attention per slot
        confidence = attn_weights.max(dim=-1)[0]  # (B, K)
        
        return attn_weights, confidence
    

    
    def _slot_attention_iteration(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single iteration of slot attention.
        
        Args:
            slots: Current slots (B, K, D_slot)
            features: Input features (B, N, D_feat)
            
        Returns:
            Tuple of:
            - Updated slots (B, K, D_slot)
            - Confidence scores (B, K)
        """
        B, K, D = slots.shape
        B, N, _ = features.shape
        
        # Compute attention
        attn_weights, confidence = self._compute_attention(slots, features)
        
        # Normalize attention weights over features (sum to 1)
        attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # Weighted sum of values
        features_norm = self.norm_inputs(features)
        v = self.project_v(features_norm)  # (B, N, D_slot)
        updates = torch.einsum('bkn,bnd->bkd', attn_normalized, v)  # (B, K, D_slot)
        
        # GRU update
        slots_flat = slots.reshape(B * K, D)
        updates_flat = updates.reshape(B * K, D)
        slots_new = self.gru(updates_flat, slots_flat)
        slots_new = slots_new.reshape(B, K, D)
        
        # MLP refinement
        slots_refined = slots_new + self.mlp(self.norm_mlp(slots_new))
        
        return slots_refined, confidence
    
    def _ttt3r_update(
        self,
        slots_new: torch.Tensor,
        slots_prev: torch.Tensor,
        attn_weights: torch.Tensor,
        is_first_frame: bool = False,
    ) -> torch.Tensor:
        """
        TTT3R-style update operation using weighted blending.
        
        In TTT3R: state_feat = new_state_feat * α + state_feat * (1 - α)
        where α = sigmoid(mean(cross_attention(state, image)))
        
        This blends the new slots with previous slots based on attention confidence.
        
        Args:
            slots_new: New slots from slot attention (B, K, D)
            slots_prev: Slots from previous frame (B, K, D)
            attn_weights: Attention weights from slot attention (B, K, N)
            is_first_frame: Whether this is the first frame (full update if True)
            
        Returns:
            Updated slots (B, K, D)
        """
        # For first frame, use new slots directly
        if is_first_frame:
            return slots_new
        
        # TTT3R update: compute update mask from cross-attention
        # attn_weights shape: (B, K, N) where K is slots, N is features
        # This is analogous to cross_attn_state in TTT3R: (nstate, nimg, ...)
        
        # Compute mean attention per slot (analogous to mean over layers and heads)
        # In TTT3R: state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
        slot_query_feat_key = attn_weights.mean(dim=-1)  # (B, K) - mean over features
        
        # Apply sigmoid to get update mask
        # In TTT3R: update_mask1 = update_mask * sigmoid(state_query_img_key) * 1.0
        update_mask = torch.sigmoid(slot_query_feat_key).unsqueeze(-1)  # (B, K, 1)
        
        # Weighted blending: S_t = S_new * update_mask + S_prev * (1 - update_mask)
        slots_updated = slots_new * update_mask + slots_prev * (1 - update_mask)
        
        return slots_updated
    
    def forward(
        self,
        features: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through recurrent slot attention.
        
        Args:
            features: Input features (B, N, D_feat) or (B, H, W, D_feat)
            prev_slots: Previous frame slots (B, K, D_slot), None for initialization
            confidence: Confidence map (B, H, W) - optional, currently unused
            
        Returns:
            Dictionary with:
            - 'slots': Current frame slots (B, K, D_slot)
            - 'attn_weights': Final attention weights (B, K, N)
        """
        B = features.shape[0]
        
        # Reshape features if spatial
        if features.ndim == 4:
            # (B, H, W, D) -> (B, H*W, D)
            B, H, W, D = features.shape
            features = features.reshape(B, H * W, D)
        
        # Initialize slots if no previous slots
        if prev_slots is None:
            slots = self.slot_initializer(B)
            slots = slots.to(features.device)
        else:
            slots = prev_slots
        
        # Iterative slot attention refinement
        final_pre_norm_attn = None
        for i in range(self.num_iterations):
            slots, slot_confidence = self._slot_attention_iteration(slots, features)
        
        # Get final attention weights for TTT3R update and masks
        # This is the pre-normalized attention (after softmax over slots)
        # In SlotContrast: pre_norm_attn = torch.softmax(dots, dim=1)
        attn_weights, _ = self._compute_attention(slots, features)  # (B, K, N)
        final_pre_norm_attn = attn_weights  # This is the grouping mask
        
        # Apply TTT3R-style update if using previous slots
        if self.use_ttt3r_update and prev_slots is not None:
            slots = self._ttt3r_update(slots, prev_slots, attn_weights, is_first_frame=False)
        elif self.use_ttt3r_update and prev_slots is None:
            # First frame - no blending needed, but keep for consistency
            slots = self._ttt3r_update(slots, slots, attn_weights, is_first_frame=True)
        
        return {
            'slots': slots,
            'attn_weights': attn_weights,
            'masks': final_pre_norm_attn,  # (B, K, N) - grouping masks (SlotContrast format)
        }
    
    def __repr__(self):
        return (
            f"RecurrentSlotAttention(\n"
            f"  num_slots={self.num_slots},\n"
            f"  slot_dim={self.slot_dim},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  num_iterations={self.num_iterations},\n"
            f"  use_ttt3r_update={self.use_ttt3r_update}\n"
            f")"
        )


# Test function
def test_slot_attention():
    """Test the recurrent slot attention module."""
    print("Testing Recurrent Slot Attention...")
    
    # Parameters
    batch_size = 2
    num_slots = 8
    slot_dim = 128
    feature_dim = 384
    num_features = 576  # 24x24 patches
    
    # Create module
    slot_attn = RecurrentSlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        feature_dim=feature_dim,
        num_iterations=3,
        use_ttt3r_update=True
    )
    
    # Test with random features
    features = torch.randn(batch_size, num_features, feature_dim)
    
    # First frame (no previous slots)
    output_t0 = slot_attn(features, prev_slots=None)
    slots_t0 = output_t0['slots']
    print(f"✓ First frame slots shape: {slots_t0.shape}")
    print(f"✓ Attention weights shape: {output_t0['attn_weights'].shape}")
    assert slots_t0.shape == (batch_size, num_slots, slot_dim)
    
    # Second frame (with previous slots)
    features_t1 = torch.randn(batch_size, num_features, feature_dim)
    output_t1 = slot_attn(features_t1, prev_slots=slots_t0)
    slots_t1 = output_t1['slots']
    print(f"✓ Second frame slots shape: {slots_t1.shape}")
    assert slots_t1.shape == (batch_size, num_slots, slot_dim)
    
    # Test with spatial features
    features_spatial = torch.randn(batch_size, 24, 24, feature_dim)
    output_spatial = slot_attn(features_spatial, prev_slots=None)
    slots_spatial = output_spatial['slots']
    print(f"✓ Spatial features slots shape: {slots_spatial.shape}")
    assert slots_spatial.shape == (batch_size, num_slots, slot_dim)
    
    print("\nSlot Attention test passed! ✓")


if __name__ == "__main__":
    test_slot_attention()
