"""
Recurrent Slot Attention following TTT3R State Management Pattern

This module implements object-centric slot attention with TTT3R-style state management:
1. register_tokens: Learnable slot embeddings (like TTT3R's state tokens)
2. _encode_state: Initialize slot states with positional encoding
3. _decoder: Attend to features and decode slots (iterative attention)
4. model_update_type="ttt3r": Confidence-weighted slot updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class RecurrentSlotAttention(nn.Module):
    """
    Recurrent Slot Attention following TTT3R's state management pattern.
    
    Architecture mirrors TTT3R's approach:
    - register_tokens: Learnable slot embeddings
    - _encode_state: Initialize slots with position encoding
    - _decoder: Iterative attention mechanism to decode slots from features
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
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.model_update_type = model_update_type
        self.epsilon = epsilon
        
        hidden_dim = hidden_dim or slot_dim
        mlp_hidden_dim = mlp_hidden_dim or slot_dim
        
        # TTT3R-style: register_tokens for slot initialization
        self.register_tokens = nn.Embedding(num_slots, slot_dim)
        
        # Layer normalization
        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
        # Attention parameters
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
        
        # For TTT3R-style confidence-weighted updates
        if model_update_type == "ttt3r":
            self.confidence_proj = nn.Linear(hidden_dim, 1)
    
    def _encode_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slot states (following TTT3R's _encode_state pattern).
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Initial slot states (batch_size, num_slots, slot_dim)
        """
        # Use register_tokens to initialize slots (like TTT3R's state initialization)
        slot_indices = torch.arange(self.num_slots, device=device)
        slots = self.register_tokens(slot_indices)  # (num_slots, slot_dim)
        slots = slots.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_slots, slot_dim)
        
        # Add small noise for diversity
        slots = slots + torch.randn_like(slots) * 0.01
        
        return slots
    
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
            attn_weights: Attention weights (B, K, N)
            attn_logits: Attention logits for confidence (B, K, N)
        """
        B, K, _ = slots.shape
        _, N, _ = features.shape
        
        # Normalize
        slots_norm = self.norm_slots(slots)
        features_norm = self.norm_inputs(features)
        
        # Project to queries, keys, values
        q = self.project_q(slots_norm)  # (B, K, D_hidden)
        k = self.project_k(features_norm)  # (B, N, D_hidden)
        v = self.project_v(features_norm)  # (B, N, D_slot)
        
        # Compute attention logits
        attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale  # (B, K, N)
        
        # Softmax over slots (competition for features)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, K, N)
        
        # Normalize over features
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.epsilon)
        
        return attn_weights, attn_logits
    
    def _slot_attention_iteration(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single iteration of slot attention.
        
        Args:
            slots: Current slot states (B, K, D_slot)
            features: Input features (B, N, D_feat)
            
        Returns:
            updated_slots: Updated slot states (B, K, D_slot)
            attn_weights: Attention weights (B, K, N)
        """
        B, K, D_slot = slots.shape
        _, N, D_feat = features.shape
        
        # Compute attention
        attn_weights, attn_logits = self._compute_attention(slots, features)
        
        # Aggregate features using attention
        features_norm = self.norm_inputs(features)
        v = self.project_v(features_norm)  # (B, N, D_slot)
        
        # Weighted sum of features
        updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)  # (B, K, D_slot)
        
        # Update slots using GRU
        # Flatten for GRU
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
        features: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode slots from features using iterative attention (TTT3R's _decoder pattern).
        
        Args:
            slots: Initial slot states (B, K, D_slot)
            features: Input features (B, N, D_feat) or (B, H, W, D_feat)
            prev_slots: Previous slot states for recurrent updates
            confidence: Optional confidence map for TTT3R-style updates
            
        Returns:
            final_slots: Final slot states (B, K, D_slot)
            attn_weights: Final attention weights (B, K, N)
            slot_confidence: Per-slot confidence (B, K) if using TTT3R update
        """
        B = features.shape[0]
        
        # Flatten spatial features if needed
        if features.ndim == 4:  # (B, H, W, D)
            B, H, W, D = features.shape
            features = features.view(B, H * W, D)
        
        # Iterative attention refinement
        current_slots = slots
        for i in range(self.num_iterations):
            current_slots, attn_weights = self._slot_attention_iteration(
                current_slots, features
            )
        
        # TTT3R-style confidence-weighted update with previous slots
        slot_confidence = None
        if prev_slots is not None and self.model_update_type == "ttt3r":
            # Compute slot confidence from attention weights
            # Higher confidence if slot attends strongly to any feature
            attn_max = attn_weights.max(dim=-1)[0]  # (B, K)
            slot_confidence = torch.sigmoid(attn_max)  # (B, K)
            
            # Update with confidence weighting
            update_weight = slot_confidence.unsqueeze(-1)  # (B, K, 1)
            current_slots = current_slots * update_weight + prev_slots * (1 - update_weight)
        
        return current_slots, attn_weights, slot_confidence
    
    def forward(
        self,
        features: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through slot attention.
        
        Args:
            features: Input features (B, N, D) or (B, H, W, D)
            prev_slots: Previous slots for recurrent processing (B, K, D_slot)
            confidence: Optional confidence map (B, H, W) or (B, N)
            
        Returns:
            Dictionary with:
                - slots: Slot representations (B, K, D_slot)
                - attn_weights: Attention weights (B, K, N)
                - slot_confidence: Per-slot confidence (B, K) if using TTT3R
        """
        B = features.shape[0]
        device = features.device
        
        # Initialize slots using _encode_state (TTT3R pattern)
        if prev_slots is None:
            slots = self._encode_state(B, device)
        else:
            # Use previous slots as initialization
            slots = prev_slots
        
        # Decode slots from features using _decoder (TTT3R pattern)
        final_slots, attn_weights, slot_confidence = self._decoder(
            slots, features, prev_slots, confidence
        )
        
        output = {
            'slots': final_slots,
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
            f"  model_update_type='{self.model_update_type}'\n"
            f")"
        )


# Test function
def test_slot_attention():
    """Test the recurrent slot attention module."""
    print("Testing Recurrent Slot Attention (TTT3R-style)...")
    
    # Parameters
    batch_size = 2
    num_slots = 8
    slot_dim = 128
    feature_dim = 384
    num_features = 576  # 24x24 patches
    
    # Create module with TTT3R-style updates
    slot_attn = RecurrentSlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        feature_dim=feature_dim,
        num_iterations=3,
        model_update_type="ttt3r"
    )
    
    # Test with random features
    features = torch.randn(batch_size, num_features, feature_dim)
    
    # First frame (no previous slots) - uses _encode_state
    output_t0 = slot_attn(features, prev_slots=None)
    slots_t0 = output_t0['slots']
    print(f"✓ First frame slots shape: {slots_t0.shape}")
    print(f"✓ Attention weights shape: {output_t0['attn_weights'].shape}")
    assert slots_t0.shape == (batch_size, num_slots, slot_dim)
    
    # Second frame (with previous slots) - uses _decoder with TTT3R updates
    features_t1 = torch.randn(batch_size, num_features, feature_dim)
    output_t1 = slot_attn(features_t1, prev_slots=slots_t0)
    slots_t1 = output_t1['slots']
    print(f"✓ Second frame slots shape: {slots_t1.shape}")
    if 'slot_confidence' in output_t1:
        print(f"✓ TTT3R slot confidence shape: {output_t1['slot_confidence'].shape}")
    assert slots_t1.shape == (batch_size, num_slots, slot_dim)
    
    # Test with spatial features
    features_spatial = torch.randn(batch_size, 24, 24, feature_dim)
    output_spatial = slot_attn(features_spatial, prev_slots=None)
    slots_spatial = output_spatial['slots']
    print(f"✓ Spatial features slots shape: {slots_spatial.shape}")
    assert slots_spatial.shape == (batch_size, num_slots, slot_dim)
    
    print("\nRecurrent Slot Attention (TTT3R-style) test passed! ✓")


if __name__ == "__main__":
    test_slot_attention()
