"""
Utility modules for POST3R.

Adapted from SlotContrast.
"""

import torch
import torch.nn as nn


class SoftToHardMask:
    """Module that converts masks from soft to hard (SlotContrast-style)."""

    def __init__(
        self,
        convert_one_hot: bool = True,
        use_threshold: bool = False,
        threshold: float = 0.5
    ):
        """
        Initialize SoftToHardMask.
        
        Args:
            convert_one_hot: Convert to one-hot encoding (take argmax)
            use_threshold: Use threshold instead of argmax
            threshold: Threshold value if use_threshold is True
        """
        self.convert_one_hot = convert_one_hot
        self.use_threshold = use_threshold
        self.threshold = threshold

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Convert soft masks to hard masks.
        
        Args:
            masks: Soft masks (B, K, H, W) or (B, T, K, H, W)
            
        Returns:
            Hard masks (same shape as input)
        """
        return soft_to_hard_mask(
            masks,
            self.convert_one_hot,
            self.use_threshold,
            self.threshold
        )


def soft_to_hard_mask(
    masks: torch.Tensor,
    convert_one_hot: bool = True,
    use_threshold: bool = False,
    threshold: float = 0.5,
):
    """
    Convert soft to hard masks.
    
    Args:
        masks: Soft masks (B, K, H, W) or (B, T, K, H, W)
               where K is number of slots/channels
        convert_one_hot: Convert to one-hot by taking argmax
        use_threshold: Use threshold instead
        threshold: Threshold value
        
    Returns:
        Hard masks in same shape as input
    """
    # masks: batch [x n_frames] x n_channels x height x width
    assert masks.ndim == 4 or masks.ndim == 5, f"Expected 4 or 5 dims, got {masks.ndim}"
    
    min_val = torch.min(masks)
    max_val = torch.max(masks)
    
    if min_val < 0:
        raise ValueError(f"Minimum mask value should be >=0, but found {min_val.cpu().numpy()}")
    if max_val > 1:
        raise ValueError(f"Maximum mask value should be <=1, but found {max_val.cpu().numpy()}")

    if use_threshold:
        masks = masks > threshold

    if convert_one_hot:
        # Determine which dimension is the channel/slot dimension
        if masks.ndim == 4:
            # (B, K, H, W)
            mask_argmax = torch.argmax(masks, dim=1)  # (B, H, W)
            masks = nn.functional.one_hot(mask_argmax, masks.shape[1]).to(torch.float32)
            # (B, H, W, K) -> (B, K, H, W)
            masks = masks.permute(0, 3, 1, 2)
        else:
            # (B, T, K, H, W)
            mask_argmax = torch.argmax(masks, dim=2)  # (B, T, H, W)
            masks = nn.functional.one_hot(mask_argmax, masks.shape[2]).to(torch.float32)
            # (B, T, H, W, K) -> (B, T, K, H, W)
            masks = masks.permute(0, 1, 4, 2, 3)

    return masks
