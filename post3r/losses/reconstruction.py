"""
Loss Functions for POST3R

Simple MSE loss for 3D pointmap reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PointmapReconstructionLoss(nn.Module):
    """
    MSE Loss for 3D pointmap reconstruction.
    """
    
    def __init__(self):
        """Initialize reconstruction loss."""
        super().__init__()
    
    def forward(
        self,
        pred_pointmap: torch.Tensor,
        gt_pointmap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MSE reconstruction loss.
        
        Args:
            pred_pointmap: Predicted pointmap (B, H, W, 3)
            gt_pointmap: Ground truth pointmap (B, H, W, 3)
            
        Returns:
            Dictionary with losses
        """
        # Compute MSE loss
        mse_loss = F.mse_loss(pred_pointmap, gt_pointmap)
        
        return {
            'total_loss': mse_loss,
            'recon_loss': mse_loss,
            'l1_loss': torch.tensor(0.0, device=mse_loss.device),
            'l2_loss': mse_loss,
            'temporal_loss': torch.tensor(0.0, device=mse_loss.device),
            'entropy_loss': torch.tensor(0.0, device=mse_loss.device),
            'diversity_loss': torch.tensor(0.0, device=mse_loss.device),
        }


# Alias for backward compatibility
POST3RLoss = PointmapReconstructionLoss
