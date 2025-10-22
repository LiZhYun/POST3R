"""
Loss functions for POST3R

Implements the reconstruction loss for 3D pointmaps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class POST3RLoss(nn.Module):
    """
    POST3R Loss Function
    
    Computes MSE loss between predicted and ground truth 3D pointmaps.
    Handles resolution mismatches by resizing GT to match prediction.
    """
    
    def __init__(
        self,
        pointmap_weight: float = 1.0,
        confidence_weighting: bool = True,
    ):
        """
        Initialize loss function.
        
        Args:
            pointmap_weight: Weight for pointmap reconstruction loss
            confidence_weighting: Whether to weight loss by TTT3R confidence
        """
        super().__init__()
        self.pointmap_weight = pointmap_weight
        self.confidence_weighting = confidence_weighting
    
    def forward(
        self,
        pred_pointmap: torch.Tensor,
        gt_pointmap: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred_pointmap: Predicted pointmap (B, H_pred, W_pred, 3)
            gt_pointmap: Ground truth pointmap (B, H_gt, W_gt, 3)
            confidence: Optional confidence weights (B, H_gt, W_gt)
            
        Returns:
            Scalar loss value
        """
        B, H_pred, W_pred, C = pred_pointmap.shape
        B_gt, H_gt, W_gt, C_gt = gt_pointmap.shape
        
        assert C == 3 and C_gt == 3, "Pointmaps must have 3 channels (XYZ)"
        assert B == B_gt, f"Batch size mismatch: {B} vs {B_gt}"
        
        # Resize GT pointmap to match prediction resolution if needed
        if (H_pred, W_pred) != (H_gt, W_gt):
            # Permute to (B, C, H, W) for interpolation
            gt_pointmap_resized = F.interpolate(
                gt_pointmap.permute(0, 3, 1, 2),
                size=(H_pred, W_pred),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
            
            # Also resize confidence if provided
            if confidence is not None:
                confidence_resized = F.interpolate(
                    confidence.unsqueeze(1),  # (B, 1, H, W)
                    size=(H_pred, W_pred),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # (B, H, W)
            else:
                confidence_resized = None
        else:
            gt_pointmap_resized = gt_pointmap
            confidence_resized = confidence
        
        # Compute MSE loss
        loss = F.mse_loss(pred_pointmap, gt_pointmap_resized, reduction='none')
        # loss shape: (B, H, W, 3)
        
        # Average over XYZ channels
        loss = loss.mean(dim=-1)  # (B, H, W)
        
        # Apply confidence weighting if available
        if self.confidence_weighting and confidence_resized is not None:
            # Normalize confidence to [0, 1] range if needed
            conf_min = confidence_resized.min()
            conf_max = confidence_resized.max()
            if conf_max > conf_min:
                confidence_norm = (confidence_resized - conf_min) / (conf_max - conf_min)
            else:
                confidence_norm = torch.ones_like(confidence_resized)
            
            # Weight loss by confidence
            loss = loss * confidence_norm
        
        # Average over spatial dimensions and batch
        loss = loss.mean()
        
        # Apply weight
        loss = self.pointmap_weight * loss
        
        return loss
    
    def __repr__(self):
        return (
            f"POST3RLoss(\n"
            f"  pointmap_weight={self.pointmap_weight},\n"
            f"  confidence_weighting={self.confidence_weighting}\n"
            f")"
        )


# Test function
def test_loss():
    """Test the loss function."""
    print("Testing POST3R Loss...")
    
    loss_fn = POST3RLoss()
    
    # Test case 1: Same resolution
    print("\nTest 1: Same resolution")
    pred = torch.randn(2, 64, 64, 3)
    gt = torch.randn(2, 64, 64, 3)
    loss = loss_fn(pred, gt)
    print(f"✓ Loss with same resolution: {loss.item():.4f}")
    
    # Test case 2: Different resolutions
    print("\nTest 2: Different resolutions (pred: 224x224, gt: 518x518)")
    pred = torch.randn(2, 224, 224, 3)
    gt = torch.randn(2, 518, 518, 3)
    loss = loss_fn(pred, gt)
    print(f"✓ Loss with different resolutions: {loss.item():.4f}")
    
    # Test case 3: With confidence weighting
    print("\nTest 3: With confidence weighting")
    pred = torch.randn(2, 224, 224, 3)
    gt = torch.randn(2, 518, 518, 3)
    confidence = torch.rand(2, 518, 518)  # Random confidence [0, 1]
    loss = loss_fn(pred, gt, confidence)
    print(f"✓ Loss with confidence weighting: {loss.item():.4f}")
    
    # Test case 4: Without confidence weighting
    print("\nTest 4: Without confidence weighting")
    loss_fn_no_conf = POST3RLoss(confidence_weighting=False)
    loss = loss_fn_no_conf(pred, gt, confidence)
    print(f"✓ Loss without confidence weighting: {loss.item():.4f}")
    
    print("\nPOST3R Loss tests passed! ✓")


if __name__ == "__main__":
    test_loss()
