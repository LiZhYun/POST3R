"""
Reconstruction loss for POST3R
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PointmapReconstructionLoss(nn.Module):
    """
    POST3R Loss Function
    
    Computes MSE loss between predicted and ground truth 3D pointmaps and features.
    Handles resolution mismatches by resizing GT to match prediction.
    """
    
    def __init__(
        self,
        pointmap_weight: float = 0.5,
        feature_weight: float = 1.0,
    ):
        """
        Initialize loss function.
        
        Args:
            pointmap_weight: Weight for pointmap reconstruction loss
            feature_weight: Weight for feature reconstruction loss
        """
        super().__init__()
        self.pointmap_weight = pointmap_weight
        self.feature_weight = feature_weight
    
    def _compute_norm(self, pointmap: torch.Tensor) -> torch.Tensor:
        """
        Compute normalization factor as average distance to origin (Eq. 3).
        
        Args:
            pointmap: (B, H, W, 3) or (B, T, H, W, 3)
            
        Returns:
            norm: (B,) or (B, T) - average L2 norm per batch
        """
        if pointmap.ndim == 5:
            # Temporal dimension: (B, T, H, W, 3)
            B, T, H, W, _ = pointmap.shape
            norms = torch.norm(pointmap, p=2, dim=-1)  # (B, T, H, W)
            avg_norm = norms.view(B, T, -1).mean(dim=2)  # (B, T)
        else:
            # Single frame: (B, H, W, 3)
            B, H, W, _ = pointmap.shape
            norms = torch.norm(pointmap, p=2, dim=-1)  # (B, H, W)
            avg_norm = norms.view(B, -1).mean(dim=1)  # (B,)
        return avg_norm
    
    def forward(
        self,
        pred_pointmap: torch.Tensor,
        gt_pointmap: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        gt_features: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute reconstruction loss.
        
        Args:
            pred_pointmap: Predicted pointmap (B, H_pred, W_pred, 3) or (B, T, H_pred, W_pred, 3)
            gt_pointmap: Ground truth pointmap (B, H_gt, W_gt, 3) or (B, T, H_gt, W_gt, 3)
            pred_features: Predicted features (B, H_pred, W_pred, D) or (B, T, H_pred, W_pred, D) - optional
            gt_features: Ground truth features (B, H_gt, W_gt, D) or (B, T, H_gt, W_gt, D) - optional
            confidence: Confidence scores from TTT3R (B, H_gt, W_gt) or (B, T, H_gt, W_gt) - optional
            
        Returns:
            Dictionary with loss components
        """
        # Handle both single frame and temporal inputs
        has_temporal = pred_pointmap.ndim == 5
        
        if has_temporal:
            # Temporal: (B, T, H, W, 3)
            B, T, H_pred, W_pred, C = pred_pointmap.shape
            B_gt, T_gt, H_gt, W_gt, C_gt = gt_pointmap.shape
            assert T == T_gt, f"Temporal dimension mismatch: {T} vs {T_gt}"
        else:
            # Single frame: (B, H, W, 3)
            B, H_pred, W_pred, C = pred_pointmap.shape
            B_gt, H_gt, W_gt, C_gt = gt_pointmap.shape
        
        assert C == 3 and C_gt == 3, "Pointmaps must have 3 channels (XYZ)"
        assert B == B_gt, f"Batch size mismatch: {B} vs {B_gt}"
        
        # Resize predicted pointmap to match GT resolution if needed
        if (H_pred, W_pred) != (H_gt, W_gt):
            if has_temporal:
                # Reshape to (B*T, 3, H, W) for interpolation
                pred_pointmap_resized = pred_pointmap.view(B * T, H_pred, W_pred, 3).permute(0, 3, 1, 2)
                pred_pointmap_resized = F.interpolate(
                    pred_pointmap_resized,
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                )
                pred_pointmap_resized = pred_pointmap_resized.permute(0, 2, 3, 1).view(B, T, H_gt, W_gt, 3)
            else:
                pred_pointmap_resized = F.interpolate(
                    pred_pointmap.permute(0, 3, 1, 2),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
        else:
            pred_pointmap_resized = pred_pointmap
        
        # Normalize pointmaps by average distance to origin (Eq. 3)
        pred_norm = self._compute_norm(pred_pointmap_resized)
        gt_norm = self._compute_norm(gt_pointmap)
        
        if has_temporal:
            pred_pointmap_normalized = pred_pointmap_resized / (pred_norm.view(B, T, 1, 1, 1) + 1e-8)
            gt_pointmap_normalized = gt_pointmap / (gt_norm.view(B, T, 1, 1, 1) + 1e-8)
        else:
            pred_pointmap_normalized = pred_pointmap_resized / (pred_norm.view(B, 1, 1, 1) + 1e-8)
            gt_pointmap_normalized = gt_pointmap / (gt_norm.view(B, 1, 1, 1) + 1e-8)
        
        # Compute per-pixel regression loss (Euclidean distance, Eq. 2)
        pointmap_loss = torch.norm(
            pred_pointmap_normalized - gt_pointmap_normalized, 
            p=2, 
            dim=-1
        )  # (B, H_gt, W_gt) or (B, T, H_gt, W_gt)
        
        # Apply confidence weighting (Eq. 4, without the log term)
        if confidence is not None:
            pointmap_loss = pointmap_loss * confidence
        
        # Average over spatial dimensions and batch (and time if present)
        pointmap_loss = pointmap_loss.mean()
        
        # Compute feature loss if features are provided
        feature_loss = torch.tensor(0.0, device=pred_pointmap.device)
        if pred_features is not None and gt_features is not None:
            if has_temporal:
                B_f, T_f, H_f_pred, W_f_pred, D_f = pred_features.shape
                B_f_gt, T_f_gt, H_f_gt, W_f_gt, D_f_gt = gt_features.shape
                assert T_f == T_f_gt, f"Temporal dimension mismatch: {T_f} vs {T_f_gt}"
            else:
                B_f, H_f_pred, W_f_pred, D_f = pred_features.shape
                B_f_gt, H_f_gt, W_f_gt, D_f_gt = gt_features.shape
            
            assert D_f == D_f_gt, f"Feature dimension mismatch: {D_f} vs {D_f_gt}"
            assert B_f == B_f_gt, f"Batch size mismatch: {B_f} vs {B_f_gt}"
            
            # Resize predicted features to match GT resolution if needed
            if (H_f_pred, W_f_pred) != (H_f_gt, W_f_gt):
                if has_temporal:
                    # Reshape to (B*T, D, H, W) for interpolation
                    pred_features_resized = pred_features.view(B_f * T_f, H_f_pred, W_f_pred, D_f).permute(0, 3, 1, 2)
                    pred_features_resized = F.interpolate(
                        pred_features_resized,
                        size=(H_f_gt, W_f_gt),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_features_resized = pred_features_resized.permute(0, 2, 3, 1).view(B_f, T_f, H_f_gt, W_f_gt, D_f)
                else:
                    pred_features_resized = F.interpolate(
                        pred_features.permute(0, 3, 1, 2),
                        size=(H_f_gt, W_f_gt),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
            else:
                pred_features_resized = pred_features
            
            # Compute feature MSE loss
            feature_loss = F.mse_loss(pred_features_resized, gt_features, reduction='mean')
        
        # Combine losses
        total_loss = (
            self.pointmap_weight * pointmap_loss +
            self.feature_weight * feature_loss
        )
        
        return {
            'loss': total_loss,
            'pointmap_loss': pointmap_loss,
            'feature_loss': feature_loss,
        }


# Alias for compatibility
POST3RLoss = PointmapReconstructionLoss

