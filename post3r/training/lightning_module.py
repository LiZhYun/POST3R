"""
PyTorch Lightning Module for POST3R

Implements the training and validation logic for POST3R model.
Follows SlotContrast's pattern for mask processing and metrics.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Callable, Union, Tuple
import torchmetrics

from post3r.models import POST3R
from post3r.losses import POST3RLoss
from post3r.modules import SoftToHardMask


class POST3RLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for POST3R training.
    
    This wraps the POST3R model and handles:
    - Training step
    - Validation step
    - Optimizer configuration
    - Logging and visualization
    - Mask processing (SlotContrast-style)
    - Metrics integration
    """
    
    def __init__(
        self,
        # Model config
        ttt3r_checkpoint: str,
        num_slots: int = 8,
        slot_dim: int = 128,
        num_iterations: int = 3,
        decoder_resolution: tuple = (64, 64),
        
        # Optimizer config
        learning_rate: float = 1e-4,
        optimizer_type: str = 'adam',
        weight_decay: float = 0.0,
        
        # Scheduler config
        scheduler_type: Optional[str] = None,
        warmup_steps: int = 10000,
        decay_steps: Optional[int] = None,
        
        # Metrics config (SlotContrast-style)
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        
        # Visualization config
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, list] = "decoder",
        
        # Logging config
        log_every_n_steps: int = 100,
    ):
        """
        Initialize POST3R Lightning module.
        
        Args:
            ttt3r_checkpoint: Path to TTT3R checkpoint
            num_slots: Number of object slots
            slot_dim: Dimension of each slot
            num_iterations: Number of slot attention iterations
            decoder_resolution: Output resolution for decoder
            learning_rate: Learning rate
            optimizer_type: Optimizer type ('adam', 'adamw')
            weight_decay: Weight decay for optimizer
            scheduler_type: Learning rate scheduler
            warmup_steps: Number of warmup steps
            decay_steps: Total steps for decay
            train_metrics: Dictionary of training metrics (SlotContrast-style)
            val_metrics: Dictionary of validation metrics (SlotContrast-style)
            visualize: Whether to log visualizations
            visualize_every_n_steps: Visualization frequency
            masks_to_visualize: Which masks to visualize ('decoder', 'grouping', or list)
            log_every_n_steps: Logging frequency
        """
        super().__init__()
        
        # Save hyperparameters (ignore metrics to avoid serialization issues)
        self.save_hyperparameters(ignore=['train_metrics', 'val_metrics'])
        
        # Create model
        self.model = POST3R(
            ttt3r_checkpoint=ttt3r_checkpoint,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            decoder_resolution=decoder_resolution,
        )
        
        # Create loss function (simple MSE)
        self.loss_fn = POST3RLoss()
        
        # Metrics (SlotContrast-style)
        self.train_metrics = nn.ModuleDict(train_metrics) if train_metrics else nn.ModuleDict()
        self.val_metrics = nn.ModuleDict(val_metrics) if val_metrics else nn.ModuleDict()
        
        # Mask processing utilities (SlotContrast-style)
        self.mask_soft_to_hard = SoftToHardMask()
        
        # Visualization config
        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        
        # Masks to visualize
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping"):
                raise ValueError(f"Unknown mask type {key}. Should be 'decoder' or 'grouping'.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]
        
        # Config
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps if decay_steps is not None else 100000
    
    def forward(self, frames: torch.Tensor, reset_memory: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through POST3R model.
        
        Args:
            frames: Input frames (B, 3, H, W) or (B, T, 3, H, W)
            reset_memory: Whether to reset recurrent memory
            
        Returns:
            Dictionary with model outputs
        """
        if frames.ndim == 5:
            # Video sequence: (B, T, 3, H, W)
            return self.model.forward_sequence(frames)
        else:
            # Single frame: (B, 3, H, W)
            return self.model(frames, reset_memory=reset_memory)
    
    def process_masks(
        self,
        masks: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process masks for metrics and visualization (SlotContrast-style).
        
        Args:
            masks: Soft masks (B, T, K, H, W) or (B, K, H, W) or (B, K, N)
            
        Returns:
            Tuple of:
            - Soft masks (unchanged)
            - Hard masks (one-hot)
        """
        if masks is None:
            return None, None
        
        # Convert soft masks to hard (one-hot) masks
        masks_hard = self.mask_soft_to_hard(masks)
        
        return masks, masks_hard
    
    @torch.no_grad()
    def aux_forward(self, outputs: Dict[str, Any], input_frames: torch.Tensor) -> Dict[str, Any]:
        """
        Compute auxiliary outputs only needed for metrics and visualizations (SlotContrast-style).
        
        Following SlotContrast's pattern: decoder masks are already spatial, but grouping masks
        need to be resized from patch format to spatial format.
        
        Args:
            outputs: Model outputs
            input_frames: Input video frames (B, T, 3, H, W) for resizing reference
            
        Returns:
            Dictionary with processed masks
        """
        aux_outputs = {}
        B, T, C, H, W = input_frames.shape
        
        # Process decoder masks (B, T, K, H_dec, W_dec) - already spatial
        decoder_masks = outputs.get('decoder_masks')
        if decoder_masks is not None:
            # Decoder masks are already in spatial format, just need to resize to input size
            # and convert to hard masks
            B, T, K, H_dec, W_dec = decoder_masks.shape
            
            # Resize to input resolution if needed (for visualization and metrics)
            if (H_dec, W_dec) != (H, W):
                # Reshape to (B*T*K, H_dec, W_dec) for resizing
                decoder_masks_resized = decoder_masks.reshape(B * T * K, H_dec, W_dec).unsqueeze(1)
                decoder_masks_resized = torch.nn.functional.interpolate(
                    decoder_masks_resized,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1).reshape(B, T, K, H, W)
            else:
                decoder_masks_resized = decoder_masks
            
            # Convert to hard masks
            decoder_masks_hard = self.mask_soft_to_hard(decoder_masks_resized)
            
            aux_outputs['decoder_masks'] = decoder_masks_resized
            aux_outputs['decoder_masks_hard'] = decoder_masks_hard
            aux_outputs['decoder_masks_vis_hard'] = decoder_masks_hard
        
        # Process grouping masks (B, T, K, N) - from slot attention in patch format
        # These need to be resized to spatial format like in SlotContrast
        grouping_masks = outputs.get('grouping_masks')
        if grouping_masks is not None:
            # grouping_masks shape: (B, T, K, N) where N is number of patches
            B, T, K, N = grouping_masks.shape
            
            # Determine patch grid size from TTT3R backbone
            # For 512x512 input with patch_size=16, we get 32x32 patches
            patch_h = patch_w = int(N ** 0.5)
            assert patch_h * patch_w == N, f"Number of patches {N} must be a perfect square"
            
            # Reshape to spatial: (B, T, K, N) -> (B*T*K, patch_h, patch_w)
            grouping_masks_spatial = grouping_masks.reshape(B * T * K, patch_h, patch_w)
            
            # Resize to input resolution: (B*T*K, patch_h, patch_w) -> (B*T*K, H, W)
            grouping_masks_resized = torch.nn.functional.interpolate(
                grouping_masks_spatial.unsqueeze(1),  # Add channel dim
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).reshape(B, T, K, H, W)
            
            # Convert to hard masks
            grouping_masks_hard = self.mask_soft_to_hard(grouping_masks_resized)
            
            aux_outputs['grouping_masks'] = grouping_masks_resized
            aux_outputs['grouping_masks_hard'] = grouping_masks_hard
            aux_outputs['grouping_masks_vis_hard'] = grouping_masks_hard
        
        return aux_outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step (SlotContrast-style)."""
        frames = batch['video']  # (B, T, 3, H, W)
        batch_size = frames.shape[0]
        
        # Forward pass through sequence
        outputs = self.model.forward_sequence(frames)
        
        # Compute auxiliary outputs for metrics/visualization if needed
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(outputs, frames)
        else:
            aux_outputs = {}
        
        # Compute losses frame by frame
        total_loss = 0.0
        num_frames = frames.shape[1]
        
        for t in range(num_frames):
            # Get predictions and targets for frame t
            pred_pointmap = outputs['recon_pointmap'][:, t]  # (B, H, W, 3)
            gt_pointmap = outputs['gt_pointmap'][:, t]  # (B, H, W, 3)
            
            # Compute loss
            losses = self.loss_fn(pred_pointmap, gt_pointmap)
            total_loss += losses['total_loss']
        
        # Average over frames
        total_loss = total_loss / num_frames
        
        # Logging dict (SlotContrast-style)
        to_log = {'train/loss': total_loss}
        
        # Update and log training metrics (SlotContrast-style)
        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                # Merge batch, outputs, and aux_outputs (aux_outputs overrides outputs)
                metric_inputs = {**batch, **outputs, **aux_outputs}
                values = metric(**metric_inputs)
                self._add_metric_to_log(to_log, f'train/{key}', values)
                metric.reset()
        
        # Log all metrics
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Delete outputs to save memory (SlotContrast-style)
        del outputs
        
        # Visualize periodically (SlotContrast-style)
        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_visualizations(batch, aux_outputs, mode='train')
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step (SlotContrast-style)."""
        frames = batch['video']  # (B, T, 3, H, W)
        batch_size = frames.shape[0]
        
        # Forward pass
        outputs = self.model.forward_sequence(frames)
        aux_outputs = self.aux_forward(outputs, frames)
        
        # Compute losses for all frames
        total_loss = 0.0
        num_frames = frames.shape[1]
        
        for t in range(num_frames):
            pred_pointmap = outputs['recon_pointmap'][:, t]  # (B, H, W, 3)
            gt_pointmap = outputs['gt_pointmap'][:, t]  # (B, H, W, 3)
            
            losses = self.loss_fn(pred_pointmap, gt_pointmap)
            total_loss += losses['total_loss']
        
        total_loss = total_loss / num_frames
        
        # Logging dict
        to_log = {'val/loss': total_loss}
        
        # Update validation metrics (SlotContrast-style)
        # Just update, don't compute yet (will compute in validation_epoch_end)
        if self.val_metrics:
            # Merge batch, outputs, and aux_outputs (aux_outputs overrides outputs)
            metric_inputs = {**batch, **outputs, **aux_outputs}
            for key, metric in self.val_metrics.items():
                metric.update(**metric_inputs)
        
        # Log losses
        self.log_dict(to_log, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        
        # Visualize first batch (SlotContrast-style)
        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            self._log_visualizations(batch, aux_outputs, mode='val')
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch (SlotContrast-style)."""
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f'val/{key}', metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)
    
    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any],
        name: str,
        values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """Add metric values to logging dict (SlotContrast-style)."""
        if isinstance(values, dict):
            # If metric returns a dict, log each sub-metric
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            # Single value
            log_dict[name] = values
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get trainable parameters (TTT3R backbone is frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        # Create optimizer
        if self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Configure scheduler if specified
        if self.scheduler_type is None:
            return optimizer
        
        if self.scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_steps,
                eta_min=self.learning_rate * 0.01
            )
        elif self.scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.trainer.max_steps // 3,
                gamma=0.1
            )
        elif self.scheduler_type.lower() in ['warmup_cosine', 'exp_decay_with_warmup']:
            # Warmup + cosine/exponential decay
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    # Linear warmup
                    return current_step / self.warmup_steps
                
                # Exponential decay after warmup
                progress = (current_step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                
                if self.scheduler_type.lower() == 'warmup_cosine':
                    # Cosine decay
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
                else:
                    # Exponential decay
                    return torch.exp(torch.tensor(-5.0 * progress)).item()
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def _log_visualizations(
        self,
        batch: Dict[str, torch.Tensor],
        aux_outputs: Dict[str, torch.Tensor],
        mode: str = 'train'
    ):
        """Log visualizations to tensorboard (SlotContrast-style)."""
        logger = self._get_tensorboard_logger()
        if logger is None:
            return
        
        step = self.trainer.global_step
        frames = batch['video']  # (B, T, 3, H, W)
        
        # Log input video frames
        self._log_video(f"{mode}/input_video", frames, global_step=step)
        
        # Log masks overlaid on input video
        for mask_key in self.mask_keys_to_visualize:
            mask_key_vis = f"{mask_key}_vis_hard"
            if mask_key_vis in aux_outputs:
                masks = aux_outputs[mask_key_vis]
                # Mix video with masks (overlay masks on frames)
                video_with_masks = self._mix_videos_with_masks(frames, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_key}",
                    video_with_masks,
                    global_step=step
                )
        
        # Log individual masks (per slot)
        self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode=mode, step=step)
    
    def _mix_videos_with_masks(
        self,
        videos: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.4
    ) -> torch.Tensor:
        """
        Overlay masks on videos (SlotContrast-style).
        
        Args:
            videos: (B, T, 3, H, W) normalized input videos
            masks: (B, T, K, H, W) hard masks or (T, K, H, W) if unbatched
            alpha: Blending factor for masks
            
        Returns:
            Videos with colored masks overlaid
        """
        # Handle both batched and unbatched inputs
        if masks.ndim == 4:
            # Unbatched: (T, K, H, W) -> add batch dimension
            masks = masks.unsqueeze(0)
        if videos.ndim == 4:
            # Unbatched: (T, 3, H, W) -> add batch dimension
            videos = videos.unsqueeze(0)
            
        B, T, K, H_mask, W_mask = masks.shape
        _, _, _, H_vid, W_vid = videos.shape
        
        # Resize masks to match video resolution if needed
        if H_mask != H_vid or W_mask != W_vid:
            # Reshape to (B*T*K, 1, H, W) for interpolation
            masks_flat = masks.view(B * T * K, 1, H_mask, W_mask)
            masks_resized = torch.nn.functional.interpolate(
                masks_flat,
                size=(H_vid, W_vid),
                mode='nearest'
            )
            # Reshape back to (B, T, K, H, W)
            masks = masks_resized.view(B, T, K, H_vid, W_vid)
        
        # Create color palette for slots
        base_colors = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ], device=videos.device)
        
        # Ensure we have exactly K colors (repeat or slice as needed)
        if K <= len(base_colors):
            colors = base_colors[:K]
        else:
            # Repeat colors to cover all K slots
            colors = base_colors.repeat((K // len(base_colors)) + 1, 1)[:K]
        
        # Denormalize videos (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=videos.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=videos.device).view(1, 1, 3, 1, 1)
        videos_denorm = videos * std + mean
        videos_denorm = torch.clamp(videos_denorm, 0, 1)
        
        # Create colored masks
        # masks: (B, T, K, H, W), colors: (K, 3)
        colored_masks = torch.einsum('btkxy,kc->btcxy', masks.float(), colors)  # (B, T, 3, H, W)
        
        # Blend with original videos
        # Compute total mask coverage per pixel
        mask_sum = masks.sum(dim=2)  # (B, T, H, W)
        mask_sum = mask_sum.clamp(min=1e-6)
        
        # Normalize colored masks by total coverage
        colored_masks = colored_masks / mask_sum.unsqueeze(2)  # (B, T, 3, H, W) / (B, T, 1, H, W)
        
        # Overlay
        videos_with_masks = (1 - alpha) * videos_denorm + alpha * colored_masks
        videos_with_masks = torch.clamp(videos_with_masks, 0, 1)
        
        return videos_with_masks
    
    def _log_masks(
        self,
        aux_outputs: Dict[str, torch.Tensor],
        mask_keys: list,
        mode: str = 'val',
        step: Optional[int] = None,
    ):
        """Log individual slot masks (SlotContrast-style)."""
        if step is None:
            step = self.trainer.global_step
        
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]  # (B, T, K, H, W) or (T, K, H, W)
                
                # Handle unbatched masks
                if masks.ndim == 4:
                    # (T, K, H, W) -> add batch dimension
                    masks = masks.unsqueeze(0)
                
                B, T, K, H, W = masks.shape
                
                # Take first batch element and invert (white background)
                first_masks = masks[0]  # (T, K, H, W)
                first_masks = first_masks.permute(1, 0, 2, 3)  # (K, T, H, W)
                first_masks_inverted = 1 - first_masks.unsqueeze(2)  # (K, T, 1, H, W)
                
                # Log as video (one per slot)
                self._log_video(
                    f"{mode}/{mask_key}",
                    first_masks_inverted,
                    global_step=step,
                    n_examples=K,
                )
    
    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
    ):
        """Log videos to tensorboard (SlotContrast-style)."""
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()
        
        if logger is not None:
            # Log as frames (grid)
            B, T, C, H, W = data.shape
            num_frames = min(max_frames, T)
            data_frames = data[:, :num_frames]
            data_frames = data_frames.flatten(0, 1)  # (B*T, C, H, W)
            
            from torchvision.utils import make_grid
            grid = make_grid(data_frames, nrow=num_frames)
            logger.experiment.add_image(
                f"{name}/frames",
                grid,
                global_step=global_step
            )
    
    def _get_tensorboard_logger(self):
        """Get TensorBoard logger (SlotContrast-style)."""
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger
        return None
    
    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return (
            f"POST3RLightningModule(\n"
            f"  num_slots={self.hparams.num_slots},\n"
            f"  slot_dim={self.hparams.slot_dim},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
