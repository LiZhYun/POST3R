"""
TTT3R Backbone Wrapper

This module wraps the pretrained TTT3R model to extract:
1. Image features (F_t) for slot attention
2. Camera poses (P_t) for 3D decoder
3. World-coordinate pointmaps (X_world,t) as pseudo-ground-truth

The backbone is FROZEN during POST3R training.

Optional DINOv2 encoder support for ablation studies.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import warnings

# Add TTT3R to path
TTT3R_PATH = Path(__file__).parent.parent.parent / "submodules" / "ttt3r" / "src"
if str(TTT3R_PATH) not in sys.path:
    sys.path.insert(0, str(TTT3R_PATH))

# Add SlotContrast to path for DINOv2 encoder
SLOTCONTRAST_PATH = Path(__file__).parent.parent.parent / "submodules" / "slotcontrast"
if str(SLOTCONTRAST_PATH) not in sys.path:
    sys.path.insert(0, str(SLOTCONTRAST_PATH))


class TTT3RBackbone(nn.Module):
    """
    Frozen TTT3R backbone for geometric feature extraction.
    
    This wrapper:
    - Loads pretrained TTT3R model
    - Extracts intermediate features from encoder
    - Manages recurrent memory state
    - Provides camera poses and 3D pointmaps
    - Ensures no gradient computation (frozen)
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: int = 512,
        device: str = 'cuda',
        frozen: bool = True,
        extract_features_from: str = 'encoder',
        use_dinov2: bool = False,  # NEW: Option to use DINOv2 encoder instead of TTT3R encoder
        dinov2_model: str = 'vit_base_patch14_dinov2',  # DINOv2 model name
        dinov2_features_key: str = 'vit_block12',  # Which DINOv2 layer to extract features from
        feature_dim: int = 768,  # DINOv2 feature dimension (768 for base, 1024 for large)
    ):
        """
        Initialize TTT3R backbone wrapper.
        
        Args:
            model_path: Path to TTT3R checkpoint (.pth file)
            input_size: Input image size (default: 512)
            device: Device to run on ('cuda' or 'cpu')
            frozen: Whether to freeze all parameters (should be True)
            extract_features_from: Which layer to extract features from
            use_dinov2: If True, use DINOv2 encoder for features instead of TTT3R encoder
            dinov2_model: DINOv2 model name (from timm)
            dinov2_features_key: Which DINOv2 layer to extract features from
            feature_dim: Output feature dimension (768 for DINOv2-base, 1024 for TTT3R)
        """
        super().__init__()
        
        self.model_path = model_path
        self.input_size = input_size
        self.device = device
        self.frozen = frozen
        self.use_dinov2 = use_dinov2
        self.feature_dim = feature_dim
        
        # Import TTT3R components
        try:
            from dust3r.model import ARCroco3DStereo
            from dust3r.utils.image import load_images
        except ImportError as e:
            raise ImportError(
                f"Failed to import TTT3R modules. "
                f"Make sure submodules are initialized. Error: {e}"
            )
        
        # Load pretrained TTT3R model (always needed for pose and pointmap)
        print(f"Loading TTT3R model from {model_path}...")
        self.model = ARCroco3DStereo.from_pretrained(model_path).to(device)
        self.model.config.model_update_type = "ttt3r"
        self.model.eval()
        
        # Freeze TTT3R parameters
        if frozen:
            self.freeze()
            print("TTT3R backbone frozen (no gradients)")
        
        # Initialize DINOv2 encoder if requested
        self.dinov2_encoder = None
        if use_dinov2:
            print(f"Initializing DINOv2 encoder: {dinov2_model}")
            from slotcontrast.modules.encoders import TimmExtractor
            
            self.dinov2_encoder = TimmExtractor(
                model=dinov2_model,
                pretrained=True,
                frozen=True,
                features=dinov2_features_key,
            ).to(device)
            
            # Compute number of patches for DINOv2
            # DINOv2 with patch14 on 518x518 image: (518/14)^2 ≈ 37^2 = 1369 patches
            # For 512x512: (512/14)^2 ≈ 36.57^2 ≈ 1369 patches (will be rounded)
            self.dinov2_patch_size = 14
            print(f"DINOv2 encoder initialized (frozen, feature_dim={feature_dim})")
        
        # Memory state (maintained across frames)
        self.memory_state = None
        
    @property
    def rope(self):
        """Get ROPE instance from TTT3R model for slot attention."""
        return self.model.rope if hasattr(self.model, 'rope') else None
        
    def freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.frozen = True
        
    def reset_memory(self):
        """Reset the recurrent memory state."""
        self.memory_state = None
    
    def _prepare_views(
        self,
        frames: torch.Tensor,
        reset_interval: int = 1000000,
    ) -> list:
        """
        Prepare views for TTT3R following demo.py prepare_input pattern.
        
        This function creates properly formatted view dictionaries that match
        TTT3R's expected input format, including proper image preprocessing.
        
        Args:
            frames: Input frames (B, T, C, H, W) for sequence or (B, C, H, W) for single frame
            reset_interval: Reset state every N frames (for very long sequences)
            
        Returns:
            List of view dictionaries in TTT3R format
        """
        from dust3r.utils.image import _resize_pil_image, ImgNorm
        from PIL import Image
        import numpy as np
        
        # Handle single frame vs sequence
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)
        
        B, T, C, H_orig, W_orig = frames.shape
        
        views = []
        
        # Process each frame in the sequence
        for t in range(T):
            frame_batch = frames[:, t]  # (B, C, H, W)
            
            # Convert to PIL for proper resizing (following TTT3R's load_images)
            processed_imgs = []
            true_shapes = []
            
            for b in range(B):
                # Convert tensor to PIL Image
                img_tensor = frame_batch[b]  # (C, H, W)
                img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Original size
                W1, H1 = pil_img.size
                
                # Resize following TTT3R's load_images logic
                if self.input_size == 224:
                    img = _resize_pil_image(pil_img, round(self.input_size * max(W1 / H1, H1 / W1)))
                else:
                    img = _resize_pil_image(pil_img, self.input_size)
                
                # Center crop following TTT3R's logic
                W, H = img.size
                cx, cy = W // 2, H // 2
                if self.input_size == 224:
                    half = min(cx, cy)
                    img = img.crop((cx - half, cy - half, cx + half, cy + half))
                else:
                    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                    # For non-square images (which is typical for videos)
                    if W != H:
                        pass  # Keep as is
                    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
                
                # Final size after crop
                W2, H2 = img.size
                true_shape = np.int32([H2, W2])  # Note: [height, width]
                
                # Normalize image following TTT3R's ImgNorm
                # ImgNorm returns a tensor, and [None] adds batch dimension
                img_normalized = ImgNorm(img)[None]  # Shape: (1, C, H, W)
                
                processed_imgs.append(img_normalized[0])  # Remove batch dim: (C, H, W)
                true_shapes.append(true_shape)
            
            # Stack batch
            processed_batch = torch.stack(processed_imgs).to(frames.device)  # (B, C, H_new, W_new)
            true_shape_batch = torch.from_numpy(np.array(true_shapes)).to(frames.device)  # (B, 2)
            
            # Create view dict following demo.py format
            view = {
                'img': processed_batch,  # (B, C, H_resized, W_resized) - properly resized
                'img_mask': torch.ones(B, dtype=torch.bool, device=frames.device),
                'ray_mask': torch.zeros(B, dtype=torch.bool, device=frames.device),
                'ray_map': torch.full(
                    (B, 6, processed_batch.shape[-2], processed_batch.shape[-1]),
                    torch.nan,
                    device=frames.device
                ),
                'true_shape': true_shape_batch.to(torch.int32),  # (B, 2) - [H_final, W_final]
                'reset': torch.tensor([t == 0 or (t+1) % reset_interval == 0], device=frames.device).repeat(B),
                'update': torch.ones(B, dtype=torch.bool, device=frames.device),
                'idx': t,
                'instance': str(t),
                'camera_pose': torch.eye(4, dtype=torch.float32, device=frames.device).unsqueeze(0).repeat(B, 1, 1),
            }
            views.append(view)
        
        return views
        
    def _prepare_input(self, frames: torch.Tensor) -> Dict:
        """
        Prepare input format for TTT3R.
        
        Args:
            frames: Input frames (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Dictionary with TTT3R input format
        """
        # Handle single frame or batch of frames
        if frames.ndim == 4:
            # Single frame: (B, C, H, W)
            batch_size = frames.shape[0]
            frames = frames.unsqueeze(1)  # (B, 1, C, H, W)
        elif frames.ndim == 5:
            # Multiple frames: (B, T, C, H, W)
            batch_size, num_frames = frames.shape[:2]
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {frames.shape}")
        
        # Create view dictionaries (TTT3R format)
        views = []
        for b in range(batch_size):
            for t in range(frames.shape[1]):
                view = {
                    'img': frames[b, t],  # (C, H, W)
                    'idx': t,
                    'instance': str(t),
                    'img_mask': torch.tensor(True),
                    'ray_mask': torch.tensor(False),
                }
                views.append(view)
        
        return views
    
    def _extract_features(self, model_output: Dict) -> torch.Tensor:
        """
        Extract image features from TTT3R encoder output.
        
        Args:
            model_output: Output dictionary from TTT3R
            
        Returns:
            Image features tensor (B, H, W, D) where D is feature dimension
        """
        # TTT3R stores features in different places depending on the model
        # Typically from encoder blocks or patch embeddings
        
        # Extract from encoder output
        if 'encoder_output' in model_output:
            features = model_output['encoder_output']
        elif hasattr(self.model, '_enc_output'):
            features = self.model._enc_output
        else:
            raise AttributeError(
                "Could not find encoder features in model output. "
                "Model must have 'encoder_output' in output dict or '_enc_output' attribute. "
                f"Available keys in model_output: {list(model_output.keys()) if isinstance(model_output, dict) else 'N/A'}"
            )
            
        return features
    
    def _extract_poses(self, pred: Dict) -> torch.Tensor:
        """
        Extract camera poses from TTT3R predictions.
        
        Args:
            pred: Prediction dictionary from TTT3R
            
        Returns:
            Camera poses (B, 3, 4) or (B, 4, 4)
        """
        if 'camera_pose' not in pred:
            raise KeyError(
                "TTT3R prediction does not contain 'camera_pose'. "
                f"Available keys: {list(pred.keys())}"
            )
        
        poses = pred['camera_pose']
        # Convert to 3x4 matrix if needed
        if poses.shape[-2:] == (4, 4):
            poses = poses[..., :3, :]  # Remove last row
        return poses
    
    def _extract_pointmaps(self, pred: Dict) -> torch.Tensor:
        """
        Extract 3D pointmaps in world coordinates from TTT3R predictions.
        
        Args:
            pred: Prediction dictionary from TTT3R
            
        Returns:
            3D pointmaps (B, H, W, 3) in world coordinates
        """
        # TTT3R predicts points in different coordinate systems
        # We want world coordinates
        
        if 'pts3d_in_other_view' in pred:
            # World coordinate points
            pointmaps = pred['pts3d_in_other_view']
        elif 'pts3d_in_self_view' in pred:
            # Camera coordinate points (need transformation)
            pointmaps = pred['pts3d_in_self_view']
        elif 'pts3d' in pred:
            pointmaps = pred['pts3d']
        else:
            raise KeyError(
                "TTT3R prediction does not contain any pointmap keys "
                "('pts3d_in_other_view', 'pts3d_in_self_view', or 'pts3d'). "
                f"Available keys: {list(pred.keys())}"
            )
        
        return pointmaps
    
    def _extract_features_dinov2(
        self, 
        original_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using DINOv2 encoder instead of TTT3R encoder.
        
        Args:
            original_images: Original input images (B, C, H, W) - NOT preprocessed by TTT3R
            
        Returns:
            Tuple of:
            - features: (B, N, D) where N is number of patches, D is feature dim (768 for DINOv2-base)
            - positions: (B, N, 2) patch positions for ROPE compatibility
        """
        # DINOv2 expects the original images (already normalized by dataset transform)
        # Do NOT use TTT3R preprocessing for DINOv2
        with torch.no_grad():
            features_dict = self.dinov2_encoder(original_images)
            
            # Get the main features
            # TimmExtractor returns dict for ViT models
            if isinstance(features_dict, dict):
                # Extract the specific layer features (e.g., 'vit_block12')
                feat_key = list(features_dict.keys())[0]
                features = features_dict[feat_key]
            else:
                features = features_dict
            
            # Features should be (B, N, D) format from ViT
            # TimmExtractor already removes CLS token for us
            if features.ndim == 3:
                B, N, D = features.shape
            else:
                raise ValueError(f"Unexpected feature shape from DINOv2: {features.shape}")
            
            # Create position embeddings compatible with ROPE
            # Generate 2D grid positions for patches
            H_patch = W_patch = int(N ** 0.5)
            y_pos = torch.arange(H_patch, device=features.device, dtype=torch.long)
            x_pos = torch.arange(W_patch, device=features.device, dtype=torch.long)
            yy, xx = torch.meshgrid(y_pos, x_pos, indexing='ij')
            positions = torch.stack([yy, xx], dim=-1).reshape(-1, 2)  # (N, 2)
            positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
            
        return features, positions
    
    def forward(
        self, 
        frames: torch.Tensor,
        return_confidence: bool = True,
        reset_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through frozen TTT3R backbone.
        
        This processes a BATCH of frames (not a sequence). For sequences, 
        use forward_sequence() which properly batches all frames together.
        
        Args:
            frames: Input frames (B, C, H, W) - single timestep
            return_confidence: Whether to return confidence scores
            reset_memory: Whether to reset recurrent state
            
        Returns:
            Dictionary with:
            - 'features': Image features (B, H_f, W_f, D) for slot attention
            - 'pose': Camera poses (B, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            - 'pointmap': 3D pointmaps (B, H, W, 3) in world coords
            - 'confidence': Optional confidence scores (B, H, W)
        """
        # Disable autocast entirely for ROCm compatibility
        # ROCm's bf16/fp16 autocast causes issues with RoPE position embeddings
        with torch.amp.autocast('cuda', enabled=False):
            return self._forward_impl(frames, return_confidence, reset_memory)
    
    def _forward_impl(
        self,
        frames: torch.Tensor,
        return_confidence: bool = True,
        reset_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single batch of frames using proper TTT3R preprocessing.
        """
        if reset_memory:
            self.reset_memory()
        
        with torch.no_grad():
            B, C, H_orig, W_orig = frames.shape
            
            # Use _prepare_views for proper TTT3R preprocessing
            views = self._prepare_views(frames.unsqueeze(1))  # Add time dimension
            view = views[0]  # Get the single frame view
            
            # Forward through TTT3R
            with torch.amp.autocast('cuda', enabled=False):
                output, state_args = self.model([view], ret_state=True)
                preds = output.ress
            
            # Update memory state
            if state_args:
                self.memory_state = state_args[-1]
            
            # Extract prediction
            pred = preds[0]
            
            # Extract encoder features
            processed_img = view['img']  # (B, C, H_resized, W_resized)
            true_shape = view['true_shape']  # (B, 2)
            
            if self.use_dinov2:
                # Use DINOv2 encoder for features - pass original frames, not preprocessed
                feat, feat_pos = self._extract_features_dinov2(frames)
                # feat: (B, N, D) where D is DINOv2 feature dim (768 for base)
                N = feat.shape[1]
                feat_size = int(N ** 0.5)
                features = feat.reshape(B, feat_size, feat_size, -1)
            else:
                # Use TTT3R encoder for features
                img_out, img_pos, _ = self.model._encode_image(processed_img, true_shape)
                feat = img_out[-1]  # (B, N, D)
                N = feat.shape[1]
                feat_pos = img_pos  # (B, N, 2) for ROPE
                feat_size = int(N ** 0.5)
                features = feat.reshape(B, feat_size, feat_size, -1)
            
            # Extract poses (always from TTT3R)
            if 'camera_pose' not in pred:
                raise KeyError(f"No 'camera_pose' in prediction. Keys: {list(pred.keys())}")
            poses = pred['camera_pose']
            
            # Extract 3D pointmaps (always from TTT3R)
            if 'pts3d_in_other_view' in pred:
                pointmaps = pred['pts3d_in_other_view']
            elif 'pts3d' in pred:
                pointmaps = pred['pts3d']
                if 'camera_pose' in pred:
                    from dust3r.utils.geometry import geotrf
                    pointmaps = geotrf(pred['camera_pose'], pointmaps)
            else:
                raise KeyError(f"No pointmap in prediction. Keys: {list(pred.keys())}")
            
            # Resize pointmaps to match original input resolution
            H_pts, W_pts = pointmaps.shape[1:3]
            if (H_pts, W_pts) != (H_orig, W_orig):
                pointmaps = torch.nn.functional.interpolate(
                    pointmaps.permute(0, 3, 1, 2),
                    size=(H_orig, W_orig),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Extract confidence scores
            if return_confidence:
                if 'conf' in pred:
                    confidence = pred['conf']
                    if confidence.shape[1:] != (H_orig, W_orig):
                        confidence = torch.nn.functional.interpolate(
                            confidence.unsqueeze(1),
                            size=(H_orig, W_orig),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                elif 'confidence' in pred:
                    confidence = pred['confidence']
                    if confidence.shape[1:] != (H_orig, W_orig):
                        confidence = torch.nn.functional.interpolate(
                            confidence.unsqueeze(1),
                            size=(H_orig, W_orig),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                else:
                    confidence = torch.ones(B, H_orig, W_orig, device=self.device)
            else:
                confidence = None
            
        return {
            'features': features,
            'feature_pos': feat_pos,  # (B, N, 2) for ROPE
            'pose': poses,
            'pointmap': pointmaps,
            'confidence': confidence,
        }
    
    def forward_sequence(
        self,
        frames: torch.Tensor,
        return_confidence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for video sequence using TTT3R's batched API.
        
        This is the CORRECT way to process sequences - TTT3R processes
        all frames together in one call to forward_recurrent_lighter.
        
        Args:
            frames: Input video (B, T, C, H, W) where T is sequence length
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with:
            - 'features': Image features (B, T, H_f, W_f, D)
            - 'pose': Camera poses (B, T, 7) in [tx, ty, tz, qx, qy, qz, qw] format
            - 'pointmap': 3D pointmaps (B, T, H, W, 3)
            - 'confidence': Optional confidence scores (B, T, H, W)
        """
        # Disable autocast entirely for ROCm compatibility
        # ROCm's bf16/fp16 autocast causes issues with RoPE position embeddings
        with torch.amp.autocast('cuda', enabled=False):
            return self._forward_sequence_impl(frames, return_confidence)
    
    def _forward_sequence_impl(
        self,
        frames: torch.Tensor,
        return_confidence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process video sequence using TTT3R's forward() API with list of views.
        
        This follows TTT3R's demo.py pattern with proper image preprocessing.
        Images are resized using TTT3R's _resize_pil_image which handles aspect ratio.
        """
        with torch.no_grad():
            B, T, C, H, W = frames.shape
            
            # Reset memory at start of sequence
            self.reset_memory()
            
            # Prepare views using TTT3R's format (handles resizing properly)
            views = self._prepare_views(frames)
            
            # Call TTT3R forward with ALL views at once (correct batched API)
            output, state_args = self.model(views, ret_state=True)
            preds = output.ress  # List of predictions, one per frame
            
            # Update memory state 
            if state_args:
                self.memory_state = state_args[-1]
            
            # Extract features and predictions for each frame
            all_features = []
            all_feature_pos = []
            all_poses = []
            all_pointmaps = []
            all_confidences = [] if return_confidence else None
            
            for t in range(T):
                pred = preds[t]
                view_t = views[t]
                
                # Extract encoder features
                processed_img = view_t['img']  # (B, C, H_resized, W_resized)
                true_shape = view_t['true_shape']  # (B, 2)
                
                # Get original frame for DINOv2
                frame_t = frames[:, t]  # (B, C, H, W)
                
                if self.use_dinov2:
                    # Use DINOv2 encoder for features - pass original frame, not preprocessed
                    feat, feat_pos = self._extract_features_dinov2(frame_t)
                    # feat: (B, N, D) where D is DINOv2 feature dim (768 for base)
                    N = feat.shape[1]
                    feat_size = int(N ** 0.5)
                    features_t = feat.reshape(B, feat_size, feat_size, -1)
                else:
                    # Use TTT3R encoder for features
                    img_out, img_pos, _ = self.model._encode_image(processed_img, true_shape)
                    feat = img_out[-1]  # (B, N, D)
                    N = feat.shape[1]
                    feat_pos = img_pos  # (B, N, 2) for ROPE
                    feat_size = int(N ** 0.5)
                    features_t = feat.reshape(B, feat_size, feat_size, -1)
                
                all_features.append(features_t)
                all_feature_pos.append(feat_pos)  # (B, N, 2) for ROPE
                
                # Extract poses - always from TTT3R
                poses_t = pred['camera_pose']
                all_poses.append(poses_t)
                
                # Extract pointmaps - always from TTT3R
                if 'pts3d_in_other_view' in pred:
                    pointmaps_t = pred['pts3d_in_other_view']
                elif 'pts3d' in pred:
                    pointmaps_t = pred['pts3d']
                    if 'camera_pose' in pred:
                        from dust3r.utils.geometry import geotrf
                        pointmaps_t = geotrf(pred['camera_pose'], pointmaps_t)
                else:
                    raise KeyError(f"No pointmap in prediction. Keys: {list(pred.keys())}")
                
                # TTT3R output pointmaps may be at patch resolution, resize to input resolution if needed
                H_pts, W_pts = pointmaps_t.shape[1:3]
                if (H_pts, W_pts) != (H, W):
                    pointmaps_t = torch.nn.functional.interpolate(
                        pointmaps_t.permute(0, 3, 1, 2),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                all_pointmaps.append(pointmaps_t)
                
                # Extract confidence
                if return_confidence:
                    if 'conf' in pred:
                        conf_t = pred['conf']
                        if conf_t.shape[1:] != (H, W):
                            conf_t = torch.nn.functional.interpolate(
                                conf_t.unsqueeze(1),
                                size=(H, W),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(1)
                    else:
                        conf_t = torch.ones(B, H, W, device=self.device)
                    all_confidences.append(conf_t)
            
            # Stack all frames
            result = {
                'features': torch.stack(all_features, dim=1),  # (B, T, H_f, W_f, D)
                'feature_pos': torch.stack(all_feature_pos, dim=1),  # (B, T, N, 2) for ROPE
                'pose': torch.stack(all_poses, dim=1),  # (B, T, 7) [tx, ty, tz, qx, qy, qz, qw]
                'pointmap': torch.stack(all_pointmaps, dim=1),  # (B, T, H, W, 3)
            }
            
            if return_confidence:
                result['confidence'] = torch.stack(all_confidences, dim=1)  # (B, T, H, W)
            
            return result
    
    def __repr__(self):
        return (
            f"TTT3RBackbone(\n"
            f"  model_path={self.model_path},\n"
            f"  input_size={self.input_size},\n"
            f"  device={self.device},\n"
            f"  frozen={self.frozen}\n"
            f")"
        )


# Test function
def test_backbone():
    """Test the TTT3R backbone wrapper."""
    print("Testing TTT3R Backbone...")
    
    # Check if checkpoint exists
    checkpoint_path = "submodules/ttt3r/src/cut3r_512_dpt_4_64.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Please download the checkpoint first.")
        return
    
    # Create backbone
    backbone = TTT3RBackbone(
        model_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test forward pass
    batch_size = 2
    frames = torch.randn(batch_size, 3, 512, 512)
    
    if torch.cuda.is_available():
        frames = frames.cuda()
    
    features, poses, pointmaps, _ = backbone(frames)
    
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Poses shape: {poses.shape}")
    print(f"✓ Pointmaps shape: {pointmaps.shape}")
    print(f"✓ All parameters frozen: {all(not p.requires_grad for p in backbone.parameters())}")
    print("\nBackbone test passed! ✓")


if __name__ == "__main__":
    test_backbone()
