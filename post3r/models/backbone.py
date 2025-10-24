"""
TTT3R Backbone Wrapper

This module wraps the pretrained TTT3R model to extract:
1. Image features (F_t) for slot attention
2. Camera poses (P_t) for 3D decoder
3. World-coordinate pointmaps (X_world,t) as pseudo-ground-truth

The backbone is FROZEN during POST3R training.
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
        device: str = 'cpu',
        frozen: bool = True,
        extract_features_from: str = 'encoder',
    ):
        """
        Initialize TTT3R backbone wrapper.
        
        Args:
            model_path: Path to TTT3R checkpoint (.pth file)
            input_size: Input image size (default: 512)
            device: Device to run on ('cuda' or 'cpu')
            frozen: Whether to freeze all parameters (should be True)
            extract_features_from: Which layer to extract features from
        """
        super().__init__()
        
        self.model_path = model_path
        self.input_size = input_size
        self.device = device
        self.frozen = frozen
        
        # Import TTT3R components
        try:
            from dust3r.model import ARCroco3DStereo
            from dust3r.utils.image import load_images
        except ImportError as e:
            raise ImportError(
                f"Failed to import TTT3R modules. "
                f"Make sure submodules are initialized. Error: {e}"
            )
        
        # Load pretrained model
        # Note: TTT3R checkpoints use OmegaConf which requires special handling in PyTorch 2.6+
        print(f"Loading TTT3R model from {model_path}...")
        self.model = ARCroco3DStereo.from_pretrained(model_path).to(device)
        self.model.config.model_update_type = "ttt3r"

        self.model.eval()
        
        # Freeze parameters
        if frozen:
            self.freeze()
            print("TTT3R backbone frozen (no gradients)")
        
        # Memory state (maintained across frames)
        self.memory_state = None
        
    def freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.frozen = True
        
    def reset_memory(self):
        """Reset the recurrent memory state."""
        self.memory_state = None
        
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
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_impl(frames, return_confidence, reset_memory)
    
    def _forward_impl(
        self,
        frames: torch.Tensor,
        return_confidence: bool = True,
        reset_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if reset_memory:
            self.reset_memory()
        # Ensure no gradients
        with torch.no_grad():
            batch_size = frames.shape[0]
            H, W = frames.shape[-2:]
            
            # TTT3R expects 512x512 images - resize if needed
            if H != self.input_size or W != self.input_size:
                frames_resized = torch.nn.functional.interpolate(
                    frames, 
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                frames_resized = frames
            
            # Prepare view dictionary for TTT3R following demo.py format
            # TTT3R expects specific keys: img, img_mask, ray_mask, reset, update, true_shape, ray_map
            # First frame resets state, subsequent frames continue from previous state
            is_first_frame = self.memory_state is None
            
            view = {
                'img': frames_resized,  # (B, C, H, W) - already resized to 512x512
                'img_mask': torch.ones(batch_size, dtype=torch.bool, device=frames.device),  # All images are valid
                'ray_mask': torch.zeros(batch_size, dtype=torch.bool, device=frames.device),  # No ray maps
                'ray_map': torch.full(
                    (batch_size, 6, self.input_size, self.input_size),
                    torch.nan,
                    device=frames.device
                ),  # Dummy ray map filled with NaN
                'true_shape': torch.tensor(
                    [[self.input_size, self.input_size]], 
                    device=frames.device
                ).repeat(batch_size, 1),  # (B, 2) - [H, W]
                'reset': torch.tensor([is_first_frame], device=frames.device).repeat(batch_size),  # Reset on first frame
                'update': torch.ones(batch_size, dtype=torch.bool, device=frames.device),  # Always update
                'idx': 0,  # Frame index (not critical for our use)
                'instance': '0',  # Instance identifier
                'camera_pose': torch.eye(4, dtype=torch.float32, device=frames.device).unsqueeze(0).repeat(batch_size, 1, 1),  # Identity pose
            }
            
            # Forward through TTT3R model using forward API
            # This is the correct API for batched processing (see inference.py)
            with torch.cuda.amp.autocast(enabled=False):
                output, state_args = self.model([view], ret_state=True)
                preds = output.ress
            
            # Update memory state for next frame
            if state_args:
                self.memory_state = state_args[-1]
            
            # Extract prediction for this frame
            pred = preds[0]  # Get first (and only) prediction
            
            # Extract encoder features using _encode_image method
            # This follows the pattern in forward_recurrent (see model.py line 1036-1041)
            # The features are computed from the image encoder
            if not hasattr(self.model, '_encode_image'):
                raise AttributeError(
                    "TTT3R model does not have '_encode_image' method. "
                    "Cannot extract encoder features."
                )
            
            # Encode image to get features
            # _encode_image returns: (img_out, img_pos, _) where:
            # - img_out: list of feature tensors from different encoder blocks
            # - img_pos: position embeddings
            img_out, img_pos, _ = self.model._encode_image(frames_resized, view['true_shape'])
            
            # Extract features from the last encoder layer
            # img_out[-1] has shape (B, N, D) where:
            # - B: batch size
            # - N: number of patches (H_patches * W_patches)
            # - D: feature dimension (enc_embed_dim = 1024)
            feat = img_out[-1]  # (B, N, D)
            N = feat.shape[1]
            
            # Reshape to spatial grid (B, H_f, W_f, D)
            # For 512x512 images with patch_size=16, we get 32x32 patches
            feat_size = int(N ** 0.5)
            features = feat.reshape(batch_size, feat_size, feat_size, -1)  # (B, H_f, W_f, D)
            
            # Extract poses - already in correct format (B, 7) [tx, ty, tz, qx, qy, qz, qw]
            if 'camera_pose' not in pred:
                raise KeyError(
                    "TTT3R prediction does not contain 'camera_pose'. "
                    f"Available keys: {list(pred.keys())}"
                )
            
            poses = pred['camera_pose']
            
            # Extract 3D pointmaps (world coordinates)
            if 'pts3d_in_other_view' in pred:
                pointmaps = pred['pts3d_in_other_view']
            elif 'pts3d' in pred:
                # Transform to world coordinates using camera pose
                pointmaps = pred['pts3d']
                if 'camera_pose' in pred:
                    # Apply camera pose transformation
                    from dust3r.utils.geometry import geotrf
                    pointmaps = geotrf(pred['camera_pose'], pointmaps)
            else:
                raise KeyError(
                    "TTT3R prediction does not contain 'pts3d' or 'pts3d_in_other_view'. "
                    f"Available keys: {list(pred.keys())}"
                )
            
            # Resize pointmaps back to original frame size if needed
            if pointmaps.shape[1:3] != (H, W):
                # pointmaps: (B, H_ttt, W_ttt, 3) -> (B, 3, H_ttt, W_ttt)
                pointmaps_resized = pointmaps.permute(0, 3, 1, 2)
                pointmaps_resized = torch.nn.functional.interpolate(
                    pointmaps_resized,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                pointmaps = pointmaps_resized.permute(0, 2, 3, 1)  # (B, H, W, 3)
            
            # Extract confidence scores
            if return_confidence:
                if 'conf' in pred:
                    confidence = pred['conf']
                    # Resize if needed
                    if confidence.shape[1:] != (H, W):
                        confidence = torch.nn.functional.interpolate(
                            confidence.unsqueeze(1),  # (B, 1, H_ttt, W_ttt)
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)  # (B, H, W)
                elif 'confidence' in pred:
                    confidence = pred['confidence']
                    if confidence.shape[1:] != (H, W):
                        confidence = torch.nn.functional.interpolate(
                            confidence.unsqueeze(1),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                else:
                    confidence = torch.ones(batch_size, H, W, device=self.device)
            else:
                confidence = None
            
        return {
            'features': features,
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
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_sequence_impl(frames, return_confidence)
    
    def _forward_sequence_impl(
        self,
        frames: torch.Tensor,
        return_confidence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            B, T, C, H, W = frames.shape
            
            # Reset memory at start of sequence
            self.reset_memory()
            
            # Resize all frames if needed
            if H != self.input_size or W != self.input_size:
                frames_resized = torch.nn.functional.interpolate(
                    frames.view(B * T, C, H, W),
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                ).view(B, T, C, self.input_size, self.input_size)
            else:
                frames_resized = frames
            
            # Process frames one by one using recurrent inference
            # This avoids RoPE position embedding index out of bounds issues
            preds = []
            for t in range(T):
                view = {
                    'img': frames_resized[:, t],  # (B, C, H, W)
                    'img_mask': torch.ones(B, dtype=torch.bool, device=frames.device),
                    'ray_mask': torch.zeros(B, dtype=torch.bool, device=frames.device),
                    'ray_map': torch.full(
                        (B, 6, self.input_size, self.input_size),
                        torch.nan,
                        device=frames.device
                    ),
                    'true_shape': torch.tensor(
                        [[self.input_size, self.input_size]], 
                        device=frames.device, dtype=torch.int32
                    ).repeat(B, 1),
                    'reset': torch.tensor([t == 0], device=frames.device),
                    'update': torch.ones(B, dtype=torch.bool, device=frames.device),
                    'idx': t,
                    'instance': str(t),
                    'camera_pose': torch.eye(4, dtype=torch.float32, device=frames.device).unsqueeze(0).repeat(B, 1, 1),
                }
                
                # Process single frame
                output, state_args = self.model([view], ret_state=True)
                preds.append(output.ress[0])
                
                # Update memory state for next frame
                if state_args:
                    self.memory_state = state_args[-1]
            
            # Extract features and predictions for each frame
            all_features = []
            all_poses = []
            all_pointmaps = []
            all_confidences = [] if return_confidence else None
            
            true_shape = torch.tensor(
                [[self.input_size, self.input_size]], 
                device=frames.device, dtype=torch.int32
            ).repeat(B, 1)
            
            for t in range(T):
                pred = preds[t]
                
                # Extract encoder features
                img_out, img_pos, _ = self.model._encode_image(frames_resized[:, t], true_shape)
                feat = img_out[-1]  # (B, N, D)
                N = feat.shape[1]
                feat_size = int(N ** 0.5)
                features_t = feat.reshape(B, feat_size, feat_size, -1)
                all_features.append(features_t)
                
                # Extract poses - already in correct format (B, 7) [tx, ty, tz, qx, qy, qz, qw]
                poses_t = pred['camera_pose']
                all_poses.append(poses_t)
                
                # Extract pointmaps
                if 'pts3d_in_other_view' in pred:
                    pointmaps_t = pred['pts3d_in_other_view']
                elif 'pts3d' in pred:
                    pointmaps_t = pred['pts3d']
                    if 'camera_pose' in pred:
                        from dust3r.utils.geometry import geotrf
                        pointmaps_t = geotrf(pred['camera_pose'], pointmaps_t)
                else:
                    raise KeyError(f"No pointmap in prediction. Keys: {list(pred.keys())}")
                
                # Resize if needed
                if pointmaps_t.shape[1:3] != (H, W):
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
