#!/usr/bin/env python3
"""
POST3R 3D Pointmap Visualization Demo

Interactive visualization of object-centric 3D pointmaps from POST3R.
Uses viser PointCloudViewer similar to TTT3R demo.

Usage:
    python scripts/demo_post3r.py \
        --checkpoint outputs/post3r_ytvis2021/epoch=00-step=015000.ckpt \
        --seq_path examples/001 \
        --device cuda
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import tempfile
import shutil
from pathlib import Path

# Add TTT3R to path
sys.path.insert(0, str(Path(__file__).parent.parent / "submodules" / "ttt3r"))
from add_ckpt_path import add_path_to_dust3r

from post3r.training.lightning_module import POST3RLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="POST3R 3D Pointmap Visualization Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to POST3R checkpoint")
    parser.add_argument("--seq_path", type=str, required=True, help="Path to image directory or video")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--size", type=int, default=512, help="Input image size")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval for videos")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum frames to process")
    parser.add_argument("--port", type=int, default=7860, help="Port for viewer")
    parser.add_argument("--downsample_factor", type=int, default=1, help="Downsample factor for point cloud")
    parser.add_argument("--vis_threshold", type=float, default=1.5, help="Visualization confidence threshold")
    return parser.parse_args()


def load_images_from_path(seq_path, frame_interval=1, max_frames=50):
    """Load images from directory or video file."""
    if os.path.isdir(seq_path):
        img_paths = sorted([
            os.path.join(seq_path, f) for f in os.listdir(seq_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        return img_paths[::frame_interval][:max_frames], None
    
    elif os.path.isfile(seq_path):
        cap = cv2.VideoCapture(seq_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
        print(f"Extracting {len(frame_indices)} frames from video...")
        
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
        return img_paths, tmpdirname
    
    raise ValueError(f"Invalid path: {seq_path}")


def prepare_images(img_paths, size):
    """Load and preprocess images."""
    add_path_to_dust3r("submodules/ttt3r/src/cut3r_512_dpt_4_64.pth")
    from src.dust3r.utils.image import load_images
    
    images = load_images(img_paths, size=size)
    # Each image is already (1, C, H, W), so concatenate along batch dim
    img_tensors = torch.cat([img['img'] for img in images], dim=0)  # (T, C, H, W)
    
    return img_tensors


def run_inference(model, images, device):
    """Run POST3R inference on image sequence."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension and move to device: (T, 3, H, W) -> (1, T, 3, H, W)
        video = images.unsqueeze(0).to(device)
        
        # Use forward_sequence for efficient batch processing
        output = model.model.forward_sequence(video)
        
        # Move to CPU and split by frame
        all_outputs = []
        T = images.shape[0]
        for i in range(T):
            frame_output = {
                'slots': output['slots'][:, i],  # (B, T, K, D) -> (B, K, D)
                'recon_pointmap': output['recon_pointmap'][:, i],  # (B, T, H, W, 3) -> (B, H, W, 3)
                'gt_pointmap': output['gt_pointmap'][:, i],  # (B, T, H, W, 3) -> (B, H, W, 3)
                'pose': output['poses'][:, i],  # (B, T, 7) -> (B, 7)
                'confidence': output['confidence'][:, i] if 'confidence' in output else None,  # (B, T, H, W) -> (B, H, W)
            }
            frame_output_cpu = {k: v.cpu() for k, v in frame_output.items()}
            all_outputs.append(frame_output_cpu)
    
    return all_outputs


def prepare_outputs_for_viewer(outputs, images):
    """Prepare POST3R outputs for PointCloudViewer following demo.py's prepare_output."""
    # Import dust3r utilities (must be after add_path_to_dust3r is called)
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    import roma
    
    # Following TTT3R's demo.py prepare_output structure
    
    T = len(outputs)
    
    # Get dimensions from gt_pointmap (full resolution from TTT3R)
    B, H_gt, W_gt, _ = outputs[0]['gt_pointmap'].shape
    
    # Extract pointmaps and upsample to gt_pointmap resolution
    # NOTE: recon_pointmap is from decoder (e.g., 64x64), gt_pointmap is from TTT3R (e.g., 384x384)
    # We upsample recon_pointmap to match gt_pointmap resolution for better visualization
    pts3ds_other = []
    for out in outputs:
        recon_pts = out['gt_pointmap']  # (1, H_recon, W_recon, 3)
        B, H_recon, W_recon, _ = recon_pts.shape
        
        if (H_recon, W_recon) != (H_gt, W_gt):
            # Upsample to gt resolution: (1, H, W, 3) -> (1, 3, H, W) -> interpolate -> (1, 3, H_gt, W_gt) -> (1, H_gt, W_gt, 3)
            recon_pts_upsampled = torch.nn.functional.interpolate(
                recon_pts.permute(0, 3, 1, 2),  # (1, 3, H_recon, W_recon)
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # (1, H_gt, W_gt, 3)
            pts3ds_other.append(recon_pts_upsampled)
        else:
            pts3ds_other.append(recon_pts)
    
    # Now use gt_pointmap resolution for everything
    H_pts, W_pts = H_gt, W_gt
    
    # Extract confidence if available, otherwise create dummy confidence
    # Confidence from TTT3R already matches gt_pointmap resolution
    if outputs[0].get('confidence') is not None:
        conf_other = [out['confidence'] for out in outputs]  # List of (1, H_gt, W_gt) tensors
    else:
        # Create dummy confidence like demo.py does
        conf_other = [torch.ones(1, H_pts, W_pts) for _ in range(T)]
    
    # Stack to get pts3ds_self for focal estimation
    pts3ds_self = torch.cat(pts3ds_other, 0)  # (T, H, W, 3)
    
    # Estimate focal length based on depth (following demo.py)
    pp = torch.tensor([W_pts // 2, H_pts // 2], device=pts3ds_self.device).float().repeat(T, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    
    # Convert images to colors (T, 3, H, W) -> list of (1, H, W, 3) tensors
    colors = 0.5 * (images.permute(0, 2, 3, 1) + 1.0)
    
    # Resize colors to match pointmap resolution if needed
    T_img, H_color, W_color, _ = colors.shape
    if (H_color, W_color) != (H_pts, W_pts):
        colors_resized = []
        for color_frame in colors:
            color_resized = cv2.resize(color_frame.numpy(), (W_pts, H_pts), interpolation=cv2.INTER_LINEAR)
            colors_resized.append(torch.from_numpy(color_resized))
        colors = torch.stack(colors_resized)
    
    # Convert to list of (1, H, W, 3) tensors like demo.py
    colors_list = [colors[i].unsqueeze(0) for i in range(T)]
    
    # Extract camera poses from POST3R outputs
    # POST3R outputs poses in [tx, ty, tz, qx, qy, qz, qw] format
    # Convert to rotation matrix and translation
    pr_poses = []
    for out in outputs:
        pose_7d = out['pose'].squeeze(0)  # (7,) [tx, ty, tz, qx, qy, qz, qw]
        
        # Convert to 4x4 transformation matrix
        t = pose_7d[:3]  # translation
        quat = pose_7d[3:]  # quaternion [qx, qy, qz, qw]
        
        # Convert quaternion to rotation matrix
        R = roma.unitquat_to_rotmat(quat.unsqueeze(0)).squeeze(0)  # (3, 3)
        
        # Build 4x4 pose matrix
        pose_4x4 = torch.eye(4)
        pose_4x4[:3, :3] = R
        pose_4x4[:3, 3] = t
        
        pr_poses.append(pose_4x4.unsqueeze(0))  # (1, 4, 4)
    
    # Extract R and t for camera dictionary
    R_c2w = torch.cat([pose[:, :3, :3] for pose in pr_poses], 0)  # (T, 3, 3)
    t_c2w = torch.cat([pose[:, :3, 3] for pose in pr_poses], 0)  # (T, 3)
    
    # Create camera dictionary following demo.py format
    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }
    
    # Convert to numpy for viewer (keep as tensors until last moment like demo.py)
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors_list]
    # conf stays as list of tensors (viewer converts internally)
    
    # Edge colors (None for each frame)
    edge_color_list = [None for _ in range(T)]
    
    return pts3ds_to_vis, colors_to_vis, conf_other, edge_color_list, cam_dict


def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("CUDA not available, using CPU")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = POST3RLightningModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    
    # Load images
    print(f"Loading images from: {args.seq_path}")
    img_paths, tmpdirname = load_images_from_path(args.seq_path, args.frame_interval, args.max_frames)
    print(f"Found {len(img_paths)} images")
    
    # Preprocess
    print("Preprocessing images...")
    images = prepare_images(img_paths, args.size)
    
    if tmpdirname:
        shutil.rmtree(tmpdirname)
    
    # Run inference
    print(f"Running POST3R inference on {len(images)} frames...")
    outputs = run_inference(model, images, device)
    
    print(f"  Reconstructed pointmap shape: {outputs[0]['recon_pointmap'].shape}")
    print(f"  GT pointmap shape: {outputs[0]['gt_pointmap'].shape}")
    print(f"  Number of slots: {outputs[0]['slots'].shape[1]}")
    
    # Prepare for viewer
    print("Preparing outputs for visualization...")
    pts3ds, colors, conf_list, edge_color_list, cam_dict = prepare_outputs_for_viewer(outputs, images)
    
    # Debug: check pointmap statistics
    print(f"\nPointmap statistics:")
    for i in range(min(3, len(pts3ds))):
        pts = pts3ds[i]
        print(f"  Frame {i}:")
        print(f"    Shape: {pts.shape}")
        print(f"    Min: {pts.min():.3f}, Max: {pts.max():.3f}, Mean: {pts.mean():.3f}")
        print(f"    NaN count: {np.isnan(pts).sum()}, Inf count: {np.isinf(pts).sum()}")
        print(f"    Confidence shape: {conf_list[i].shape}")
    
    # Launch viewer
    from viser_utils import PointCloudViewer
    
    print(f"\nLaunching point cloud viewer on port {args.port}...")
    print(f"Open browser at: http://localhost:{args.port}")
    print("\nViewing POST3R reconstructed 3D pointmaps")
    print("Use viewer controls to:")
    print("  - Rotate: drag with mouse")
    print("  - Pan: right-click drag")
    print("  - Zoom: scroll")
    print("  - Navigate frames: use slider")
    
    viewer = PointCloudViewer(
        model=None,
        state_args=None,
        pc_list=pts3ds,
        color_list=colors,
        conf_list=conf_list,
        cam_dict=cam_dict,
        device=device,
        edge_color_list=edge_color_list,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size=args.size,
        port=args.port,
        downsample_factor=args.downsample_factor,
    )
    
    viewer.run()


if __name__ == "__main__":
    main()
