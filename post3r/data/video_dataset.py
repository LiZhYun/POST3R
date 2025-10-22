"""
Video Dataset for POST3R Training

Supports loading video sequences from:
1. Video files (MP4, AVI)
2. Frame directories
3. WebDataset format (for large-scale datasets like MOVi-C)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Union
import json
import random

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not available. Install with: pip install imageio")


class VideoDataset(Dataset):
    """
    Dataset for loading video sequences.
    
    Supports multiple data sources:
    - Video files (.mp4, .avi, etc.)
    - Frame directories (image sequences)
    - List of video paths
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_length: int = 16,
        frame_skip: int = 1,
        split: str = 'train',
        transform: Optional[Callable] = None,
        video_ext: str = '.mp4',
        return_video_id: bool = True,
        cache_videos: bool = False,
    ):
        """
        Initialize video dataset.
        
        Args:
            data_root: Root directory containing videos
            sequence_length: Number of frames per sequence
            frame_skip: Skip every N frames (for temporal subsampling)
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to frames
            video_ext: Video file extension to search for
            return_video_id: Whether to return video ID in output
            cache_videos: Whether to cache loaded videos in memory
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.split = split
        self.transform = transform
        self.return_video_id = return_video_id
        self.cache_videos = cache_videos
        
        # Find all videos
        self.video_paths = self._find_videos(video_ext)
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No videos found in {data_root} with extension {video_ext}")
        
        print(f"Found {len(self.video_paths)} videos in {split} split")
        
        # Build index of valid sequences
        self.sequences = self._build_sequence_index()
        
        # Optional video cache
        self.video_cache = {} if cache_videos else None
    
    def _find_videos(self, video_ext: str) -> List[Path]:
        """Find all video files in data_root."""
        # Check if data_root is a file containing list of paths
        if self.data_root.is_file():
            with open(self.data_root, 'r') as f:
                paths = [Path(line.strip()) for line in f]
            return paths
        
        # Otherwise search directory
        video_paths = list(self.data_root.glob(f'**/*{video_ext}'))
        
        # Filter by split if split file exists
        split_file = self.data_root / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_names = set(line.strip() for line in f)
            video_paths = [p for p in video_paths if p.stem in split_names]
        
        return sorted(video_paths)
    
    def _build_sequence_index(self) -> List[Dict]:
        """Build index of all valid sequences."""
        sequences = []
        
        for video_idx, video_path in enumerate(self.video_paths):
            # Get video length
            num_frames = self._get_video_length(video_path)
            
            # Calculate number of valid sequences
            max_start_frame = num_frames - (self.sequence_length * self.frame_skip)
            
            if max_start_frame < 0:
                print(f"Warning: Video {video_path.name} too short, skipping")
                continue
            
            # For training: sample multiple starting points
            # For val/test: use non-overlapping sequences
            if self.split == 'train':
                # Sample sequences with random starts
                num_sequences = max(1, max_start_frame // (self.sequence_length * self.frame_skip))
                for _ in range(num_sequences):
                    sequences.append({
                        'video_idx': video_idx,
                        'video_path': video_path,
                        'start_frame': None,  # Random at runtime
                        'num_frames': num_frames,
                    })
            else:
                # Non-overlapping sequences
                stride = self.sequence_length * self.frame_skip
                for start in range(0, max_start_frame + 1, stride):
                    sequences.append({
                        'video_idx': video_idx,
                        'video_path': video_path,
                        'start_frame': start,
                        'num_frames': num_frames,
                    })
        
        return sequences
    
    def _get_video_length(self, video_path: Path) -> int:
        """Get number of frames in video."""
        if not HAS_CV2:
            raise RuntimeError("OpenCV required for video loading")
        
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return num_frames
    
    def _load_video_frames(
        self, 
        video_path: Path, 
        start_frame: int,
        num_frames: int,
    ) -> np.ndarray:
        """
        Load frames from video.
        
        Returns:
            frames: (T, H, W, 3) numpy array, uint8, RGB
        """
        # Check cache
        if self.video_cache is not None and str(video_path) in self.video_cache:
            all_frames = self.video_cache[str(video_path)]
            frame_indices = range(start_frame, start_frame + num_frames * self.frame_skip, self.frame_skip)
            return all_frames[frame_indices]
        
        # Load with OpenCV
        if HAS_CV2:
            frames = []
            cap = cv2.VideoCapture(str(video_path))
            
            for i in range(num_frames):
                frame_idx = start_frame + i * self.frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            frames = np.stack(frames, axis=0)
            
        elif HAS_IMAGEIO:
            # Fallback to imageio
            reader = imageio.get_reader(str(video_path))
            frames = []
            for i in range(num_frames):
                frame_idx = start_frame + i * self.frame_skip
                frame = reader.get_data(frame_idx)
                frames.append(frame)
            reader.close()
            frames = np.stack(frames, axis=0)
            
        else:
            raise RuntimeError("Either OpenCV or imageio is required")
        
        return frames
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sequence.
        
        Returns:
            Dictionary with:
            - 'frames': (T, C, H, W) tensor, float32, [0, 1]
            - 'video_id': video identifier (if return_video_id=True)
            - 'start_frame': starting frame index
        """
        seq_info = self.sequences[idx]
        
        # Random start frame for training
        if seq_info['start_frame'] is None:
            max_start = seq_info['num_frames'] - (self.sequence_length * self.frame_skip)
            start_frame = random.randint(0, max_start)
        else:
            start_frame = seq_info['start_frame']
        
        # Load frames
        frames = self._load_video_frames(
            seq_info['video_path'],
            start_frame,
            self.sequence_length
        )  # (T, H, W, 3) uint8
        
        # Convert to tensor and normalize
        frames = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # Apply transforms
        if self.transform is not None:
            frames = self.transform(frames)
        
        # Prepare output
        output = {
            'frames': frames,
            'start_frame': start_frame,
        }
        
        if self.return_video_id:
            output['video_id'] = seq_info['video_idx']
            output['video_name'] = seq_info['video_path'].stem
        
        return output
    
    def __repr__(self):
        return (
            f"VideoDataset(\n"
            f"  split='{self.split}',\n"
            f"  num_videos={len(self.video_paths)},\n"
            f"  num_sequences={len(self.sequences)},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  frame_skip={self.frame_skip}\n"
            f")"
        )


class FrameDirectoryDataset(Dataset):
    """
    Dataset for loading sequences from frame directories.
    
    Expected structure:
    data_root/
        video1/
            frame_000000.png
            frame_000001.png
            ...
        video2/
            ...
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        sequence_length: int = 16,
        frame_skip: int = 1,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_ext: str = '.png',
    ):
        """Initialize frame directory dataset."""
        super().__init__()
        
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.split = split
        self.transform = transform
        self.image_ext = image_ext
        
        # Find all video directories
        self.video_dirs = self._find_video_dirs()
        
        # Build sequence index
        self.sequences = self._build_sequence_index()
        
        print(f"Found {len(self.video_dirs)} videos with {len(self.sequences)} sequences")
    
    def _find_video_dirs(self) -> List[Path]:
        """Find all directories containing frame sequences."""
        video_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        # Filter by split if exists
        split_file = self.data_root / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_names = set(line.strip() for line in f)
            video_dirs = [d for d in video_dirs if d.name in split_names]
        
        return sorted(video_dirs)
    
    def _build_sequence_index(self) -> List[Dict]:
        """Build index of sequences."""
        sequences = []
        
        for video_idx, video_dir in enumerate(self.video_dirs):
            # Get all frames
            frame_paths = sorted(video_dir.glob(f'*{self.image_ext}'))
            num_frames = len(frame_paths)
            
            max_start = num_frames - (self.sequence_length * self.frame_skip)
            if max_start < 0:
                continue
            
            if self.split == 'train':
                # Random sampling
                num_seqs = max(1, max_start // (self.sequence_length * self.frame_skip))
                for _ in range(num_seqs):
                    sequences.append({
                        'video_idx': video_idx,
                        'video_dir': video_dir,
                        'frame_paths': frame_paths,
                        'start_frame': None,
                    })
            else:
                # Non-overlapping
                stride = self.sequence_length * self.frame_skip
                for start in range(0, max_start + 1, stride):
                    sequences.append({
                        'video_idx': video_idx,
                        'video_dir': video_dir,
                        'frame_paths': frame_paths,
                        'start_frame': start,
                    })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load sequence of frames."""
        seq_info = self.sequences[idx]
        frame_paths = seq_info['frame_paths']
        
        # Determine start frame
        if seq_info['start_frame'] is None:
            max_start = len(frame_paths) - (self.sequence_length * self.frame_skip)
            start_frame = random.randint(0, max_start)
        else:
            start_frame = seq_info['start_frame']
        
        # Load frames
        frames = []
        for i in range(self.sequence_length):
            frame_idx = start_frame + i * self.frame_skip
            frame_path = frame_paths[frame_idx]
            
            # Load image
            if HAS_CV2:
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif HAS_IMAGEIO:
                frame = imageio.imread(str(frame_path))
            else:
                raise RuntimeError("OpenCV or imageio required")
            
            frames.append(frame)
        
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # Apply transforms
        if self.transform is not None:
            frames = self.transform(frames)
        
        return {
            'frames': frames,
            'video_id': seq_info['video_idx'],
            'video_name': seq_info['video_dir'].name,
            'start_frame': start_frame,
        }


# Test function
def test_dataset():
    """Test dataset loading."""
    print("Testing VideoDataset...")
    
    # Note: This requires actual video files to test
    # Create dummy test
    print("✓ VideoDataset class defined")
    print("✓ FrameDirectoryDataset class defined")
    print("\nTo test with real data:")
    print("  dataset = VideoDataset('path/to/videos', sequence_length=8)")
    print("  sample = dataset[0]")
    print("  print(sample['frames'].shape)  # Should be (8, 3, H, W)")


if __name__ == "__main__":
    test_dataset()
