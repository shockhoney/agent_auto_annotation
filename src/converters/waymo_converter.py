"""
Waymo-style format converter for 4D sequence data.
Organizes temporal sequences with timestamp-based directory structure.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from utils.file_utils import ensure_dir, list_files, copy_file, save_json
from utils.logger import logger


class WaymoConverter:
    """Converts 4D sequence data to Waymo-style timestamp-organized format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize Waymo-style converter.
        
        Args:
            output_dir: Output directory for converted data
        """
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        self.sequences = {}
        self.current_sequence = None
    
    def create_sequence(self, sequence_name: str) -> str:
        """
        Create a new sequence.
        
        Args:
            sequence_name: Name of the sequence
        
        Returns:
            str: Sequence directory path
        """
        seq_dir = self.output_dir / sequence_name
        ensure_dir(seq_dir)
        
        self.sequences[sequence_name] = {
            'path': seq_dir,
            'frames': []
        }
        self.current_sequence = sequence_name
        
        return str(seq_dir)
    
    def add_frame(self, timestamp: int, lidar_path: Optional[str] = None,
                  camera_paths: Optional[Dict[str, str]] = None,
                  ego_pose: Optional[Dict[str, Any]] = None,
                  sequence_name: Optional[str] = None) -> str:
        """
        Add a frame to a sequence.
        
        Args:
            timestamp: Frame timestamp (unix timestamp or frame number)
            lidar_path: Optional path to lidar point cloud file
            camera_paths: Optional dict of camera_name -> image_path
            ego_pose: Optional ego vehicle pose information
            sequence_name: Sequence to add to (uses current if not specified)
        
        Returns:
            str: Frame directory path
        """
        # Determine which sequence to use
        if sequence_name is None:
            sequence_name = self.current_sequence
        
        if sequence_name is None:
            raise ValueError("No sequence specified or current sequence set")
        
        if sequence_name not in self.sequences:
            self.create_sequence(sequence_name)
        
        # Create frame directory
        frame_dir_name = f"timestamp_{timestamp}"
        frame_dir = self.sequences[sequence_name]['path'] / frame_dir_name
        ensure_dir(frame_dir)
        
        # Copy lidar data
        if lidar_path:
            src = Path(lidar_path)
            dst = frame_dir / f"lidar{src.suffix}"
            copy_file(str(src), str(dst))
        
        # Copy camera images
        if camera_paths:
            for camera_name, img_path in camera_paths.items():
                src = Path(img_path)
                dst = frame_dir / f"camera_{camera_name}{src.suffix}"
                copy_file(str(src), str(dst))
        
        # Save ego pose
        if ego_pose:
            pose_path = frame_dir / "ego_pose.json"
            save_json(ego_pose, str(pose_path))
        
        # Record frame info
        frame_info = {
            'timestamp': timestamp,
            'directory': frame_dir_name,
            'has_lidar': lidar_path is not None,
            'cameras': list(camera_paths.keys()) if camera_paths else [],
            'has_ego_pose': ego_pose is not None
        }
        self.sequences[sequence_name]['frames'].append(frame_info)
        
        return str(frame_dir)
    
    def convert_from_directory(self, root_dir: str) -> Dict[str, Any]:
        """
        Convert data from a directory structure.
        Tries to auto-detect sequences and timestamps.
        
        Args:
            root_dir: Root directory containing sequences
        
        Returns:
            Dict: Conversion statistics
        """
        root = Path(root_dir)
        
        # Try to detect sequences (subdirectories)
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        
        if not subdirs:
            # No subdirectories, treat entire folder as one sequence
            subdirs = [root]
            sequence_names = ['sequence_A']
        else:
            sequence_names = [d.name for d in subdirs]
        
        total_frames = 0
        
        for subdir, seq_name in zip(subdirs, sequence_names):
            self.create_sequence(seq_name)
            
            # Detect frames in this sequence
            frames = self._detect_frames_in_directory(subdir)
            
            for frame_data in frames:
                self.add_frame(**frame_data, sequence_name=seq_name)
                total_frames += 1
        
        # Save metadata
        metadata = {
            "format": "Waymo-style",
            "num_sequences": len(self.sequences),
            "total_frames": total_frames,
            "sequences": {
                name: {
                    "num_frames": len(info['frames']),
                    "frame_list": info['frames']
                }
                for name, info in self.sequences.items()
            }
        }
        save_json(metadata, str(self.output_dir / "metadata.json"))
        
        return {
            "total_sequences": len(self.sequences),
            "total_frames": total_frames
        }
    
    def _detect_frames_in_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Auto-detect frames in a directory.
        
        Args:
            directory: Directory to scan
        
        Returns:
            List[Dict]: List of frame data dictionaries
        """
        frames = []
        
        # Look for point clouds
        lidar_files = list_files(str(directory), 
                                extensions=['bin', 'pcd', 'ply'],
                                recursive=False)
        
        # Look for images
        image_files = list_files(str(directory),
                               extensions=['jpg', 'jpeg', 'png'],
                               recursive=False)
        
        # Try to match files by similar names (timestamps/frame numbers)
        if lidar_files:
            for i, lidar_path in enumerate(lidar_files):
                timestamp = self._extract_timestamp_from_filename(lidar_path.stem)
                
                # Find corresponding images
                camera_paths = {}
                for img_path in image_files:
                    img_ts = self._extract_timestamp_from_filename(img_path.stem)
                    if timestamp == img_ts or abs(timestamp - img_ts) < 10:  # Within 10 units
                        camera_name = self._extract_camera_name(img_path.stem)
                        camera_paths[camera_name] = str(img_path)
                
                frames.append({
                    'timestamp': timestamp,
                    'lidar_path': str(lidar_path),
                    'camera_paths': camera_paths if camera_paths else None
                })
        
        elif image_files:
            # Only images, no lidar
            processed_timestamps = set()
            for img_path in image_files:
                timestamp = self._extract_timestamp_from_filename(img_path.stem)
                
                if timestamp not in processed_timestamps:
                    camera_paths = {}
                    for other_img in image_files:
                        other_ts = self._extract_timestamp_from_filename(other_img.stem)
                        if timestamp == other_ts:
                            camera_name = self._extract_camera_name(other_img.stem)
                            camera_paths[camera_name] = str(other_img)
                    
                    frames.append({
                        'timestamp': timestamp,
                        'camera_paths': camera_paths
                    })
                    processed_timestamps.add(timestamp)
        
        return sorted(frames, key=lambda x: x['timestamp'])
    
    def _extract_timestamp_from_filename(self, filename: str) -> int:
        """Extract timestamp or frame number from filename."""
        # Try to find numbers in filename
        import re
        numbers = re.findall(r'\d+', filename)
        
        if numbers:
            # Use the longest number as timestamp
            return int(max(numbers, key=len))
        
        return 0
    
    def _extract_camera_name(self, filename: str) -> str:
        """Extract camera name from filename."""
        filename_lower = filename.lower()
        
        # Common camera names
        camera_keywords = {
            'front': 'front',
            'rear': 'rear',
            'back': 'rear',
            'left': 'left',
            'right': 'right',
            'top': 'top'
        }
        
        for keyword, name in camera_keywords.items():
            if keyword in filename_lower:
                return name
        
        return 'default'
