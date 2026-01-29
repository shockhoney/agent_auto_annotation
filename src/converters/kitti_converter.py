"""
KITTI format converter for 3D point cloud data.
Converts various 3D point cloud formats to KITTI standard structure.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from utils.file_utils import ensure_dir, list_files, save_json
from utils.logger import logger


class KITTIConverter:
    """Converts 3D point cloud data to KITTI format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize KITTI converter.
        
        Args:
            output_dir: Output directory for converted data
        """
        self.output_dir = Path(output_dir)
        
        # KITTI directory structure
        self.velodyne_dir = self.output_dir / "velodyne"
        self.calib_dir = self.output_dir / "calib"
        self.label_dir = self.output_dir / "label"
        
        ensure_dir(self.velodyne_dir)
        ensure_dir(self.calib_dir)
        ensure_dir(self.label_dir)
        
        self.frame_id = 0
    
    def add_pointcloud(self, pointcloud_path: str, frame_name: Optional[str] = None) -> str:
        """
        Add a point cloud to KITTI dataset.
        
        Args:
            pointcloud_path: Path to source point cloud
            frame_name: Optional custom frame name (otherwise auto-generated)
        
        Returns:
            str: Frame name/ID
        """
        src_path = Path(pointcloud_path)
        if not src_path.exists():
            logger.warning(f"Point cloud not found: {pointcloud_path}")
            return ""
        
        # Generate frame name
        if frame_name is None:
            frame_name = f"{self.frame_id:06d}"
        
        # Load and convert point cloud
        points = self._load_pointcloud(src_path)
        
        if points is None:
            logger.warning(f"Failed to load point cloud: {pointcloud_path}")
            return ""
        
        # Save in KITTI binary format
        output_path = self.velodyne_dir / f"{frame_name}.bin"
        self._save_kitti_bin(points, output_path)
        
        # Create default calibration file
        self._create_default_calib(frame_name)
        
        self.frame_id += 1
        return frame_name
    
    def _load_pointcloud(self, path: Path) -> Optional[np.ndarray]:
        """Load point cloud from various formats."""
        ext = path.suffix.lower()
        
        try:
            if ext == '.bin':
                # Already in binary format
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                return points
            
            elif ext == '.pcd':
                # PCD format
                return self._load_pcd(path)
            
            elif ext == '.ply':
                # PLY format
                return self._load_ply(path)
            
            elif ext in ['.txt', '.xyz']:
                # Simple text format
                points = np.loadtxt(path, dtype=np.float32)
                # Ensure 4 columns (x, y, z, intensity)
                if points.shape[1] == 3:
                    points = np.hstack([points, np.zeros((points.shape[0], 1))])
                return points
            
            else:
                logger.warning(f"Unsupported point cloud format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading point cloud {path}: {e}")
            return None
    
    def _load_pcd(self, path: Path) -> Optional[np.ndarray]:
        """Load PCD format point cloud."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points, dtype=np.float32)
            
            # Add intensity channel (zeros if not available)
            if not pcd.colors:
                intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
            else:
                # Use first color channel as intensity
                intensity = np.asarray(pcd.colors)[:, 0:1].astype(np.float32)
            
            return np.hstack([points, intensity])
        except ImportError:
            logger.warning("Open3D not installed, using fallback PCD parser")
            return self._load_pcd_fallback(path)
    
    def _load_pcd_fallback(self, path: Path) -> Optional[np.ndarray]:
        """Fallback PCD loader without Open3D."""
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA'):
                data_start = i + 1
                break
        
        # Parse points
        points = []
        for line in lines[data_start:]:
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) >= 3:
                    # Take first 4 values or pad with zeros
                    points.append(coords[:4] if len(coords) >= 4 else coords + [0] * (4 - len(coords)))
            except:
                continue
        
        return np.array(points, dtype=np.float32) if points else None
    
    def _load_ply(self, path: Path) -> Optional[np.ndarray]:
        """Load PLY format point cloud."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points, dtype=np.float32)
            intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
            return np.hstack([points, intensity])
        except ImportError:
            logger.warning("Open3D required for PLY format")
            return None
    
    def _save_kitti_bin(self, points: np.ndarray, output_path: Path) -> None:
        """Save points in KITTI binary format."""
        # Ensure 4 columns (x, y, z, intensity)
        if points.shape[1] < 4:
            padding = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, padding])
        elif points.shape[1] > 4:
            points = points[:, :4]
        
        points.astype(np.float32).tofile(output_path)
        logger.debug(f"Saved point cloud: {output_path}")
    
    def _create_default_calib(self, frame_name: str) -> None:
        """Create default calibration file for a frame."""
        # Create default calibration (identity matrices)
        calib_path = self.calib_dir / f"{frame_name}.txt"
        
        # Default calibration matrices
        P0 = "P0: 1 0 0 0 0 1 0 0 0 0 1 0"
        P1 = "P1: 1 0 0 0 0 1 0 0 0 0 1 0"
        P2 = "P2: 1 0 0 0 0 1 0 0 0 0 1 0"
        P3 = "P3: 1 0 0 0 0 1 0 0 0 0 1 0"
        R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
        Tr_velo_to_cam = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0"
        Tr_imu_to_velo = "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0"
        
        with open(calib_path, 'w') as f:
            f.write(f"{P0}\n{P1}\n{P2}\n{P3}\n")
            f.write(f"{R0_rect}\n")
            f.write(f"{Tr_velo_to_cam}\n")
            f.write(f"{Tr_imu_to_velo}\n")
    
    def convert_from_directory(self, pointcloud_dir: str) -> Dict[str, Any]:
        """
        Convert point clouds from a directory.
        
        Args:
            pointcloud_dir: Directory containing point cloud files
        
        Returns:
            Dict: Conversion statistics
        """
        pc_files = list_files(pointcloud_dir, 
                             extensions=['bin', 'pcd', 'ply', 'las', 'txt', 'xyz'],
                             recursive=True)
        
        logger.info(f"Found {len(pc_files)} point clouds to convert")
        
        converted = 0
        for pc_path in pc_files:
            frame_name = self.add_pointcloud(str(pc_path))
            if frame_name:
                converted += 1
        
        # Save metadata
        metadata = {
            "format": "KITTI",
            "num_frames": converted,
            "velodyne_dir": str(self.velodyne_dir.relative_to(self.output_dir)),
            "calib_dir": str(self.calib_dir.relative_to(self.output_dir)),
        }
        save_json(metadata, str(self.output_dir / "metadata.json"))
        
        return {
            "total_frames": converted,
            "velodyne_files": len(list(self.velodyne_dir.glob("*.bin"))),
            "calib_files": len(list(self.calib_dir.glob("*.txt")))
        }
