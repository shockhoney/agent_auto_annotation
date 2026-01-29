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
    
    def __init__(self, output_dir: str, calib_dir: Optional[str] = None, 
                 camera_name: str = 'camera_front_wide', image_root: Optional[str] = None):
        """
        Initialize KITTI converter.
        
        Args:
            output_dir: Output directory for converted data
            calib_dir: Optional directory containing YAML calibration files
            camera_name: Name of the camera to use (default: camera_front_wide)
            image_root: Optional root directory containing image subdirectories
        """
        self.output_dir = Path(output_dir)
        
        # KITTI directory structure
        self.velodyne_dir = self.output_dir / "velodyne"
        self.calib_dir = self.output_dir / "calib"
        self.label_dir = self.output_dir / "label"
        self.image_dir = self.output_dir / "image_2"  # Main camera images
        
        ensure_dir(self.velodyne_dir)
        ensure_dir(self.calib_dir)
        ensure_dir(self.label_dir)
        ensure_dir(self.image_dir)
        
        # Calibration and image settings
        self.calib_dir_src = Path(calib_dir) if calib_dir else None
        self.camera_name = camera_name
        self.image_root = Path(image_root) if image_root else None
        self.yaml_calib = None
        
        # Load YAML calibration if provided
        if self.calib_dir_src:
            self._load_camera_calibration()
        
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
        
        # Create calibration file (real or default)
        if self.yaml_calib:
            self._create_calib_from_yaml(frame_name)
        else:
            self._create_default_calib(frame_name)
        
        # Copy matched image if image root is provided
        if self.image_root:
            pc_timestamp = self._extract_timestamp_from_path(src_path)
            if pc_timestamp is not None:
                self._copy_matched_image(pc_timestamp, frame_name)
        
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
        
        # Skip metadata.json - information embedded in data structure
        
        return {
            "total_frames": converted,
            "velodyne_files": len(list(self.velodyne_dir.glob("*.bin"))),
            "calib_files": len(list(self.calib_dir.glob("*.txt"))),
            "image_files": len(list(self.image_dir.glob("*.[jp][pn]g")))
        }
    
    def _load_camera_calibration(self) -> None:
        """Load camera calibration from YAML file."""
        import yaml
        
        calib_file = self.calib_dir_src / f"{self.camera_name}.yaml"
        if not calib_file.exists():
            logger.warning(f"Calibration file not found: {calib_file}, using defaults")
            return
        
        try:
            # Read file and skip %YAML:1.0 directive line
            with open(calib_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip lines starting with % (YAML directives)
                content = ''.join(line for line in lines if not line.strip().startswith('%'))
                self.yaml_calib = yaml.safe_load(content)
            logger.info(f"Loaded calibration from: {calib_file}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.yaml_calib = None
    
    def _rodrigues_to_matrix(self, rvec: list) -> np.ndarray:
        """Convert rotation vector to rotation matrix (Rodrigues formula)."""
        rvec = np.array(rvec, dtype=np.float64)
        theta = np.linalg.norm(rvec)
        
        if theta < 1e-10:
            return np.eye(3)
        
        r = rvec / theta
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        return R
    
    def _create_calib_from_yaml(self, frame_name: str) -> None:
        """Create KITTI calibration file from YAML parameters."""
        if not self.yaml_calib:
            self._create_default_calib(frame_name)
            return
        
        calib_path = self.calib_dir / f"{frame_name}.txt"
        
        try:
            # Extract parameters from YAML
            fx = self.yaml_calib.get('fx', 1.0)
            fy = self.yaml_calib.get('fy', 1.0)
            cx = self.yaml_calib.get('cx', 0.0)
            cy = self.yaml_calib.get('cy', 0.0)
            
            # Build projection matrix P2 (3x4)
            P2 = f"P2: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0"
            
            # Use P2 for all cameras (simplified)
            P0 = f"P0: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0"
            P1 = f"P1: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0"
            P3 = f"P3: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0"
            
            # Rectification rotation (identity)
            R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
            
            # Extract rotation and translation for Tr_velo_to_cam
            r_s2b = self.yaml_calib.get('r_s2b', [0, 0, 0])
            t_s2b = self.yaml_calib.get('t_s2b', [0, 0, 0])
            
            # Convert rotation vector to matrix
            R = self._rodrigues_to_matrix(r_s2b)
            t = np.array(t_s2b)
            
            # Build Tr_velo_to_cam (3x4 transformation matrix)
            Tr_velo_to_cam = (
                f"Tr_velo_to_cam: {R[0,0]} {R[0,1]} {R[0,2]} {t[0]} "
                f"{R[1,0]} {R[1,1]} {R[1,2]} {t[1]} "
                f"{R[2,0]} {R[2,1]} {R[2,2]} {t[2]}"
            )
            
            # Default Tr_imu_to_velo
            Tr_imu_to_velo = "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0"
            
            # Write calibration file
            with open(calib_path, 'w') as f:
                f.write(f"{P0}\n{P1}\n{P2}\n{P3}\n")
                f.write(f"{R0_rect}\n")
                f.write(f"{Tr_velo_to_cam}\n")
                f.write(f"{Tr_imu_to_velo}\n")
            
            logger.debug(f"Created calibration with real parameters: {calib_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create calibration from YAML: {e}, using defaults")
            self._create_default_calib(frame_name)
    
    def _extract_timestamp_from_path(self, path: Path) -> Optional[float]:
        """Extract timestamp from point cloud filename."""
        try:
            # Filename format: "1744693431.39.pcd"
            stem = path.stem  # "1744693431.39"
            return float(stem)
        except Exception as e:
            logger.debug(f"Could not extract timestamp from {path.name}: {e}")
            return None
    
    def _copy_matched_image(self, pc_timestamp: float, frame_name: str) -> None:
        """Find and copy the closest matching image."""
        # Image directory: sensor_camera_front_wide_video
        image_subdir = self.image_root / f"sensor_{self.camera_name}_video"
        
        if not image_subdir.exists():
            logger.warning(f"Image directory not found: {image_subdir}")
            return
        
        # Find all images
        images = list(image_subdir.glob('*.jpg')) + list(image_subdir.glob('*.png'))
        
        if not images:
            logger.warning(f"No images found in {image_subdir}")
            return
        
        # Find closest image by timestamp
        closest_image = None
        min_diff = float('inf')
        
        for img_path in images:
            try:
                # Extract timestamp from filename: "1744693432.191269.jpg"
                img_ts = float(img_path.stem)
                diff = abs(img_ts - pc_timestamp)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_image = img_path
            except:
                continue
        
        if closest_image and min_diff < 0.1:  # 100ms threshold
            # Copy image to KITTI image_2 directory
            import shutil
            dst_path = self.image_dir / f"{frame_name}{closest_image.suffix}"
            shutil.copy2(str(closest_image), str(dst_path))
            logger.debug(f"Copied image: {closest_image.name} -> {dst_path.name} (Δt={min_diff*1000:.1f}ms)")
        else:
            logger.warning(f"No matching image found for timestamp {pc_timestamp} (min_diff={min_diff:.3f}s)")

