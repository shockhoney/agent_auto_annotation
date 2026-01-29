"""
Data validator module.
Validates converted data and tests data loader compatibility.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from utils.file_utils import load_json, load_yaml
from utils.logger import logger


class DataValidator:
    """Validates converted data and creates data loader verification."""
    
    def __init__(self, data_root: str, data_type: str):
        """
        Initialize validator.
        
        Args:
            data_root: Root directory of converted data
            data_type: Type of data ('2d', '3d', '4d')
        """
        self.data_root = Path(data_root)
        self.data_type = data_type
        self.validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def validate(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Dict: Validation results
        """
        logger.info(f"Validating {self.data_type} dataset...")
        
        # Check directory structure
        self._validate_directory_structure()
        
        # Check configuration files
        self._validate_config_files()
        
        # Type-specific validation
        if self.data_type == '2d':
            self._validate_2d()
        elif self.data_type == '3d':
            self._validate_3d()
        elif self.data_type == '4d':
            self._validate_4d()
        
        # Try to load a sample
        self._test_sample_loading()
        
        # Summary
        self._print_summary()
        
        return self.validation_results
    
    def _validate_directory_structure(self) -> None:
        """Validate directory structure exists."""
        if not self.data_root.exists():
            self._add_failure(f"Data root does not exist: {self.data_root}")
            return
        
        self._add_success("Data root directory exists")
    
    def _validate_config_files(self) -> None:
        """Validate configuration files exist and are valid."""
        # Check data_config.yaml
        config_path = self.data_root / "data_config.yaml"
        if config_path.exists():
            try:
                config = load_yaml(str(config_path))
                self._add_success("data_config.yaml is valid")
            except Exception as e:
                self._add_failure(f"data_config.yaml is invalid: {e}")
        else:
            self._add_warning("data_config.yaml not found")
        
        # Skip metadata.json check - file no longer generated
    
    def _validate_2d(self) -> None:
        """Validate 2D COCO format."""
        # Check images directory
        images_dir = self.data_root / "images"
        if not images_dir.exists():
            self._add_failure("Images directory not found")
            return
        
        image_files = list(images_dir.glob("*.[jp][pn]g"))
        self._add_success(f"Found {len(image_files)} images")
        
        # Check annotations file
        ann_path = self.data_root / "annotations.json"
        if not ann_path.exists():
            self._add_failure("annotations.json not found")
            return
        
        try:
            ann_data = load_json(str(ann_path))
            
            # Validate COCO structure
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in ann_data:
                    self._add_failure(f"Missing required key in COCO annotations: {key}")
                else:
                    self._add_success(f"COCO annotations has '{key}' ({len(ann_data[key])} items)")
        
        except Exception as e:
            self._add_failure(f"Failed to validate annotations: {e}")
    
    def _validate_3d(self) -> None:
        """Validate 3D KITTI format."""
        # Check velodyne directory
        velodyne_dir = self.data_root / "velodyne"
        if not velodyne_dir.exists():
            self._add_failure("velodyne directory not found")
            return
        
        bin_files = list(velodyne_dir.glob("*.bin"))
        self._add_success(f"Found {len(bin_files)} velodyne files")
        
        # Check calib directory
        calib_dir = self.data_root / "calib"
        if not calib_dir.exists():
            self._add_failure("calib directory not found")
            return
        
        calib_files = list(calib_dir.glob("*.txt"))
        self._add_success(f"Found {len(calib_files)} calibration files")
        
        # Check matching
        if len(bin_files) != len(calib_files):
            self._add_warning(f"Mismatch: {len(bin_files)} velodyne vs {len(calib_files)} calib files")
    
    def _validate_4d(self) -> None:
        """Validate 4D Waymo-style format."""
        # Check for sequence directories
        seq_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and 'sequence' in d.name.lower()]
        
        if not seq_dirs:
            self._add_failure("No sequence directories found")
            return
        
        self._add_success(f"Found {len(seq_dirs)} sequences")
        
        # Validate first sequence
        first_seq = seq_dirs[0]
        timestamp_dirs = [d for d in first_seq.iterdir() if d.is_dir() and 'timestamp' in d.name]
        
        if timestamp_dirs:
            self._add_success(f"Sequence '{first_seq.name}' has {len(timestamp_dirs)} frames")
        else:
            self._add_failure(f"No timestamp directories in sequence '{first_seq.name}'")
    
    def _test_sample_loading(self) -> None:
        """Test loading a sample from the dataset."""
        try:
            if self.data_type == '2d':
                self._load_2d_sample()
            elif self.data_type == '3d':
                self._load_3d_sample()
            elif self.data_type == '4d':
                self._load_4d_sample()
        except Exception as e:
            self._add_failure(f"Failed to load sample: {e}")
    
    def _load_2d_sample(self) -> None:
        """Try to load a 2D sample."""
        images_dir = self.data_root / "images"
        images = list(images_dir.glob("*.[jp][pn]g"))
        
        if images:
            try:
                from PIL import Image
                img = Image.open(images[0])
                self._add_success(f"Successfully loaded sample image: {images[0].name} ({img.size})")
            except ImportError:
                self._add_warning("Pillow not installed, skipping image loading test")
    
    def _load_3d_sample(self) -> None:
        """Try to load a 3D sample."""
        velodyne_dir = self.data_root / "velodyne"
        bin_files = list(velodyne_dir.glob("*.bin"))
        
        if bin_files:
            try:
                import numpy as np
                points = np.fromfile(bin_files[0], dtype=np.float32).reshape(-1, 4)
                self._add_success(f"Successfully loaded sample point cloud: {bin_files[0].name} ({points.shape[0]} points)")
            except Exception as e:
                self._add_failure(f"Failed to load point cloud: {e}")
    
    def _load_4d_sample(self) -> None:
        """Try to load a 4D sample."""
        seq_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and 'sequence' in d.name.lower()]
        
        if seq_dirs:
            first_seq = seq_dirs[0]
            timestamp_dirs = [d for d in first_seq.iterdir() if d.is_dir()]
            
            if timestamp_dirs:
                frame_dir = timestamp_dirs[0]
                files = list(frame_dir.glob("*"))
                self._add_success(f"Successfully accessed sample frame: {frame_dir.name} ({len(files)} files)")
    
    def _add_success(self, message: str) -> None:
        """Add a success message."""
        self.validation_results['passed'].append(message)
        logger.info(f"✓ {message}")
    
    def _add_failure(self, message: str) -> None:
        """Add a failure message."""
        self.validation_results['failed'].append(message)
        logger.error(f"✗ {message}")
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.validation_results['warnings'].append(message)
        logger.warning(f"⚠ {message}")
    
    def _print_summary(self) -> None:
        """Print validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✓ Passed:   {len(self.validation_results['passed'])}")
        logger.info(f"⚠ Warnings: {len(self.validation_results['warnings'])}")
        logger.info(f"✗ Failed:   {len(self.validation_results['failed'])}")
        logger.info("=" * 60)
