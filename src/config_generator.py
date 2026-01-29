"""
Configuration file generator for model integration.
Generates data_config.yaml and dataset manifests.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

from utils.file_utils import save_yaml, save_json, load_json, ensure_dir
from utils.logger import logger


class ConfigGenerator:
    """Generates configuration files for seamless model integration."""
    
    def __init__(self, data_root: str, data_type: str):
        """
        Initialize config generator.
        
        Args:
            data_root: Root directory of standardized data
            data_type: Type of data ('2d', '3d', '4d')
        """
        self.data_root = Path(data_root)
        self.data_type = data_type
        self.config = {}
    
    def generate_config(self, train_split: float = 0.8, skip_split: bool = False) -> Dict[str, Any]:
        """
        Generate complete configuration based on data type.
        
        Args:
            train_split: Fraction of data to use for training (default: 0.8)
            skip_split: If True, skip train/val split (for inference mode)
        
        Returns:
            Dict: Generated configuration
        """
        logger.info(f"Generating configuration for {self.data_type} data...")
        
        if skip_split:
            logger.info("推理模式：跳过train/val划分")
        
        # Load metadata if available
        metadata_path = self.data_root / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = load_json(str(metadata_path))
        
        # Generate base config
        self.config = {
            'task_type': self._get_task_type(),
            'data_root': str(self.data_root),
            'format': metadata.get('format', self._get_default_format()),
        }
        
        # Add type-specific configurations
        if self.data_type == '2d':
            self._configure_2d(metadata, train_split, skip_split)
        elif self.data_type == '3d':
            self._configure_3d(metadata, train_split, skip_split)
        elif self.data_type == '4d':
            self._configure_4d(metadata, train_split, skip_split)
        
        # Save configuration
        config_path = self.data_root / "data_config.yaml"
        save_yaml(self.config, str(config_path))
        logger.info(f"✓ Configuration saved to: {config_path}")
        
        # Skip manifest generation - information already in data_config.yaml
        
        return self.config
    
    def _get_task_type(self) -> str:
        """Get task type based on data type."""
        task_map = {
            '2d': '2D_Detection',
            '3d': '3D_Detection',
            '4d': '4D_Tracking'
        }
        return task_map.get(self.data_type, 'Unknown')
    
    def _get_default_format(self) -> str:
        """Get default format for data type."""
        format_map = {
            '2d': 'coco',
            '3d': 'kitti',
            '4d': 'waymo'
        }
        return format_map.get(self.data_type, 'unknown')
    
    def _configure_2d(self, metadata: Dict, train_split: float, skip_split: bool = False) -> None:
        """Configure for 2D image data."""
        # Add image-specific config
        self.config.update({
            'annotations_file': 'annotations.json',
            'images_dir': 'images',
        })
        
        # Extract class mapping from COCO annotations
        ann_path = self.data_root / "annotations.json"
        if ann_path.exists():
            ann_data = load_json(str(ann_path))
            
            # Create class map
            class_map = {}
            for cat in ann_data.get('categories', []):
                class_map[cat['name']] = cat['id'] - 1  # 0-indexed
            
            self.config['class_map'] = class_map
            self.config['num_classes'] = len(class_map)
            
            if not skip_split:
                # Create train/val split
                images = ann_data.get('images', [])
                split_idx = int(len(images) * train_split)
                
                train_ids = [img['id'] for img in images[:split_idx]]
                val_ids = [img['id'] for img in images[split_idx:]]
                
                # Save split files
                self._save_split_list(train_ids, 'train.txt')
                self._save_split_list(val_ids, 'val.txt')
                
                self.config['train_list'] = 'train.txt'
                self.config['val_list'] = 'val.txt'
    
    def _configure_3d(self, metadata: Dict, train_split: float, skip_split: bool = False) -> None:
        """Configure for 3D point cloud data."""
        # Add point cloud-specific config
        self.config.update({
            'velodyne_dir': 'velodyne',
            'calib_dir': 'calib',
            'label_dir': 'label',
        })
        
        # Get list of frames
        velodyne_dir = self.data_root / "velodyne"
        if velodyne_dir.exists():
            frames = sorted([f.stem for f in velodyne_dir.glob('*.bin')])
            
            if not skip_split:
                split_idx = int(len(frames) * train_split)
                train_frames = frames[:split_idx]
                val_frames = frames[split_idx:]
                
                # Save split files
                self._save_split_list(train_frames, 'train.txt')
                self._save_split_list(val_frames, 'val.txt')
                
                self.config['train_list'] = 'train.txt'
                self.config['val_list'] = 'val.txt'
            
            self.config['num_frames'] = len(frames)
        
        # Default class map for autonomous driving
        self.config['class_map'] = {
            'car': 0,
            'pedestrian': 1,
            'cyclist': 2
        }
        self.config['num_classes'] = 3
    
    def _configure_4d(self, metadata: Dict, train_split: float, skip_split: bool = False) -> None:
        """Configure for 4D sequence data."""
        # Add sequence-specific config
        sequences = metadata.get('sequences', {})
        
        self.config['sequences'] = list(sequences.keys())
        self.config['num_sequences'] = len(sequences)
        
        # Split sequences into train/val
        seq_names = list(sequences.keys())
        split_idx = int(len(seq_names) * train_split)
        
        self.config['train_sequences'] = seq_names[:split_idx]
        self.config['val_sequences'] = seq_names[split_idx:]
        
        # Frame information
        total_frames = sum(seq['num_frames'] for seq in sequences.values())
        self.config['total_frames'] = total_frames
        
        # Default class map
        self.config['class_map'] = {
            'vehicle': 0,
            'pedestrian': 1,
            'cyclist': 2
        }
        self.config['num_classes'] = 3
    
    def _save_split_list(self, items: List, filename: str) -> None:
        """Save train/val split list to file."""
        filepath = self.data_root / filename
        with open(filepath, 'w') as f:
            for item in items:
                f.write(f"{item}\n")
    
    def _generate_manifest(self, metadata: Dict) -> None:
        """Generate comprehensive dataset manifest."""
        manifest = {
            'dataset_info': {
                'name': f'Auto-converted {self.data_type.upper()} Dataset',
                'type': self.data_type,
                'format': self.config.get('format'),
                'root': str(self.data_root)
            },
            'statistics': {
                'num_classes': self.config.get('num_classes', 0),
                'class_names': list(self.config.get('class_map', {}).keys())
            },
            'config': self.config,
            'metadata': metadata
        }
        
        manifest_path = self.data_root / "dataset_manifest.json"
        save_json(manifest, str(manifest_path))
        logger.info(f"✓ Manifest saved to: {manifest_path}")


def generate_config(data_root: str, data_type: str, train_split: float = 0.8) -> Dict[str, Any]:
    """
    Convenience function to generate dataset configuration.
    
    Args:
        data_root: Root directory of standardized data
        data_type: Type of data ('2d', '3d', '4d')
        train_split: Train/val split ratio
    
    Returns:
        Dict: Generated configuration
    """
    generator = ConfigGenerator(data_root, data_type)
    return generator.generate_config(train_split)
