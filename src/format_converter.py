"""
Main format converter orchestrator.
Routes data to appropriate converters based on detected type.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from inspect_data import DataInspector
from converters import COCOConverter, KITTIConverter, WaymoConverter
from utils.logger import logger
from utils.file_utils import ensure_dir


class FormatConverter:
    """Main orchestrator for format conversion based on data type."""
    
    def __init__(self, input_dir: str, output_dir: str, force_type: Optional[str] = None):
        """
        Initialize format converter.
        
        Args:
            input_dir: Input directory containing raw data
            output_dir: Output directory for converted data
            force_type: Optional forced data type ('2d', '3d', '4d')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.force_type = force_type
        
        ensure_dir(self.output_dir)
        
        self.inspection_result = None
        self.conversion_result = None
    
    def convert(self) -> Dict[str, Any]:
        """
        Main conversion method - inspects data and converts to appropriate format.
        
        Returns:
            Dict: Conversion results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Format Conversion Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Inspect data
        logger.info("\n[Step 1/3] Inspecting input data...")
        inspector = DataInspector(str(self.input_dir))
        self.inspection_result = inspector.inspect()
        
        # Save inspection report
        inspection_report_path = self.output_dir / "inspection_report.json"
        inspector.save_report(str(inspection_report_path))
        
        # Determine data type
        if self.force_type:
            data_type = self.force_type
            logger.info(f"Using forced data type: {data_type}")
        else:
            inferred = self.inspection_result['inferred_type']
            data_type = inferred.split('_')[0]  # Extract '2d', '3d', or '4d'
            logger.info(f"Detected data type: {inferred}")
        
        # Step 2: Convert based on type
        logger.info(f"\n[Step 2/3] Converting to standardized format...")
        
        if data_type == '2d':
            self.conversion_result = self._convert_2d()
        elif data_type == '3d':
            self.conversion_result = self._convert_3d()
        elif data_type == '4d':
            self.conversion_result = self._convert_4d()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Step 3: Summary
        logger.info(f"\n[Step 3/3] Conversion complete!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Conversion statistics: {self.conversion_result}")
        logger.info("=" * 60)
        
        return {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'data_type': data_type,
            'inspection': self.inspection_result,
            'conversion': self.conversion_result
        }
    
    def _convert_2d(self) -> Dict[str, Any]:
        """Convert 2D image data to COCO format."""
        logger.info("Converting to COCO format...")
        
        output_2d = self.output_dir / "pre_labeling_2d"
        converter = COCOConverter(str(output_2d))
        
        # Find image directories
        image_dirs = self._find_image_directories()
        
        if not image_dirs:
            # Use entire input directory
            stats = converter.convert_from_directory(str(self.input_dir))
        else:
            # Convert from first found image directory
            stats = converter.convert_from_directory(str(image_dirs[0]))
        
        # Save COCO annotations
        converter.save()
        
        logger.info(f"✓ Converted {stats['total_images']} images")
        logger.info(f"✓ Created {stats['total_annotations']} annotations")
        logger.info(f"✓ Identified {stats['total_categories']} categories")
        
        return stats
    
    def _convert_3d(self) -> Dict[str, Any]:
        """Convert 3D point cloud data to KITTI format."""
        logger.info("Converting to KITTI format...")
        
        output_3d = self.output_dir / "pre_labeling_3d"
        converter = KITTIConverter(str(output_3d))
        
        # Find point cloud directories
        pc_dirs = self._find_pointcloud_directories()
        
        if not pc_dirs:
            # Use entire input directory
            stats = converter.convert_from_directory(str(self.input_dir))
        else:
            # Convert from first found point cloud directory
            stats = converter.convert_from_directory(str(pc_dirs[0]))
        
        logger.info(f"✓ Converted {stats['total_frames']} point cloud frames")
        logger.info(f"✓ Created {stats['velodyne_files']} velodyne files")
        logger.info(f"✓ Created {stats['calib_files']} calibration files")
        
        return stats
    
    def _convert_4d(self) -> Dict[str, Any]:
        """Convert 4D sequence data to Waymo-style format."""
        logger.info("Converting to Waymo-style format...")
        
        output_4d = self.output_dir / "pre_labeling_4d"
        converter = WaymoConverter(str(output_4d))
        
        # Convert from input directory
        stats = converter.convert_from_directory(str(self.input_dir))
        
        logger.info(f"✓ Created {stats['total_sequences']} sequences")
        logger.info(f"✓ Converted {stats['total_frames']} total frames")
        
        return stats
    
    def _find_image_directories(self) -> list:
        """Find directories containing images."""
        dirs = []
        for d in self.input_dir.rglob('*'):
            if d.is_dir():
                # Check if directory contains images
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
                has_images = any((d / f"*{ext}").exists() for ext in image_exts)
                if has_images or 'image' in d.name.lower():
                    dirs.append(d)
        return dirs or [self.input_dir]
    
    def _find_pointcloud_directories(self) -> list:
        """Find directories containing point clouds."""
        dirs = []
        for d in self.input_dir.rglob('*'):
            if d.is_dir():
                # Check if directory contains point clouds
                pc_exts = ['.pcd', '.bin', '.ply', '.las']
                has_pcs = any((d / f"*{ext}").exists() for ext in pc_exts)
                if has_pcs or 'velodyne' in d.name.lower() or 'lidar' in d.name.lower():
                    dirs.append(d)
        return dirs or [self.input_dir]
