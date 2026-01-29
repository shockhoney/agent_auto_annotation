"""
Data inspection module for automatically detecting directory structure and data types.
This is the "eyes" of the agent - it understands the raw data layout.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import json

try:
    import numpy as np
    import cv2
    from PIL import Image
except ImportError:
    np = None
    cv2 = None
    Image = None

from utils.file_utils import list_files, get_file_extension
from utils.logger import logger


# File type categorization
FILE_TYPES = {
    '2d_image': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
    '3d_pointcloud': ['pcd', 'bin', 'ply', 'las', 'laz'],
    'annotation': ['json', 'xml', 'txt', 'csv', 'yaml', 'yml'],
    'calibration': ['txt', 'yaml', 'yml'],
    'video': ['mp4', 'avi', 'mov', 'mkv']
}


class DataInspector:
    """Inspects directory structure and identifies data types."""
    
    def __init__(self, root_dir: str, max_depth: int = 5, sample_size: int = 2):
        """
        Initialize data inspector.
        
        Args:
            root_dir: Root directory to inspect
            max_depth: Maximum depth for recursive scanning
            sample_size: Number of files to sample per directory
        """
        self.root_dir = Path(root_dir)
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.inspection_result = {}
    
    def inspect(self) -> Dict[str, Any]:
        """
        Main inspection method - analyzes directory structure and data types.
        
        Returns:
            Dict containing:
                - directory_tree: Hierarchical structure
                - file_counts: File counts by type and directory
                - samples: Metadata from sampled files
                - inferred_type: Detected dataset type (2d/3d/4d)
                - schema: Inferred annotation schema
        """
        logger.info(f"Starting inspection of: {self.root_dir}")
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {self.root_dir}")
        
        # Step 1: Build directory tree
        dir_tree = self._build_directory_tree()
        
        # Step 2: Count files by type
        file_counts = self._count_files_by_type()
        
        # Step 3: Sample files and extract metadata
        samples = self._sample_and_extract_metadata()
        
        # Step 4: Infer dataset type
        inferred_type = self._infer_dataset_type(file_counts)
        
        # Step 5: Infer annotation schema
        schema = self._infer_annotation_schema(samples)
        
        self.inspection_result = {
            'root_directory': str(self.root_dir),
            'directory_tree': dir_tree,
            'file_counts': file_counts,
            'samples': samples,
            'inferred_type': inferred_type,
            'schema': schema,
            'total_files': sum(file_counts.get('by_category', {}).values())
        }
        
        logger.info(f"Inspection complete. Detected type: {inferred_type}")
        return self.inspection_result
    
    def _build_directory_tree(self, path: Optional[Path] = None, depth: int = 0) -> Dict[str, Any]:
        """Build hierarchical directory tree."""
        if path is None:
            path = self.root_dir
        
        if depth > self.max_depth:
            return {'name': path.name, 'type': 'directory', 'truncated': True}
        
        tree = {
            'name': path.name,
            'type': 'directory',
            'children': []
        }
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items:
                if item.is_dir():
                    tree['children'].append(self._build_directory_tree(item, depth + 1))
                else:
                    tree['children'].append({
                        'name': item.name,
                        'type': 'file',
                        'extension': get_file_extension(str(item))
                    })
        except PermissionError:
            tree['error'] = 'Permission denied'
        
        return tree
    
    def _count_files_by_type(self) -> Dict[str, Any]:
        """Count files by category and directory."""
        counts = {
            'by_category': defaultdict(int),
            'by_directory': defaultdict(lambda: defaultdict(int)),
            'by_extension': defaultdict(int)
        }
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                ext = get_file_extension(str(file_path))
                counts['by_extension'][ext] += 1
                
                # Categorize file
                category = self._categorize_file(ext)
                counts['by_category'][category] += 1
                
                # Count by directory
                rel_dir = file_path.parent.relative_to(self.root_dir)
                counts['by_directory'][str(rel_dir)][category] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return {
            'by_category': dict(counts['by_category']),
            'by_directory': {k: dict(v) for k, v in counts['by_directory'].items()},
            'by_extension': dict(counts['by_extension'])
        }
    
    def _categorize_file(self, extension: str) -> str:
        """Categorize file by extension."""
        for category, exts in FILE_TYPES.items():
            if extension in exts:
                return category
        return 'unknown'
    
    def _sample_and_extract_metadata(self) -> List[Dict[str, Any]]:
        """Sample files from each directory and extract metadata."""
        samples = []
        
        # Get all subdirectories
        dirs = [self.root_dir] + [d for d in self.root_dir.rglob('*') if d.is_dir()]
        
        for directory in dirs:
            # Get files directly in this directory (non-recursive)
            files = [f for f in directory.glob('*') if f.is_file()]
            
            if not files:
                continue
            
            # Sample random files
            sample_files = random.sample(files, min(self.sample_size, len(files)))
            
            for file_path in sample_files:
                metadata = self._extract_file_metadata(file_path)
                if metadata:
                    samples.append(metadata)
        
        return samples
    
    def _extract_file_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from a single file."""
        ext = get_file_extension(str(file_path))
        category = self._categorize_file(ext)
        
        metadata = {
            'path': str(file_path.relative_to(self.root_dir)),
            'filename': file_path.name,
            'extension': ext,
            'category': category,
            'size_bytes': file_path.stat().st_size
        }
        
        try:
            # Extract type-specific metadata
            if category == '2d_image':
                metadata.update(self._get_image_metadata(file_path))
            elif category == '3d_pointcloud':
                metadata.update(self._get_pointcloud_metadata(file_path))
            elif category == 'annotation':
                metadata.update(self._get_annotation_metadata(file_path))
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _get_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract image metadata (resolution, channels, etc.)."""
        if Image is None:
            return {}
        
        try:
            with Image.open(file_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format
                }
        except:
            return {}
    
    def _get_pointcloud_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract point cloud metadata (number of points, etc.)."""
        ext = get_file_extension(str(file_path))
        metadata = {}
        
        try:
            if ext == 'bin':
                # KITTI binary format (assume x,y,z,intensity)
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                metadata['num_points'] = points.shape[0]
                metadata['dimensions'] = points.shape[1]
            elif ext == 'pcd':
                # PCD format - simple parsing
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('POINTS'):
                            metadata['num_points'] = int(line.split()[1])
                            break
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def _get_annotation_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract annotation file metadata (peek at structure)."""
        ext = get_file_extension(str(file_path))
        metadata = {}
        
        try:
            if ext == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read first 100 lines or entire small file
                    content = ''.join([f.readline() for _ in range(100)])
                    data = json.loads(content) if content.strip().endswith('}') else None
                    
                    if data:
                        metadata['json_keys'] = list(data.keys()) if isinstance(data, dict) else []
                        metadata['structure_type'] = type(data).__name__
            elif ext == 'xml':
                metadata['format'] = 'xml'
            elif ext == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [f.readline() for _ in range(5)]
                    metadata['sample_lines'] = [l.strip() for l in lines if l.strip()]
        except Exception as e:
            metadata['parse_error'] = str(e)
        
        return metadata
    
    def _infer_dataset_type(self, file_counts: Dict[str, Any]) -> str:
        """Infer whether this is a 2D, 3D, or 4D dataset."""
        by_category = file_counts['by_category']
        
        has_images = by_category.get('2d_image', 0) > 0
        has_pointclouds = by_category.get('3d_pointcloud', 0) > 0
        has_videos = by_category.get('video', 0) > 0
        
        # Check directory structure for timestamps (4D indicator)
        has_temporal_structure = self._check_temporal_structure()
        
        if has_temporal_structure or has_videos:
            return '4d_sequence'
        elif has_pointclouds:
            return '3d_pointcloud'
        elif has_images:
            return '2d_image'
        else:
            return 'unknown'
    
    def _check_temporal_structure(self) -> bool:
        """Check if directory structure suggests temporal sequences."""
        # Look for timestamp-like directory names or numbered sequences
        dirs = [d.name for d in self.root_dir.rglob('*') if d.is_dir()]
        
        # Check for patterns like: timestamp_*, frame_*, seq_*
        temporal_patterns = ['timestamp', 'frame', 'seq', 'sequence']
        return any(any(pattern in d.lower() for pattern in temporal_patterns) for d in dirs)
    
    def _infer_annotation_schema(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer annotation schema from sampled annotation files."""
        schema = {
            'format': 'unknown',
            'detected_fields': [],
            'likely_standard': None
        }
        
        # Find annotation samples
        annotation_samples = [s for s in samples if s['category'] == 'annotation']
        
        if not annotation_samples:
            return schema
        
        # Analyze JSON samples
        json_samples = [s for s in annotation_samples if s['extension'] == 'json']
        if json_samples:
            keys = []
            for sample in json_samples:
                if 'json_keys' in sample:
                    keys.extend(sample['json_keys'])
            
            schema['detected_fields'] = list(set(keys))
            
            # Detect COCO format
            coco_keys = {'images', 'annotations', 'categories'}
            if coco_keys.issubset(set(keys)):
                schema['likely_standard'] = 'COCO'
            
            # Detect KITTI-like format
            if any('calib' in k.lower() for k in keys):
                schema['likely_standard'] = 'KITTI'
        
        return schema
    
    def save_report(self, output_path: str) -> None:
        """Save inspection report to JSON file."""
        if not self.inspection_result:
            raise ValueError("No inspection results available. Run inspect() first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inspection_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Inspection report saved to: {output_path}")


def inspect_dataset(root_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to inspect a dataset.
    
    Args:
        root_dir: Root directory to inspect
        output_path: Optional path to save inspection report
    
    Returns:
        Dict: Inspection results
    """
    inspector = DataInspector(root_dir)
    results = inspector.inspect()
    
    if output_path:
        inspector.save_report(output_path)
    
    return results
