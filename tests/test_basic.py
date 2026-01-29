"""
Basic unit tests for the agent auto-annotation system.
Run with: python -m pytest tests/
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inspect_data import DataInspector
from converters import COCOConverter, KITTIConverter, WaymoConverter
from config_generator import ConfigGenerator
from utils.file_utils import *


class TestFileUtils:
    """Test file utility functions."""
    
    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested" / "dir"
            result = ensure_dir(str(test_dir))
            assert result.exists()
            assert result.is_dir()
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("test.jpg") == "jpg"
        assert get_file_extension("file.PCD") == "pcd"
        assert get_file_extension("/path/to/file.BIN") == "bin"
    
    def test_save_load_json(self):
        """Test JSON save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}
            
            save_json(data, str(test_file))
            loaded = load_json(str(test_file))
            
            assert loaded == data


class TestCOCOConverter:
    """Test COCO converter."""
    
    def test_converter_initialization(self):
        """Test COCO converter initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = COCOConverter(tmpdir)
            assert converter.output_dir.exists()
            assert converter.images_dir.exists()
    
    def test_add_category(self):
        """Test adding categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = COCOConverter(tmpdir)
            
            cat_id1 = converter.add_category("car")
            cat_id2 = converter.add_category("pedestrian")
            cat_id3 = converter.add_category("car")  # Duplicate
            
            assert cat_id1 != cat_id2
            assert cat_id1 == cat_id3  # Should return same ID
            assert len(converter.coco_data['categories']) == 2


class TestKITTIConverter:
    """Test KITTI converter."""
    
    def test_converter_initialization(self):
        """Test KITTI converter initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = KITTIConverter(tmpdir)
            assert converter.velodyne_dir.exists()
            assert converter.calib_dir.exists()
            assert converter.label_dir.exists()


class TestConfigGenerator:
    """Test configuration generator."""
    
    def test_config_generation(self):
        """Test basic config generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal structure
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()
            
            generator = ConfigGenerator(tmpdir, '2d')
            config = generator.generate_config()
            
            assert config['task_type'] == '2D_Detection'
            assert config['data_root'] == tmpdir


class TestDataInspector:
    """Test data inspector."""
    
    def test_inspector_initialization(self):
        """Test data inspector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inspector = DataInspector(tmpdir)
            assert inspector.root_dir.exists()
    
    def test_file_categorization(self):
        """Test file type categorization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inspector = DataInspector(tmpdir)
            
            assert inspector._categorize_file('jpg') == '2d_image'
            assert inspector._categorize_file('pcd') == '3d_pointcloud'
            assert inspector._categorize_file('json') == 'annotation'
            assert inspector._categorize_file('xyz') == 'unknown'


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
