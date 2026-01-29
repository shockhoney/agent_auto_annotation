"""
COCO format converter for 2D image annotations.
Converts various 2D annotation formats to COCO JSON standard.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

from utils.file_utils import ensure_dir, list_files, copy_file, save_json
from utils.logger import logger


class COCOConverter:
    """Converts 2D image annotations to COCO format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize COCO converter.
        
        Args:
            output_dir: Output directory for converted data
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        ensure_dir(self.images_dir)
        
        self.coco_data = {
            "info": {
                "description": "Auto-converted dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.image_id = 1
        self.annotation_id = 1
        self.category_map = {}
    
    def add_category(self, name: str, supercategory: str = "object") -> int:
        """
        Add a category to COCO data.
        
        Args:
            name: Category name
            supercategory: Super category name
        
        Returns:
            int: Category ID
        """
        if name not in self.category_map:
            category_id = len(self.category_map) + 1
            self.category_map[name] = category_id
            self.coco_data["categories"].append({
                "id": category_id,
                "name": name,
                "supercategory": supercategory
            })
            return category_id
        return self.category_map[name]
    
    def add_image(self, image_path: str, file_name: Optional[str] = None) -> int:
        """
        Add an image to COCO dataset.
        
        Args:
            image_path: Path to source image
            file_name: Optional custom filename
        
        Returns:
            int: Image ID
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image processing")
        
        src_path = Path(image_path)
        if not src_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return -1
        
        # Get image dimensions
        with Image.open(src_path) as img:
            width, height = img.size
        
        # Copy image to output directory
        if file_name is None:
            file_name = f"{self.image_id:06d}{src_path.suffix}"
        
        dst_path = self.images_dir / file_name
        copy_file(str(src_path), str(dst_path))
        
        # Add to COCO data
        image_data = {
            "id": self.image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        }
        self.coco_data["images"].append(image_data)
        
        current_id = self.image_id
        self.image_id += 1
        
        return current_id
    
    def add_annotation(self, image_id: int, category_name: str, bbox: List[float], 
                      segmentation: Optional[List] = None, area: Optional[float] = None) -> None:
        """
        Add an annotation to COCO dataset.
        
        Args:
            image_id: Image ID this annotation belongs to
            category_name: Category name
            bbox: Bounding box [x, y, width, height]
            segmentation: Optional segmentation polygon
            area: Optional area (calculated from bbox if not provided)
        """
        category_id = self.add_category(category_name)
        
        # Calculate area if not provided
        if area is None:
            area = bbox[2] * bbox[3]
        
        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        
        if segmentation:
            annotation["segmentation"] = segmentation
        
        self.coco_data["annotations"].append(annotation)
        self.annotation_id += 1
    
    def convert_from_directory(self, image_dir: str, annotation_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert images from a directory structure.
        
        Args:
            image_dir: Directory containing images
            annotation_dir: Optional directory containing annotations
        
        Returns:
            Dict: Conversion statistics
        """
        image_files = list_files(image_dir, extensions=['jpg', 'jpeg', 'png', 'bmp'], recursive=True)
        
        logger.info(f"Found {len(image_files)} images to convert")
        
        for img_path in image_files:
            image_id = self.add_image(str(img_path))
            
            # If annotation directory provided, look for corresponding annotation
            if annotation_dir and image_id > 0:
                self._process_annotation_file(img_path, annotation_dir, image_id)
        
        return {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "total_categories": len(self.coco_data["categories"])
        }
    
    def _process_annotation_file(self, image_path: Path, annotation_dir: str, image_id: int) -> None:
        """Process annotation file corresponding to an image."""
        # Look for JSON/XML annotation with same base name
        ann_base = image_path.stem
        ann_dir = Path(annotation_dir)
        
        # Try JSON first
        json_ann = ann_dir / f"{ann_base}.json"
        if json_ann.exists():
            self._parse_json_annotation(json_ann, image_id)
            return
        
        # Try XML (VOC format)
        xml_ann = ann_dir / f"{ann_base}.xml"
        if xml_ann.exists():
            self._parse_xml_annotation(xml_ann, image_id)
            return
    
    def _parse_json_annotation(self, ann_path: Path, image_id: int) -> None:
        """Parse JSON annotation file."""
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # If it has objects/shapes list
                objects = data.get('objects') or data.get('shapes') or []
                for obj in objects:
                    category = obj.get('label') or obj.get('class_name') or 'object'
                    
                    # Extract bbox if available
                    bbox = obj.get('bbox') or obj.get('bounding_box')
                    if bbox:
                        # Convert to COCO format [x, y, w, h]
                        if len(bbox) == 4:
                            self.add_annotation(image_id, category, bbox)
        except Exception as e:
            logger.warning(f"Failed to parse annotation {ann_path}: {e}")
    
    def _parse_xml_annotation(self, ann_path: Path, image_id: int) -> None:
        """Parse XML annotation (Pascal VOC format)."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                category = obj.find('name').text
                bndbox = obj.find('bndbox')
                
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Convert to COCO format [x, y, w, h]
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                self.add_annotation(image_id, category, bbox)
        except Exception as e:
            logger.warning(f"Failed to parse XML annotation {ann_path}: {e}")
    
    def save(self, filename: str = "annotations.json") -> str:
        """
        Save COCO data to JSON file.
        
        Args:
            filename: Output filename
        
        Returns:
            str: Path to saved file
        """
        output_path = self.output_dir / filename
        save_json(self.coco_data, str(output_path))
        logger.info(f"COCO annotations saved to: {output_path}")
        
        # Also save metadata
        metadata = {
            "format": "COCO",
            "num_images": len(self.coco_data["images"]),
            "num_annotations": len(self.coco_data["annotations"]),
            "num_categories": len(self.coco_data["categories"]),
            "categories": [cat["name"] for cat in self.coco_data["categories"]]
        }
        save_json(metadata, str(self.output_dir / "metadata.json"))
        
        return str(output_path)
