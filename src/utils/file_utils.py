"""
File utility functions for the agent auto-annotation system.
Handles file I/O, directory operations, and path management.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
    
    Returns:
        Path: Path object of the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(file_path: str) -> str:
    """
    Get file extension (lowercase, without dot).
    
    Args:
        file_path: Path to file
    
    Returns:
        str: File extension (e.g., 'jpg', 'pcd')
    """
    return Path(file_path).suffix.lower().lstrip('.')


def list_files(directory: str, extensions: Optional[List[str]] = None, recursive: bool = False) -> List[Path]:
    """
    List files in a directory, optionally filtered by extensions.
    
    Args:
        directory: Directory to scan
        extensions: List of extensions to filter (e.g., ['jpg', 'png'])
        recursive: Whether to scan recursively
    
    Returns:
        List[Path]: List of file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    if recursive:
        files = [f for f in dir_path.rglob('*') if f.is_file()]
    else:
        files = [f for f in dir_path.glob('*') if f.is_file()]
    
    if extensions:
        extensions = [ext.lower().lstrip('.') for ext in extensions]
        files = [f for f in files if get_file_extension(str(f)) in extensions]
    
    return sorted(files)


def copy_file(src: str, dst: str, create_dirs: bool = True) -> None:
    """
    Copy a file from src to dst.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories
    """
    if create_dirs:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dict: Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation (default: 2)
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Dict: Loaded YAML data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get relative path from base_path to file_path.
    
    Args:
        file_path: Full file path
        base_path: Base directory path
    
    Returns:
        str: Relative path
    """
    return str(Path(file_path).relative_to(base_path))
