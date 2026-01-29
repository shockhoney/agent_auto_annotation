"""
Coordinate transformation utilities for 3D point cloud processing.
Handles rotation matrices, translation vectors, and coordinate system conversions.
"""

import numpy as np
from typing import Tuple, Optional


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (in radians).
    
    Args:
        roll: Rotation around x-axis
        pitch: Rotation around y-axis
        yaw: Rotation around z-axis
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Roll (X-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def euler_from_rotation_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Tuple[float, float, float]: (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply rotation and translation to 3D points.
    
    Args:
        points: Nx3 array of 3D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    
    Returns:
        np.ndarray: Transformed Nx3 points
    """
    return (R @ points.T).T + t


def create_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix from R and t.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def inverse_transformation(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse transformation.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (R_inv, t_inv)
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def cart_to_hom(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to homogeneous coordinates.
    
    Args:
        points: Nx3 array of 3D points
    
    Returns:
        np.ndarray: Nx4 array in homogeneous coordinates
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])


def hom_to_cart(points: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous coordinates to Cartesian coordinates.
    
    Args:
        points: Nx4 array in homogeneous coordinates
    
    Returns:
        np.ndarray: Nx3 array of 3D points
    """
    return points[:, :3] / points[:, 3:4]
