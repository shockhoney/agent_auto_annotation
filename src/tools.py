import os
import shutil
import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path

# --- 2D 处理工具 ---
def standardize_2d_image(source_file, sample_name, ext="png"):
    """
    将图像移动到统一目录并重命名。
    """
    target_dir = Path("/app/data/output/2d/images")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / f"{sample_name}.{ext}"
    shutil.copy(source_file, target_path)
    return f"Standardized image saved to: {target_path}"

# --- 3D 处理工具 ---
def convert_3d_to_kitti_style(source_file, sample_name):
    """
    读取点云文件（.pcd/.ply）并导出为 KITTI 风格的二进制 .bin 文件。
    """
    target_dir = Path("/app/data/output/3d/velodyne")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用 Open3D 读取
        pcd = o3d.io.read_point_cloud(str(source_file))
        # 提取点坐标为 numpy 数组 (N, 3)
        points = np.asarray(pcd.points).astype(np.float32)
        
        # 如果是 KITTI 格式，通常需要 (N, 4)，第四列是反射强度（intensity）
        # 如果原始点云没有强度，我们补一列 0
        if not pcd.has_colors() and points.shape[1] == 3:
            intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack((points, intensity))
            
        target_path = target_dir / f"{sample_name}.bin"
        points.tofile(str(target_path)) # 写入二进制文件
        return f"Converted 3D cloud saved to: {target_path}"
    except Exception as e:
        return f"Failed to convert 3D: {str(e)}"

# --- 4D/序列处理工具 ---
def sync_4d_timestamps(lidar_times, camera_times, threshold_ms=50):
    """
    简单的 4D 帧同步逻辑（示例）：
    给定雷达和相机的时间戳列表，寻找最接近的配对。
    """
    # 实际应用中会调用 pandas 的 merge_asof
    df_lidar = pd.DataFrame({'lidar_time': lidar_times})
    df_cam = pd.DataFrame({'camera_time': camera_times})
    
    # 这里的逻辑可以根据你的具体文件名格式深度定制
    return "Timestamp sync logic ready"

# --- 核心感知工具 ---
def scan_directory(base_path="/app/data/input"):
    """
    Agent 启动后调用的第一个函数。
    返回一个字典，描述目录结构。
    """
    path_obj = Path(base_path)
    if not path_obj.exists():
        return "Error: Input path does not exist."
    
    structure = {}
    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        # 仅返回前 5 个文件作为样本，避免 Token 过长
        structure[rel_path] = {
            "file_count": len(files),
            "samples": files[:5],
            "sub_dirs": dirs
        }
    return structure
