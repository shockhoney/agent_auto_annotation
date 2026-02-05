"""
数据预处理工具集 
仅包含实际使用的核心工具
"""
from typing import Dict, Any, List
from langchain_core.tools import tool
from pathlib import Path
import json
import shutil
import collections
import concurrent.futures


@tool
def extract_archive(archive_path: str, output_dir: str) -> str:
    """解压压缩包到指定目录。支持zip、tar、gz、7z格式。"""
    import py7zr
    import tarfile
    import zipfile
    
    try:
        archive = Path(archive_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        suffix = archive.suffix.lower()
        if suffix == '.zip':
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(out)
        elif suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive, 'r:*') as tf:
                tf.extractall(out)
        elif suffix == '.7z':
            with py7zr.SevenZipFile(archive, mode='r') as z:
                z.extractall(out)
        else:
            shutil.unpack_archive(str(archive), str(out))
        
        return json.dumps({"status": "success", "output_dir": str(out)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
def scan_directory(dir_path: str) -> str:
    """扫描目录，按文件类型统计。返回图片、点云、标定文件的分布情况。"""
    try:
        root = Path(dir_path)
        if not root.exists():
            return json.dumps({"error": f"目录不存在: {dir_path}"}, ensure_ascii=False)
        
        ext_map = {
            'images': ['.jpg', '.jpeg', '.png', '.bmp'],
            'pointclouds': ['.pcd', '.ply', '.bin'],
            'calibration': ['.yaml', '.yml'],
            'pose': ['.json']
        }
        
        distribution = {}
        calibration_files = []
        total_files = 0
        
        for p in root.rglob('*'):
            if p.is_file():
                total_files += 1
                ext = p.suffix.lower()
                parent = str(p.parent)
                
                if parent not in distribution:
                    distribution[parent] = {"path": parent, "types": {}, "count": 0}
                distribution[parent]["types"][ext] = distribution[parent]["types"].get(ext, 0) + 1
                distribution[parent]["count"] += 1
                
                if ext in ext_map['calibration']:
                    calibration_files.append(str(p))
        
        return json.dumps({
            "total_files": total_files,
            "distribution": distribution,
            "calibration_files": calibration_files
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_convert_images(src_dir: str, dst_dir: str, target_format: str = "jpg") -> str:
    """批量转换目录中的图片格式。"""
    from PIL import Image
    
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def process_image(img_path):
            try:
                rel_path = img_path.relative_to(src)
                out_path = dst / rel_path.with_suffix(f'.{target_format}')
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.open(img_path).convert('RGB').save(out_path, quality=95)
                return True
            except:
                return False
        
        images = list(src.rglob('*.jpg')) + list(src.rglob('*.jpeg')) + list(src.rglob('*.png'))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(process_image, images))
        
        return json.dumps({
            "status": "success",
            "converted": sum(results),
            "failed": len(results) - sum(results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_convert_pointclouds(src_dir: str, dst_dir: str, target_format: str = "pcd") -> str:
    """批量转换目录中的点云格式。"""
    import open3d as o3d
    import numpy as np
    
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def process_pc(pc_path):
            try:
                out_path = dst / pc_path.name
                out_path = out_path.with_suffix(f'.{target_format}')
                
                if pc_path.suffix.lower() == '.bin':
                    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                else:
                    pcd = o3d.io.read_point_cloud(str(pc_path))
                o3d.io.write_point_cloud(str(out_path), pcd)
                return True
            except:
                return False
        
        pcs = list(src.rglob('*.pcd')) + list(src.rglob('*.ply')) + list(src.rglob('*.bin'))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(process_pc, pcs))
        
        return json.dumps({
            "status": "success",
            "converted": sum(results),
            "failed": len(results) - sum(results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_process_egopose(src_dir: str, dst_dir: str) -> str:
    """批量处理ego_pose JSON文件，保留原始数据并添加旋转矩阵。"""
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def q2m(w, x, y, z):
            return [
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ]
        
        def process_json(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'orientation' not in data:
                    return False
                
                # 保留原始数据，添加旋转矩阵
                result = data.copy()
                
                # 处理 quaternion_global
                q_global = data['orientation'].get('quaternion_global', {})
                if q_global:
                    result['rotation_matrix_global'] = q2m(
                        q_global.get('w', 0), q_global.get('x', 0),
                        q_global.get('y', 0), q_global.get('z', 0)
                    )
                
                # 处理 quaternion_local
                q_local = data['orientation'].get('quaternion_local', {})
                if q_local:
                    result['rotation_matrix_local'] = q2m(
                        q_local.get('w', 0), q_local.get('x', 0),
                        q_local.get('y', 0), q_local.get('z', 0)
                    )
                
                with open(dst / json_file.name, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return True
            except:
                return False
        
        files = list(src.glob('*.json'))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(process_json, files))
        
        return json.dumps({
            "status": "success",
            "processed": sum(results),
            "failed": len(results) - sum(results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_process_all_calibrations(src_dir: str, dst_dir: str) -> str:
    """批量处理YAML标定文件，输出完整标定信息。"""
    import yaml
    import numpy as np
    
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def rvec2mat(rvec):
            v = np.array(rvec, dtype=np.float64)
            theta = np.linalg.norm(v)
            if theta < 1e-10:
                return np.eye(3).tolist()
            k = v / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.cos(theta)*np.eye(3) + (1-np.cos(theta))*np.outer(k, k) + np.sin(theta)*K
            return R.tolist()
        
        def process_yaml(yaml_file):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.startswith('%YAML'):
                        content = '\n'.join(content.split('\n')[2:])
                    data = yaml.safe_load(content)
                
                if data is None:
                    return False
                
                # 原始旋转向量和平移向量
                r_s2b = data.get('r_s2b', [0, 0, 0])
                t_s2b = data.get('t_s2b', [0, 0, 0])
                
                # 内参
                fx, fy = float(data.get('fx', 0)), float(data.get('fy', 0))
                cx, cy = float(data.get('cx', 0)), float(data.get('cy', 0))
                
                result = {
                    "sensor_name": data.get('sensor_name', yaml_file.stem),
                    "sensor_type": data.get('sensor_type', 'camera'),
                    # 原始数据
                    "rotation_vector": r_s2b,
                    "translation_vector": t_s2b,
                    # 处理后的旋转矩阵
                    "rotation_matrix": rvec2mat(r_s2b),
                    # 内参矩阵 3x3
                    "camera_intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    # 畸变系数
                    "distortion_coefficients": {
                        "kc2": float(data.get('kc2', 0)),
                        "kc3": float(data.get('kc3', 0)),
                        "kc4": float(data.get('kc4', 0)),
                        "kc5": float(data.get('kc5', 0))
                    },
                    # 图像分辨率
                    "image_size": {
                        "width": int(data.get('width', 0)),
                        "height": int(data.get('height', 0))
                    },
                    # 是否鱼眼相机
                    "is_fisheye": bool(data.get('is_fisheye', False)),
                    # 额外信息
                    "camera_model": data.get('camera_model', 'unknown')
                }
                
                with open(dst / f"{yaml_file.stem}.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return True
            except:
                return False
        
        files = list(src.glob('*.yaml')) + list(src.glob('*.yml'))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(process_yaml, files))
        
        return json.dumps({
            "status": "success",
            "processed": sum(results),
            "failed": len(results) - sum(results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def create_output_structure(base_dir: str) -> str:
    """创建标准的输出目录结构。"""
    try:
        base = Path(base_dir)
        for subdir in ['sensors', 'egopose', 'img', 'pointclouds']:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        return json.dumps({"status": "success", "created": str(base)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def generate_dataset_info(output_dir: str) -> str:
    """扫描输出目录并生成dataset_info.json，包含目录结构。"""
    try:
        out = Path(output_dir)
        
        # 生成目录结构树（每目录最多显示2个文件）
        def build_tree(path, prefix=""):
            tree = []
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            dirs = [x for x in items if x.is_dir()]
            files = [x for x in items if x.is_file()]
            
            # 目录全部显示
            for i, d in enumerate(dirs):
                is_last = i == len(dirs) - 1 and len(files) == 0
                connector = "└── " if is_last else "├── "
                tree.append(f"{prefix}{connector}{d.name}/")
                extension = "    " if is_last else "│   "
                tree.extend(build_tree(d, prefix + extension))
            
            # 文件最多显示2个
            MAX_FILES = 2
            for i, f in enumerate(files[:MAX_FILES]):
                is_last = i == min(len(files), MAX_FILES) - 1 and len(files) <= MAX_FILES
                connector = "└── " if is_last else "├── "
                tree.append(f"{prefix}{connector}{f.name}")
            
            if len(files) > MAX_FILES:
                tree.append(f"{prefix}└── ... ")
            
            return tree
        
        dir_tree = [out.name] + build_tree(out)
        
        info = {
            "dataset_name": out.name,
            "statistics": {
                "sensors": len(list((out / "sensors").glob("*.json"))) if (out / "sensors").exists() else 0,
                "egopose_frames": len(list((out / "egopose").glob("*.json"))) if (out / "egopose").exists() else 0,
                "images": len(list((out / "img").rglob("*.jpg"))) if (out / "img").exists() else 0,
                "pointclouds": len(list((out / "pointclouds").rglob("*.pcd"))) if (out / "pointclouds").exists() else 0
            },
            "directory_structure": dir_tree
        }
        with open(out / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

