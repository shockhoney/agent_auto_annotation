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
            'pose': ['.json', '.txt', '.csv']
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
    """批量转换目录及子目录中的图片格式为.jpg格式"""
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
def batch_convert_4d_pointclouds(src_dir: str, dst_dir: str, target_format: str = "pcd") -> str:
    """批量转换4D点云(含时间或强度维度)到标准格式。"""
    import open3d as o3d
    import numpy as np
    
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def process_4d_pc(pc_path):
            try:
                out_path = dst / pc_path.name
                out_path = out_path.with_suffix(f'.{target_format}')
                
                if pc_path.suffix.lower() == '.bin':
                    # 4D点云: (N, 4) 或 (N, 5) - x,y,z + intensity/time
                    data = np.fromfile(pc_path, dtype=np.float32)
                    # 根据数据大小判断维度
                    if len(data) % 5 == 0:
                        points = data.reshape(-1, 5)
                    elif len(data) % 4 == 0:
                        points = data.reshape(-1, 4)
                    else:
                        return False
                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    # 将第4维(intensity/time)归一化为颜色
                    if points.shape[1] >= 4:
                        intensity = points[:, 3]
                        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
                        colors = np.stack([intensity, intensity, intensity], axis=1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    pcd = o3d.io.read_point_cloud(str(pc_path))
                
                o3d.io.write_point_cloud(str(out_path), pcd)
                return True
            except:
                return False
        
        pcs = list(src.rglob('*.pcd')) + list(src.rglob('*.ply')) + list(src.rglob('*.bin'))
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(process_4d_pc, pcs))
        
        return json.dumps({
            "status": "success",
            "converted": sum(results),
            "failed": len(results) - sum(results)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_process_egopose(src_dir: str, dst_dir: str, key_mapping: str = "{}") -> str:
    """批量处理ego_pose JSON文件，保留原始数据并添加旋转矩阵。key_mapping为JSON字符串用于键名映射。"""
    try:
        src, dst = Path(src_dir), Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"源目录不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        # 解析映射
        mapping = {}
        try:
            mapping = json.loads(key_mapping) if isinstance(key_mapping, str) else key_mapping or {}
        except: pass
        
        def apply_mapping(data, mapping):
            """递归映射键名"""
            if isinstance(data, dict):
                return {mapping.get(k, k): apply_mapping(v, mapping) for k, v in data.items()}
            return data
        
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
                
                # 应用键名映射
                mapped = apply_mapping(data, mapping) if mapping else data
                
                if 'orientation' not in mapped:
                    return False
                
                # 保留原始数据，添加旋转矩阵
                result = mapped.copy()
                
                # 处理 quaternion_local
                q_local = mapped['orientation'].get('quaternion_local', {})
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
def batch_process_all_calibrations(src_dir: str, dst_dir: str, key_mapping: str = "{}") -> str:
    """批量处理YAML标定文件，输出完整标定信息。key_mapping为JSON字符串，如{"focal_x":"fx"}"""
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
                
                # 应用键名映射: 将用户键名替换为标准键名
                mapping = {}
                try:
                    mapping = json.loads(key_mapping) if isinstance(key_mapping, str) else key_mapping or {}
                except: pass
                mapped_data = {}
                for k, v in data.items():
                    mapped_data[mapping.get(k, k)] = v  # 有映射用映射，无映射用原名
                
                # 原始旋转向量和平移向量
                r_s2b = mapped_data.get('r_s2b', [0, 0, 0])
                t_s2b = mapped_data.get('t_s2b', [0, 0, 0])
                
                # 内参
                fx, fy = float(mapped_data.get('fx', 0)), float(mapped_data.get('fy', 0))
                cx, cy = float(mapped_data.get('cx', 0)), float(mapped_data.get('cy', 0))
                
                result = {
                    # 原始数据
                    "sensor_name": mapped_data.get('sensor_name', yaml_file.stem),
                    "rotation_vector": r_s2b,
                    "translation_vector": t_s2b,
                    # 处理后的旋转矩阵
                    "rotation_matrix": rvec2mat(r_s2b),
                    # 内参矩阵 3x3
                    "camera_intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    # 畸变系数
                    "distortion_coefficients": {
                        "kc2": float(mapped_data.get('kc2', 0)),
                        "kc3": float(mapped_data.get('kc3', 0)),
                        "kc4": float(mapped_data.get('kc4', 0)),
                        "kc5": float(mapped_data.get('kc5', 0))
                    },
                    # 是否鱼眼相机
                    "is_fisheye": bool(mapped_data.get('is_fisheye', False)),
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
def batch_process_calibration_json(src_file: str, dst_dir: str, key_mapping: str = "{}") -> str:
    """处理JSON格式的相机标定文件，提取内外参信息。key_mapping为JSON字符串用于键名映射。"""
    import numpy as np
    
    def rvec_to_matrix(rvec):
        """旋转向量转3x3旋转矩阵 (Rodrigues公式)"""
        v = np.array(rvec, dtype=np.float64)
        theta = np.linalg.norm(v)
        if theta < 1e-10:
            return np.eye(3).tolist()
        k = v / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.cos(theta)*np.eye(3) + (1-np.cos(theta))*np.outer(k, k) + np.sin(theta)*K
        return R.tolist()
    
    def parse_extrinsic(ext_data):
        """解析外参，支持多种格式：4x4齐次矩阵(16)、3x3旋转矩阵(9)、旋转向量(3)"""
        if not ext_data:
            return [[1,0,0],[0,1,0],[0,0,1]], [0,0,0]
        
        ext = ext_data if isinstance(ext_data, list) else []
        n = len(ext)
        
        if n == 16:
            # 4x4齐次矩阵展平 -> 提取3x3旋转和平移
            mat = [ext[i:i+4] for i in range(0, 16, 4)]
            rotation = [row[:3] for row in mat[:3]]
            translation = [mat[0][3], mat[1][3], mat[2][3]]
        elif n == 9:
            # 3x3旋转矩阵展平 -> 直接使用
            rotation = [ext[i:i+3] for i in range(0, 9, 3)]
            translation = [0, 0, 0]
        elif n == 3:
            # 旋转向量 -> 转换为旋转矩阵
            rotation = rvec_to_matrix(ext)
            translation = [0, 0, 0]
        else:
            rotation = [[1,0,0],[0,1,0],[0,0,1]]
            translation = [0, 0, 0]
        
        return rotation, translation
    
    try:
        src = Path(src_file)
        dst = Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"文件不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        with open(src, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 应用键名映射
            mapping = {}
            try:
                mapping = json.loads(key_mapping) if isinstance(key_mapping, str) else key_mapping or {}
            except: pass
            
            processed = 0
            for sensor_name, params in data.items():
                if not isinstance(params, dict):
                    continue
                
                # 应用映射到每个传感器的参数
                mapped_params = {}
                for k, v in params.items():
                    mapped_params[mapping.get(k, k)] = v
                
                # 解析外参 (支持多种格式)
                rotation, translation = parse_extrinsic(mapped_params.get('extrinsic', []))
            
                # 如果有单独的平移向量，使用它
                if 'translation' in mapped_params:
                    translation = mapped_params['translation']
                
                # 解析内参矩阵 (3x3 展平为9元素)
                intrinsic = mapped_params.get('intrinsic', [])
                if len(intrinsic) == 9:
                    int_matrix = [intrinsic[i:i+3] for i in range(0, 9, 3)]
                else:
                    int_matrix = [[1,0,0],[0,1,0],[0,0,1]]
                
                result = {
                    "sensor_name": sensor_name,
                    "rotation_matrix": rotation,
                    "translation_vector": translation,
                    "camera_intrinsic": int_matrix,
                    "distortion_coefficients": mapped_params.get('distortion', []),
                    "is_fisheye": 'fisheye' in sensor_name.lower()
                }
                
                with open(dst / f"{sensor_name}.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                processed += 1
        
        return json.dumps({"status": "success", "processed": processed}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def batch_process_pose_txt(src_file: str, dst_dir: str) -> str:
    """处理TXT格式的位姿文件，每行格式: 帧号,时间戳,x,y,z,qx,qy,qz,qw"""
    try:
        src = Path(src_file)
        dst = Path(dst_dir)
        if not src.exists():
            return json.dumps({"error": f"文件不存在: {src}"}, ensure_ascii=False)
        dst.mkdir(parents=True, exist_ok=True)
        
        def q2m(w, x, y, z):
            return [
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ]
        
        processed = 0
        with open(src, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                
                try:
                    frame_id = int(parts[0])
                    timestamp = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                    
                    result = {
                        "raw_source": line,  # 原始行信息溯源
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "position": {"x": x, "y": y, "z": z},
                        "orientation": {
                            "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw}
                        },
                        "rotation_matrix": q2m(qw, qx, qy, qz)
                    }
                    
                    with open(dst / f"{timestamp}.json", 'w', encoding='utf-8') as out:
                        json.dump(result, out, ensure_ascii=False, indent=2)
                    processed += 1
                except (ValueError, IndexError):
                    continue
        
        return json.dumps({"status": "success", "processed": processed}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def create_output_structure(base_dir: str, task_mode: str = "4D") -> str:
    """根据任务类型创建对应的输出目录结构。
    - 2D: 仅 img
    - 3D: camera, lidar, calibration, egopose
    - 4D: camera, lidar, lidar_4d, calibration, egopose
    """
    try:
        base = Path(base_dir)
        
        # 根据任务类型定义目录结构
        dirs_map = {
            '2D': ['camera'],
            '3D': ['camera', 'lidar', 'calibration', 'egopose'],
            '4D': ['camera', 'lidar', 'lidar_4d', 'calibration', 'egopose'],
            'Auto': ['camera', 'lidar', 'lidar_4d', 'calibration', 'egopose']  # Auto默认创建全部
        }
        
        subdirs = dirs_map.get(task_mode, dirs_map['4D'])
        for subdir in subdirs:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        
        return json.dumps({
            "status": "success", 
            "task_mode": task_mode,
            "created_dirs": subdirs,
            "base": str(base)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def generate_dataset_info(output_dir: str, dataset_name: str = "", task_mode: str = "4D") -> str:
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
            "dataset_name": dataset_name if dataset_name else out.name,
            "statistics": {
                "task_type": task_mode,
                "calib": len(list((out / "calibration").glob("*.json"))) if (out / "calibration").exists() else 0,
                "egopose": len(list((out / "egopose").glob("*.json"))) if (out / "egopose").exists() else 0,
                "camera": len(list((out / "camera").rglob("*.jpg"))) if (out / "camera").exists() else 0,
                "lidar": len(list((out / "lidar").rglob("*.pcd"))) if (out / "lidar").exists() else 0,
                "lidar_4d": len(list((out / "lidar_4d").rglob("*.pcd"))) if (out / "lidar_4d").exists() else 0
            },
            "directory_structure": dir_tree
        }
        with open(out / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

