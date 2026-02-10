"""
多Agent并发架构
"""
from typing import Dict, Any, List
from pathlib import Path
import json
import asyncio

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from .llm_tools import (
    extract_archive, scan_directory, batch_convert_images,
    batch_convert_pointclouds, batch_convert_4d_pointclouds, batch_process_egopose,
    batch_process_all_calibrations, batch_process_calibration_json, batch_process_pose_txt,
    create_output_structure, generate_dataset_info)


class BaseAgent:
    """Agent基类，减少重复代码"""
    SYSTEM_PROMPT = ""
    
    def __init__(self, llm, tools):
        self.agent = create_react_agent(llm, tools)
    
    async def run(self, task: str) -> Dict:
        result = await asyncio.to_thread(
            self.agent.invoke,
            {"messages": [HumanMessage(content=f"{self.SYSTEM_PROMPT}\n\n{task}")]}
        )
        return {"agent": self.__class__.__name__, "result": result}


class FormatConvertAgent(BaseAgent):
    """格式转换Agent - 处理图片和点云"""
    PROMPTS = {
        'image': "你是图片处理专家。调用 batch_convert_images 将图片转换为JPG格式。",
        'pointcloud_3d': "你是3D点云处理专家。调用 batch_convert_pointclouds 将点云转换为PCD格式。",
        'pointcloud_4d': "你是4D点云处理专家。调用 batch_convert_4d_pointclouds 将4D点云转换为PCD格式。"
    }
    TOOLS = {
        'image': [batch_convert_images],
        'pointcloud_3d': [batch_convert_pointclouds],
        'pointcloud_4d': [batch_convert_4d_pointclouds]
    }
    
    def __init__(self, llm, data_type: str):
        self.SYSTEM_PROMPT = self.PROMPTS[data_type]
        super().__init__(llm, self.TOOLS[data_type])


class PoseAgent(BaseAgent):
    """位姿处理Agent - 支持JSON目录和TXT单文件"""
    PROMPTS = {
        'json': "你是位姿信息处理专家。调用 batch_process_egopose 处理JSON格式位姿文件。",
        'txt': "你是位姿信息处理专家。调用 batch_process_pose_txt 处理TXT格式位姿文件。"
    }
    TOOLS = {
        'json': [batch_process_egopose],
        'txt': [batch_process_pose_txt]
    }
    
    def __init__(self, llm, data_type: str):
        self.SYSTEM_PROMPT = self.PROMPTS[data_type]
        super().__init__(llm, self.TOOLS[data_type])


class CalibrationAgent(BaseAgent):
    """标定处理Agent - 支持JSON和YAML格式"""
    PROMPTS = {
        'json': "你是标定信息处理专家。调用 batch_process_calibration_json 处理JSON格式标定文件。",
        'yaml': "你是标定信息处理专家。调用 batch_process_all_calibrations 处理YAML格式标定文件。"
    }
    TOOLS = {
        'json': [batch_process_calibration_json],
        'yaml': [batch_process_all_calibrations]
    }
    
    def __init__(self, llm, data_type: str):
        self.SYSTEM_PROMPT = self.PROMPTS[data_type]
        super().__init__(llm, self.TOOLS[data_type])


# ============ 协调器 ============
class OrchestratorAgent:
    """主协调器 - 分发任务并并发执行"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prep_agent = create_react_agent(
            llm, [extract_archive, scan_directory, create_output_structure, generate_dataset_info])
    
    async def run(self, archive_path: str, work_dir: str, output_dir: str, task_mode: str = '4D', manual_calib: str = '', manual_pose: str = '', calib_yaml_mapping: dict = None, calib_json_mapping: dict = None, pose_json_mapping: dict = None) -> Dict:
        # task_mode: 2D=图片, 3D=图片+3D点云+标定+位姿, 4D=全部(含4D点云)
        results = {"detection": {}, "agents_launched": [], "workers": [], "errors": [], "task_mode": task_mode}
        
        try:
            # 阶段1: 准备
            prep_task = f"请执行: 1) extract_archive({archive_path}, {work_dir}) 2) scan_directory({work_dir}) 3) create_output_structure({output_dir}, {task_mode})"
            await asyncio.to_thread(self.prep_agent.invoke, {"messages": [HumanMessage(content=prep_task)]})
            
            # 检测目录
            work = Path(work_dir)
            
            # 图片目录
            image_dirs = [str(d) for d in work.rglob("*") if d.is_dir() and 
                         (list(d.glob("*.jpg")) or list(d.glob("*.png")))]
            
            # 点云目录检测 - 区分3D和4D
            # 判断依据: 4D点云数量少、文件大; 3D点云数量多、文件小
            pc_3d_dirs = []
            pc_4d_dirs = []
            for d in work.rglob("*"):
                if not d.is_dir():
                    continue
                pc_files = list(d.glob("*.pcd")) + list(d.glob("*.bin"))
                if not pc_files:
                    continue
                
                file_count = len(pc_files)
                avg_size = sum(f.stat().st_size for f in pc_files) / file_count
                
                # 阈值: 4D点云单文件通常 > 1MB, 数量较少
                # 3D点云单文件通常 < 500KB, 数量较多
                if file_count <= 30 and avg_size > 1024 * 1024 * 10:  # <= 30个文件 且 平均 > 10MB
                    pc_4d_dirs.append(str(d))
                else:
                    pc_3d_dirs.append(str(d))
            
            # 位姿检测 - 基于内容特征，支持JSON和TXT格式
            pose_file = None  # 单文件 (如 pose.txt)
            pose_file_type = None  # 'txt' 或 'json'
            egopose_dir = None   # 目录 (多个JSON文件)
            
            for f in work.rglob("*"):
                if f.is_file():
                    try:
                        # TXT格式位姿: 每行多列数值 (帧号,时间戳,x,y,z,qx,qy,qz,qw)
                        if f.suffix.lower() == '.txt':
                            content = f.read_text(encoding='utf-8')[:500]
                            lines = [l for l in content.split('\n') if l.strip()]
                            if lines:
                                parts = lines[0].split(',')
                                # 至少8个字段且多数可转为数字
                                if len(parts) >= 8:
                                    numeric_count = sum(1 for p in parts if p.replace('.','').replace('-','').replace('e','').isdigit())
                                    if numeric_count >= 6:
                                        pose_file = str(f)
                                        pose_file_type = 'txt'
                        
                        # JSON格式位姿目录
                        elif f.suffix.lower() == '.json' and not pose_file:
                            content = f.read_text(encoding='utf-8')[:1000].lower()
                            if sum(1 for kw in ['orientation', 'position', 'quaternion', 'rotation'] if kw in content) >= 2:
                                egopose_dir = str(f.parent)
                    except: pass
                    
                    if pose_file:
                        break
            
            # 标定检测 - 基于内容特征，支持JSON和YAML格式
            calib_file = None  # 单文件 (如 params.json)
            calib_file_type = None  # 'json' 或 'yaml'
            calib_dir = None   # 目录 (多个YAML文件)
            
            for f in work.rglob("*"):
                if f.is_file():
                    try:
                        content = f.read_text(encoding='utf-8')[:2000].lower()
                        # 检测标定特征关键词
                        if sum(1 for kw in ['intrinsic', 'extrinsic', 'distortion', 'camera_matrix', 'fx', 'fy'] if kw in content) >= 2:
                            if f.suffix.lower() == '.json':
                                calib_file = str(f)
                                calib_file_type = 'json'
                                break
                            elif f.suffix.lower() in ['.yaml', '.yml']:
                                calib_dir = str(f.parent)
                                calib_file_type = 'yaml'
                                break
                    except: pass
            
            # 手动路径回退 - 标定
            if not calib_file and not calib_dir and manual_calib:
                manual_path = work / manual_calib.strip()
                if manual_path.exists():
                    if manual_path.is_file():
                        if manual_path.suffix.lower() == '.json':
                            calib_file, calib_file_type = str(manual_path), 'json'
                        elif manual_path.suffix.lower() in ['.yaml', '.yml']:
                            calib_dir, calib_file_type = str(manual_path.parent), 'yaml'
                    elif manual_path.is_dir():
                        yamls = list(manual_path.glob('*.yaml')) + list(manual_path.glob('*.yml'))
                        jsons = list(manual_path.glob('*.json'))
                        if yamls:
                            calib_dir, calib_file_type = str(manual_path), 'yaml'
                        elif jsons:
                            calib_file, calib_file_type = str(jsons[0]), 'json'
            
            # 手动路径回退 - 位姿
            if not pose_file and not egopose_dir and manual_pose:
                manual_path = work / manual_pose.strip()
                if manual_path.exists():
                    if manual_path.is_file() and manual_path.suffix.lower() == '.txt':
                        pose_file, pose_file_type = str(manual_path), 'txt'
                    elif manual_path.is_file() and manual_path.suffix.lower() == '.json':
                        egopose_dir = str(manual_path.parent)
                    elif manual_path.is_dir():
                        egopose_dir = str(manual_path)
            
            results["detection"] = {
                "image_dirs": image_dirs, 
                "pointcloud_3d_dirs": pc_3d_dirs,
                "pointcloud_4d_dirs": pc_4d_dirs,
                "pose_file": pose_file,
                "pose_file_type": pose_file_type,
                "egopose_dir": egopose_dir,
                "calib_file": calib_file,
                "calib_file_type": calib_file_type,
                "calib_dir": calib_dir
            }
            
            # 阶段2: 并发执行
            tasks = []
            
            # Auto模式: 根据检测结果自动判断任务类型
            effective_mode = task_mode
            if task_mode == 'Auto':
                if pc_4d_dirs:
                    effective_mode = '4D'
                elif pc_3d_dirs:
                    effective_mode = '3D'
                else:
                    effective_mode = '2D'
                results["effective_mode"] = effective_mode
            
            # 图片处理 (2D/3D/4D都包含)
            if image_dirs:
                for d in image_dirs:
                    name = Path(d).name
                    agent = FormatConvertAgent(self.llm, 'image')
                    tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/camera/{name}"))
                    results["agents_launched"].append(f"FormatConvertAgent(image) -> {name}")
            
            # 3D点云处理 (3D/4D模式)
            if effective_mode in ['3D', '4D'] and pc_3d_dirs:
                for d in pc_3d_dirs:
                    agent = FormatConvertAgent(self.llm, 'pointcloud_3d')
                    tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/lidar"))
                    results["agents_launched"].append(f"FormatConvertAgent(3d) -> {Path(d).name}")
            
            # 4D点云处理 (仅4D模式)
            if effective_mode == '4D' and pc_4d_dirs:
                for d in pc_4d_dirs:
                    agent = FormatConvertAgent(self.llm, 'pointcloud_4d')
                    tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/lidar_4d"))
                    results["agents_launched"].append(f"FormatConvertAgent(4d) -> {Path(d).name}")
            
            # 位姿处理 (3D/4D模式)
            if effective_mode in ['3D', '4D']:
                if pose_file and pose_file_type == 'txt':
                    agent = PoseAgent(self.llm, 'txt')
                    tasks.append(agent.run(f"src_file: {pose_file}, dst_dir: {output_dir}/egopose"))
                    results["agents_launched"].append(f"PoseAgent(txt) -> {Path(pose_file).name}")
                elif egopose_dir:
                    agent = PoseAgent(self.llm, 'json')
                    mapping_arg = json.dumps(pose_json_mapping or {})
                    tasks.append(agent.run(f"src_dir: {egopose_dir}, dst_dir: {output_dir}/egopose, key_mapping: {mapping_arg}"))
                    results["agents_launched"].append(f"PoseAgent(json) -> egopose")
            
            # 标定处理 (3D/4D模式)
            if effective_mode in ['3D', '4D']:
                if calib_file and calib_file_type == 'json':
                    agent = CalibrationAgent(self.llm, 'json')
                    mapping_arg = json.dumps(calib_json_mapping or {})
                    tasks.append(agent.run(f"src_file: {calib_file}, dst_dir: {output_dir}/calibration, key_mapping: {mapping_arg}"))
                    results["agents_launched"].append(f"CalibrationAgent(json) -> {Path(calib_file).name}")
                elif calib_dir:
                    agent = CalibrationAgent(self.llm, 'yaml')
                    mapping_arg = json.dumps(calib_yaml_mapping or {})
                    tasks.append(agent.run(f"src_dir: {calib_dir}, dst_dir: {output_dir}/calibration, key_mapping: {mapping_arg}"))
                    results["agents_launched"].append(f"CalibrationAgent(yaml) -> calibration")
            
            # 并发执行
            if tasks:
                worker_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in worker_results:
                    if isinstance(r, Exception):
                        results["errors"].append(str(r))
                    else:
                        results["workers"].append(r)
            
            # 阶段3: 生成报告
            dataset_name = Path(archive_path).stem
            await asyncio.to_thread(
                self.prep_agent.invoke,
                {"messages": [HumanMessage(content=f"调用 generate_dataset_info({output_dir}, {dataset_name}, {task_mode})")]}
            )
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results


class TaskGraph:
    """对外接口 - 兼容 app.py 调用"""
    
    def __init__(self, llm, system_prompt: str = ""):
        self.orchestrator = OrchestratorAgent(llm)
    
    def run(self, archive_path: str = None, work_dir: str = None, output_dir: str = None, task_mode: str = '4D', manual_calib: str = '', manual_pose: str = '', calib_yaml_mapping: dict = None, calib_json_mapping: dict = None, pose_json_mapping: dict = None, **kwargs) -> Dict:
        try:
            result = asyncio.run(self.orchestrator.run(archive_path, work_dir, output_dir, task_mode, manual_calib, manual_pose, calib_yaml_mapping, calib_json_mapping, pose_json_mapping))
            
            summary = "\n".join([f"- {a}" for a in result.get("agents_launched", [])])
            return {
                "messages": [AIMessage(content=f"处理完成。\n\n启动的Agent:\n{summary}")],
                "results": result.get("agents_launched", []),
                "detection": result.get("detection", {}),
                "agents_launched": result.get("agents_launched", []),
                "workers": result.get("workers", []),
                "errors": result.get("errors", []),
                "success": len(result.get("errors", [])) == 0
            }
        except Exception as e:
            return {"messages": [AIMessage(content=f"失败: {e}")], "errors": [str(e)], "success": False}
