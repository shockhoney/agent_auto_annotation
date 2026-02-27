"""
多Agent并发架构 - 支持分步确认
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
    create_output_structure, generate_dataset_info,
    download_from_cos, download_from_url)


class BaseAgent:
    """Agent基类，减少重复代码"""
    SYSTEM_PROMPT = ""

    def __init__(self, llm, tools):
        self.agent = create_react_agent(llm, tools)

    async def run(self, task: str) -> Dict:
        result = await asyncio.to_thread(
            self.agent.invoke,
            {"messages": [HumanMessage(content=f"{self.SYSTEM_PROMPT}\n\n{task}")]})
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
    """主协调器 - 支持分步和一步到位两种模式"""

    def __init__(self, llm):
        self.llm = llm
        self.prep_agent = create_react_agent(
            llm, [extract_archive, scan_directory, create_output_structure, generate_dataset_info])

    # ---------- 步骤1: 解压并扫描目录 ----------
    def step_extract_and_scan(self, archive_path: str, work_dir: str) -> Dict:
        """解压压缩包并扫描目录结构，返回目录层次和文件统计。"""
        prep_task = f"请执行: 1) extract_archive({archive_path}, {work_dir}) 2) scan_directory({work_dir})"
        asyncio.run(asyncio.to_thread(self.prep_agent.invoke, {"messages": [HumanMessage(content=prep_task)]}))

        work = Path(work_dir)
        # 构建目录树（用于展示）
        dir_tree = []
        file_stats = {}
        for p in sorted(work.rglob('*')):
            if p.is_file():
                ext = p.suffix.lower()
                file_stats[ext] = file_stats.get(ext, 0) + 1
            if p.is_dir():
                rel = p.relative_to(work)
                child_count = sum(1 for _ in p.iterdir())
                dir_tree.append({"path": str(rel), "children": child_count})

        return {
            "dir_tree": dir_tree[:50],  # 限制展示数量
            "file_stats": file_stats,
            "total_files": sum(file_stats.values()),
            "work_dir": work_dir
        }

    # ---------- 步骤2: 检测文件类型 ----------
    def step_detect(self, work_dir: str, task_mode: str = '4D',
                    manual_calib: str = '', manual_pose: str = '') -> Dict:
        """检测文件类型分布，返回检测到的各类数据路径。"""
        work = Path(work_dir)

        # 图片目录
        image_dirs = [str(d) for d in work.rglob("*") if d.is_dir() and
                     (list(d.glob("*.jpg")) or list(d.glob("*.png")))]

        # 点云检测
        pc_3d_dirs, pc_4d_dirs = [], []
        for d in work.rglob("*"):
            if not d.is_dir():
                continue
            pc_files = list(d.glob("*.pcd")) + list(d.glob("*.bin"))
            if not pc_files:
                continue
            file_count = len(pc_files)
            avg_size = sum(f.stat().st_size for f in pc_files) / file_count
            if file_count <= 30 and avg_size > 1024 * 1024 * 10:
                pc_4d_dirs.append(str(d))
            else:
                pc_3d_dirs.append(str(d))

        # 位姿检测
        pose_file, pose_file_type, egopose_dir = None, None, None
        for f in work.rglob("*"):
            if not f.is_file():
                continue
            try:
                if f.suffix.lower() == '.txt':
                    content = f.read_text(encoding='utf-8')[:500]
                    lines = [l for l in content.split('\n') if l.strip()]
                    if lines:
                        parts = lines[0].split(',')
                        if len(parts) >= 8:
                            numeric_count = sum(1 for p in parts if p.replace('.','').replace('-','').replace('e','').isdigit())
                            if numeric_count >= 6:
                                pose_file, pose_file_type = str(f), 'txt'
                elif f.suffix.lower() == '.json' and not pose_file:
                    content = f.read_text(encoding='utf-8')[:1000].lower()
                    if sum(1 for kw in ['orientation', 'position', 'quaternion', 'rotation'] if kw in content) >= 2:
                        egopose_dir = str(f.parent)
            except: pass
            if pose_file:
                break

        # 标定检测
        calib_file, calib_file_type, calib_dir = None, None, None
        for f in work.rglob("*"):
            if not f.is_file():
                continue
            try:
                content = f.read_text(encoding='utf-8')[:2000].lower()
                if sum(1 for kw in ['intrinsic', 'extrinsic', 'distortion', 'camera_matrix', 'fx', 'fy'] if kw in content) >= 2:
                    if f.suffix.lower() == '.json':
                        calib_file, calib_file_type = str(f), 'json'
                        break
                    elif f.suffix.lower() in ['.yaml', '.yml']:
                        calib_dir, calib_file_type = str(f.parent), 'yaml'
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

        # Auto模式推断
        effective_mode = task_mode
        if task_mode == 'Auto':
            if pc_4d_dirs:
                effective_mode = '4D'
            elif pc_3d_dirs:
                effective_mode = '3D'
            else:
                effective_mode = '2D'

        # 标定文件内容摘要（抽样读取用于确认）
        calib_preview = {}
        try:
            if calib_file and calib_file_type == 'json':
                with open(calib_file, 'r', encoding='utf-8') as cf:
                    raw = json.load(cf)
                # 只取前2个传感器的键名
                for i, (k, v) in enumerate(raw.items()):
                    if i >= 2:
                        break
                    if isinstance(v, dict):
                        calib_preview[k] = list(v.keys())
            elif calib_dir and calib_file_type == 'yaml':
                yaml_files = list(Path(calib_dir).glob('*.yaml')) + list(Path(calib_dir).glob('*.yml'))
                for yf in yaml_files[:2]:
                    content = yf.read_text(encoding='utf-8')[:500]
                    calib_preview[yf.name] = content[:300]
        except: pass

        detection = {
            "image_dirs": image_dirs,
            "pointcloud_3d_dirs": pc_3d_dirs,
            "pointcloud_4d_dirs": pc_4d_dirs,
            "pose_file": pose_file,
            "pose_file_type": pose_file_type,
            "egopose_dir": egopose_dir,
            "calib_file": calib_file,
            "calib_file_type": calib_file_type,
            "calib_dir": calib_dir,
            "calib_preview": calib_preview,
            "effective_mode": effective_mode,
            "task_mode": task_mode
        }
        return detection

    # ---------- 步骤3: 并发处理 ----------
    def step_process(self, detection: Dict, output_dir: str,
                     calib_yaml_mapping: dict = None, calib_json_mapping: dict = None,
                     pose_json_mapping: dict = None) -> Dict:
        """根据检测结果并发执行各Agent处理任务。"""
        return asyncio.run(self._async_process(detection, output_dir,
                                                calib_yaml_mapping, calib_json_mapping, pose_json_mapping))

    async def _async_process(self, detection: Dict, output_dir: str,
                              calib_yaml_mapping, calib_json_mapping, pose_json_mapping) -> Dict:
        results = {"agents_launched": [], "workers": [], "errors": []}
        effective_mode = detection.get("effective_mode", "4D")
        task_mode = detection.get("task_mode", effective_mode)

        # 创建输出目录
        await asyncio.to_thread(
            self.prep_agent.invoke,
            {"messages": [HumanMessage(content=f"调用 create_output_structure({output_dir}, {effective_mode})")]}
        )

        tasks = []
        image_dirs = detection.get("image_dirs", [])
        pc_3d_dirs = detection.get("pointcloud_3d_dirs", [])
        pc_4d_dirs = detection.get("pointcloud_4d_dirs", [])

        # 图片
        for d in image_dirs:
            name = Path(d).name
            agent = FormatConvertAgent(self.llm, 'image')
            tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/camera/{name}"))
            results["agents_launched"].append(f"FormatConvertAgent(image) -> {name}")

        # 3D点云
        if effective_mode in ['3D', '4D']:
            for d in pc_3d_dirs:
                agent = FormatConvertAgent(self.llm, 'pointcloud_3d')
                tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/lidar"))
                results["agents_launched"].append(f"FormatConvertAgent(3d) -> {Path(d).name}")

        # 4D点云
        if effective_mode == '4D':
            for d in pc_4d_dirs:
                agent = FormatConvertAgent(self.llm, 'pointcloud_4d')
                tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/lidar_4d"))
                results["agents_launched"].append(f"FormatConvertAgent(4d) -> {Path(d).name}")

        # 位姿
        if effective_mode in ['3D', '4D']:
            pose_file = detection.get("pose_file")
            pose_file_type = detection.get("pose_file_type")
            egopose_dir = detection.get("egopose_dir")
            if pose_file and pose_file_type == 'txt':
                agent = PoseAgent(self.llm, 'txt')
                tasks.append(agent.run(f"src_file: {pose_file}, dst_dir: {output_dir}/egopose"))
                results["agents_launched"].append(f"PoseAgent(txt) -> {Path(pose_file).name}")
            elif egopose_dir:
                agent = PoseAgent(self.llm, 'json')
                mapping_arg = json.dumps(pose_json_mapping or {})
                tasks.append(agent.run(f"src_dir: {egopose_dir}, dst_dir: {output_dir}/egopose, key_mapping: {mapping_arg}"))
                results["agents_launched"].append("PoseAgent(json) -> egopose")

        # 标定
        if effective_mode in ['3D', '4D']:
            calib_file = detection.get("calib_file")
            calib_file_type = detection.get("calib_file_type")
            calib_dir = detection.get("calib_dir")
            if calib_file and calib_file_type == 'json':
                agent = CalibrationAgent(self.llm, 'json')
                mapping_arg = json.dumps(calib_json_mapping or {})
                tasks.append(agent.run(f"src_file: {calib_file}, dst_dir: {output_dir}/calibration, key_mapping: {mapping_arg}"))
                results["agents_launched"].append(f"CalibrationAgent(json) -> {Path(calib_file).name}")
            elif calib_dir:
                agent = CalibrationAgent(self.llm, 'yaml')
                mapping_arg = json.dumps(calib_yaml_mapping or {})
                tasks.append(agent.run(f"src_dir: {calib_dir}, dst_dir: {output_dir}/calibration, key_mapping: {mapping_arg}"))
                results["agents_launched"].append("CalibrationAgent(yaml) -> calibration")

        if tasks:
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in worker_results:
                if isinstance(r, Exception):
                    results["errors"].append(str(r))
                else:
                    results["workers"].append(r)
        return results

    # ---------- 步骤4: 生成报告 ----------
    def step_finalize(self, output_dir: str, archive_path: str, task_mode: str = '4D') -> Dict:
        """生成 dataset_info.json 并返回最终统计。"""
        dataset_name = Path(archive_path).stem
        asyncio.run(asyncio.to_thread(
            self.prep_agent.invoke,
            {"messages": [HumanMessage(content=f"调用 generate_dataset_info({output_dir}, {dataset_name}, {task_mode})")]}
        ))
        # 读取生成的info
        info_path = Path(output_dir) / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"dataset_name": dataset_name}

    # ---------- 一步到位 (兼容旧接口) ----------
    async def run(self, archive_path: str, work_dir: str, output_dir: str,
                  task_mode: str = '4D', manual_calib: str = '', manual_pose: str = '',
                  calib_yaml_mapping: dict = None, calib_json_mapping: dict = None,
                  pose_json_mapping: dict = None) -> Dict:
        results = {"detection": {}, "agents_launched": [], "workers": [], "errors": [], "task_mode": task_mode}
        try:
            # 步骤1+2
            self.step_extract_and_scan(archive_path, work_dir)
            detection = self.step_detect(work_dir, task_mode, manual_calib, manual_pose)
            results["detection"] = detection
            results["effective_mode"] = detection.get("effective_mode")

            # 步骤3
            proc = self.step_process(detection, output_dir,
                                      calib_yaml_mapping, calib_json_mapping, pose_json_mapping)
            results["agents_launched"] = proc["agents_launched"]
            results["workers"] = proc["workers"]
            results["errors"] = proc["errors"]

            # 步骤4
            self.step_finalize(output_dir, archive_path, task_mode)
        except Exception as e:
            results["errors"].append(str(e))
        return results


class TaskGraph:
    """对外接口 - 兼容 app.py 调用，同时支持分步执行"""

    def __init__(self, llm, system_prompt: str = ""):
        self.orchestrator = OrchestratorAgent(llm)

    def run(self, archive_path: str = None, work_dir: str = None, output_dir: str = None,
            task_mode: str = '4D', manual_calib: str = '', manual_pose: str = '',
            calib_yaml_mapping: dict = None, calib_json_mapping: dict = None,
            pose_json_mapping: dict = None, **kwargs) -> Dict:
        """一步到位执行 (跳过确认时使用)"""
        try:
            result = asyncio.run(self.orchestrator.run(archive_path, work_dir, output_dir, task_mode,
                                                        manual_calib, manual_pose,
                                                        calib_yaml_mapping, calib_json_mapping, pose_json_mapping))
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

    # 分步接口
    def run_step_extract(self, archive_path: str, work_dir: str) -> Dict:
        return self.orchestrator.step_extract_and_scan(archive_path, work_dir)

    def run_step_detect(self, work_dir: str, task_mode: str = '4D',
                        manual_calib: str = '', manual_pose: str = '') -> Dict:
        return self.orchestrator.step_detect(work_dir, task_mode, manual_calib, manual_pose)

    def run_step_process(self, detection: Dict, output_dir: str,
                         calib_yaml_mapping: dict = None, calib_json_mapping: dict = None,
                         pose_json_mapping: dict = None) -> Dict:
        return self.orchestrator.step_process(detection, output_dir,
                                               calib_yaml_mapping, calib_json_mapping, pose_json_mapping)

    def run_step_finalize(self, output_dir: str, archive_path: str, task_mode: str = '4D') -> Dict:
        return self.orchestrator.step_finalize(output_dir, archive_path, task_mode)
