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
    batch_convert_pointclouds, batch_process_egopose,
    batch_process_all_calibrations, create_output_structure, generate_dataset_info)


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


class ImageAgent(BaseAgent):
    SYSTEM_PROMPT = "你是图片处理专家。调用 batch_convert_images 将图片转换为JPG格式。"
    def __init__(self, llm):
        super().__init__(llm, [batch_convert_images])


class PointCloudAgent(BaseAgent):
    SYSTEM_PROMPT = "你是3D点云处理专家。调用 batch_convert_pointclouds 将点云转换为PCD格式。"
    def __init__(self, llm):
        super().__init__(llm, [batch_convert_pointclouds])


class EgoPoseAgent(BaseAgent):
    SYSTEM_PROMPT = "你是车身位姿信息处理专家。调用 batch_process_egopose 将四元数转换为旋转矩阵添加到车身位姿态文件中。"
    def __init__(self, llm):
        super().__init__(llm, [batch_process_egopose])


class CalibrationAgent(BaseAgent):
    SYSTEM_PROMPT = "你是传感器标定信息处理专家。调用 batch_process_all_calibrations 处理YAML标定文件。"
    def __init__(self, llm):
        super().__init__(llm, [batch_process_all_calibrations])


# ============ 协调器 ============
class OrchestratorAgent:
    """主协调器 - 分发任务并并发执行"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prep_agent = create_react_agent(
            llm, [extract_archive, scan_directory, create_output_structure, generate_dataset_info])
    
    async def run(self, archive_path: str, work_dir: str, output_dir: str) -> Dict:
        results = {"detection": {}, "agents_launched": [], "workers": [], "errors": []}
        
        try:
            # 阶段1: 准备
            prep_task = f"请执行: 1) extract_archive({archive_path}, {work_dir}) 2) scan_directory({work_dir}) 3) create_output_structure({output_dir})"
            await asyncio.to_thread(self.prep_agent.invoke, {"messages": [HumanMessage(content=prep_task)]})
            
            # 检测目录
            work = Path(work_dir)
            
            # 图片目录
            image_dirs = [str(d) for d in work.rglob("*") if d.is_dir() and 
                         (list(d.glob("*.jpg")) or list(d.glob("*.png")))]
            
            # 点云目录
            pc_dirs = [str(d) for d in work.rglob("*") if d.is_dir() and 
                      (list(d.glob("*.pcd")) or list(d.glob("*.bin")))]
            
            # 位姿目录 (10+个JSON且内容有位姿特征)
            egopose_dir = None
            for d in work.rglob("*"):
                if d.is_dir():
                    jsons = list(d.glob("*.json"))
                    if len(jsons) > 10:
                        try:
                            content = jsons[0].read_text(encoding='utf-8')[:1000].lower()
                            if sum(1 for kw in ['orientation', 'position', 'rotation', 'x', 'y', 'z', 'w'] if kw in content) >= 3:
                                egopose_dir = str(d)
                                break
                        except: pass
            
            # 标定目录
            calib_dir = None
            for d in work.rglob("*"):
                if d.is_dir() and (list(d.glob("*.yaml")) or list(d.glob("*.yml"))):
                    calib_dir = str(d)
                    break
            
            results["detection"] = {
                "image_dirs": image_dirs, "pointcloud_dirs": pc_dirs,
                "egopose_dir": egopose_dir, "calib_dir": calib_dir
            }
            
            # 阶段2: 并发执行
            tasks = []
            
            if image_dirs:
                for d in image_dirs:
                    name = Path(d).name
                    agent = ImageAgent(self.llm)
                    tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/img/{name}"))
                    results["agents_launched"].append(f"ImageAgent -> {name}")
            
            if pc_dirs:
                for d in pc_dirs:
                    name = Path(d).name
                    agent = PointCloudAgent(self.llm)
                    tasks.append(agent.run(f"src_dir: {d}, dst_dir: {output_dir}/pointclouds/{name}"))
                    results["agents_launched"].append(f"PointCloudAgent -> {name}")
            
            if egopose_dir:
                agent = EgoPoseAgent(self.llm)
                tasks.append(agent.run(f"src_dir: {egopose_dir}, dst_dir: {output_dir}/egopose"))
                results["agents_launched"].append(f"EgoPoseAgent -> egopose")
            else:
                results["errors"].append("未检测到位姿目录")
            
            if calib_dir:
                agent = CalibrationAgent(self.llm)
                tasks.append(agent.run(f"src_dir: {calib_dir}, dst_dir: {output_dir}/sensors"))
                results["agents_launched"].append(f"CalibrationAgent -> sensors")
            else:
                results["errors"].append("未检测到标定目录")
            
            # 并发执行
            if tasks:
                worker_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in worker_results:
                    if isinstance(r, Exception):
                        results["errors"].append(str(r))
                    else:
                        results["workers"].append(r)
            
            # 阶段3: 生成报告
            await asyncio.to_thread(
                self.prep_agent.invoke,
                {"messages": [HumanMessage(content=f"调用 generate_dataset_info({output_dir})")]}
            )
            
        except Exception as e:
            results["errors"].append(str(e))
        
        return results


class TaskGraph:
    """对外接口 - 兼容 app.py 调用"""
    
    def __init__(self, llm, system_prompt: str = ""):
        self.orchestrator = OrchestratorAgent(llm)
    
    def run(self, archive_path: str = None, work_dir: str = None, output_dir: str = None, **kwargs) -> Dict:
        try:
            result = asyncio.run(self.orchestrator.run(archive_path, work_dir, output_dir))
            
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
