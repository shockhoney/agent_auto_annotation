"""数据预处理 Agent - Streamlit 界面"""
import streamlit as st
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agents import TaskGraph
from config import LLM_CONFIG
from langchain.chat_models import init_chat_model

st.set_page_config(page_title="数据预处理Agent", layout="wide")

# Session State
for key in ['processed', 'logs', 'summary', 'zip_data', 'errors']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['logs', 'errors'] else (False if key == 'processed' else "")

st.title("数据预处理 Agent")

# 侧边栏
with st.sidebar:
    st.header("LLM 配置")
    model = st.text_input("Model", value=LLM_CONFIG.get('model', ''))
    api_key = st.text_input("API Key", type="password", value=LLM_CONFIG.get('api_key', ''))
    base_url = st.text_input("Base URL", value=LLM_CONFIG.get('base_url', ''))
    
    st.divider()
    st.header("任务类型")
    task_mode = st.radio("数据类型", ["Auto", "2D", "3D", "4D"], horizontal=True,
                         help="Auto: 自动检测; 2D: 仅图片; 3D: 图片+3D点云+标定+位姿; 4D: 全部")
    
    st.divider()
    st.header("手动指定 (可选)")
    st.caption("自动检测失败时，可手动指定文件相对路径和字段映射")
    
    manual_calib = st.text_input("标定文件路径", value="", placeholder="如: metadata/calib_anno/params.json",
                                  help="标定文件或文件夹的相对路径 (相对于压缩包根目录)")
    manual_pose = st.text_input("位姿文件路径", value="", placeholder="如: slam_results/pose.txt",
                                 help="位姿文件或文件夹的相对路径 (相对于压缩包根目录)")
    
    with st.expander("标定字段映射 (YAML)", expanded=False):
        st.caption("格式: 你的键名=标准键名，每行一个")
        calib_yaml_mapping = st.text_area(
            "YAML标定映射", value="", height=120,
            placeholder="focal_x=fx 内参-焦距-fx\nfocal_y=fy 内参-焦距-fy\nprincipal_x=cx 内参-主点-cx\nprincipal_y=cy 内参-主点-cy\nrot_vec=r_s2b 外参-旋转向量-r_s2b\ntrans_vec=t_s2b 外参-平移向量-t_s2b",
            help="标准键名: fx, fy, cx, cy, r_s2b, t_s2b, kc2, kc3, kc4, kc5, sensor_name, is_fisheye"
        )
    
    with st.expander("标定字段映射 (JSON)", expanded=False):
        st.caption("格式: 你的键名=标准键名，每行一个")
        calib_json_mapping = st.text_area(
            "JSON标定映射", value="", height=100,
            placeholder="camera_intrinsic=intrinsic 内参-焦距-fx\ncamera_extrinsic=extrinsic 外参-旋转向量-r_s2b\ndist_coeffs=distortion 内参-畸变系数-kc2,kc3,kc4,kc5",
            help="标准键名: intrinsic, extrinsic, distortion, translation"
        )
    
    with st.expander("位姿字段映射", expanded=False):
        st.caption("TXT格式 (无需映射): 每行格式为 帧号,时间戳,x,y,z,qx,qy,qz,qw")
        st.caption("JSON格式: 格式 你的键名=标准键名，每行一个")
        pose_json_mapping = st.text_area(
            "JSON位姿映射", value="", height=100,
            placeholder="ori=orientation 方向\nquat=quaternion_local 四元数\npos=position 位置",
            help="标准键名: orientation, quaternion_local, position"
        )

# 上传 - 支持多文件
uploaded_files = st.file_uploader("上传数据压缩包", type=["zip", "tar", "gz", "7z"], accept_multiple_files=True)

def reset():
    for key in ['processed', 'logs', 'summary', 'zip_data', 'errors']:
        st.session_state[key] = [] if key in ['logs', 'errors'] else (False if key == 'processed' else "")

if uploaded_files and st.button("开始处理", on_click=reset):
    if not api_key:
        st.warning("请输入 API Key")
        st.stop()
    
    # 初始化LLM
    try:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        llm = init_chat_model(model, model_provider="openai", **kwargs)
    except Exception as e:
        st.error(f"LLM初始化失败: {e}")
        st.stop()
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        all_output_dir = tmp_path / "all_output"
        all_output_dir.mkdir()
        
        # 准备所有文件
        file_configs = []
        for idx, uploaded in enumerate(uploaded_files):
            file_work_dir = tmp_path / f"work_{idx}"
            file_work_dir.mkdir()
            file_output_dir = tmp_path / f"output_{idx}"
            file_output_dir.mkdir()
            input_path = tmp_path / uploaded.name
            input_path.write_bytes(uploaded.getbuffer())
            file_configs.append({
                "name": uploaded.name,
                "input_path": input_path.resolve().as_posix(),
                "work_dir": file_work_dir.resolve().as_posix(),
                "output_dir": file_output_dir.resolve().as_posix(),
                "local_output": file_output_dir
            })
        
        # 解析字段映射
        def parse_mapping(text):
            mapping = {}
            for line in text.strip().split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    mapping[k.strip()] = v.strip()
            return mapping
        
        yaml_map = parse_mapping(calib_yaml_mapping) if calib_yaml_mapping else {}
        json_map = parse_mapping(calib_json_mapping) if calib_json_mapping else {}
        pose_map = parse_mapping(pose_json_mapping) if pose_json_mapping else {}
        
        # 并行处理函数
        def process_file(cfg):
            agent = TaskGraph(llm)
            return agent.run(
                archive_path=cfg["input_path"],
                work_dir=cfg["work_dir"],
                output_dir=cfg["output_dir"],
                task_mode=task_mode,
                manual_calib=manual_calib,
                manual_pose=manual_pose,
                calib_yaml_mapping=yaml_map,
                calib_json_mapping=json_map,
                pose_json_mapping=pose_map
            ), cfg
        
        # 使用线程池并行执行
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with st.spinner(f"并行处理 {len(uploaded_files)} 个文件..."):
            with ThreadPoolExecutor(max_workers=min(4, len(file_configs))) as executor:
                futures = {executor.submit(process_file, cfg): cfg for cfg in file_configs}
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(e)
        # 收集结果并为每个任务独立打包
        individual_zips = []  # 存储各个独立的zip
        for res in results:
            if isinstance(res, Exception):
                st.session_state.errors.append(str(res))
                continue
            result, cfg = res
            st.session_state.logs.append(f"--- {cfg['name']} ---")
            
            if "detection" in result:
                d = result["detection"]
                mode_info = f" [Auto->{result.get('effective_mode', '')}]" if result.get('effective_mode') else ""
                st.session_state.logs.append(f"[检测]{mode_info} 图片: {len(d.get('image_dirs', []))} | 3D: {len(d.get('pointcloud_3d_dirs', []))} | 4D: {len(d.get('pointcloud_4d_dirs', []))}")
            
            if "agents_launched" in result:
                for a in result["agents_launched"]:
                    st.session_state.logs.append(f"[Agent] {a}")
            
            if "errors" in result:
                st.session_state.errors.extend(result["errors"])
            
            # 为每个任务独立打包
            dataset_name = Path(cfg['name']).stem
            if cfg['local_output'].exists() and any(cfg['local_output'].rglob('*')):
                zip_path = tmp_path / f"{dataset_name}.zip"
                shutil.make_archive(str(tmp_path / dataset_name), 'zip', str(cfg['local_output']))
                individual_zips.append({
                    "name": dataset_name,
                    "data": zip_path.read_bytes()
                })
        
        # 显示下载按钮
        if individual_zips:
            st.session_state.processed = True
            st.success(f"处理完成! 共处理 {len(individual_zips)} 个文件")
            
            # 单独下载按钮
            st.markdown("### 下载结果")
            cols = st.columns(min(3, len(individual_zips)))
            for idx, zip_info in enumerate(individual_zips):
                with cols[idx % 3]:
                    st.download_button(
                        f"{zip_info['name']}", 
                        zip_info['data'], 
                        f"{zip_info['name']}.zip", 
                        "application/zip",
                        key=f"dl_{idx}"
                    )
            
            # 一键下载全部
            if len(individual_zips) > 1:
                all_output_dir.mkdir(exist_ok=True)
                for zip_info in individual_zips:
                    (all_output_dir / f"{zip_info['name']}.zip").write_bytes(zip_info['data'])
                shutil.make_archive(str(tmp_path / "all_results"), 'zip', str(all_output_dir))
                all_zip_bytes = (tmp_path / "all_results.zip").read_bytes()
                st.download_button("下载全部", all_zip_bytes, "all_results.zip", "application/zip", key="dl_all")
        else:
            st.error("输出为空")

# 结果显示
if st.session_state.processed or st.session_state.logs:
    st.divider()
    
    if st.session_state.zip_data:
        st.download_button("下载结果", st.session_state.zip_data, "processed_data.zip", "application/zip")
    
    if st.session_state.summary:
        st.markdown("### 摘要")
        st.markdown(st.session_state.summary)
    
    with st.expander("Agent 日志", expanded=True):
        for log in st.session_state.logs:
            if "[检测]" in log:
                st.info(log)
            elif "[Agent]" in log:
                st.success(log)
            else:
                st.text(log)
    
    if st.session_state.errors:
        with st.expander("警告", expanded=True):
            for e in st.session_state.errors:
                st.warning(e)

if not uploaded_files:
    st.info("请上传数据文件")
