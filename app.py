"""数据预处理 Agent - Streamlit 界面"""
import streamlit as st
import tempfile
import shutil
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from agents import TaskGraph
from config import LLM_CONFIG, COS_CONFIG
from agents.llm_tools import download_from_cos, download_from_url
from langchain.chat_models import init_chat_model

st.set_page_config(page_title="数据预处理Agent", layout="wide")

# Session State 初始化
INIT_STATE = {
    'processed': False, 'logs': [], 'errors': [], 'summary': '', 'zip_data': '',
    'current_step': 0,  # 0=未开始, 1=已解压, 2=已检测, 3=已处理, 4=已完成
    'scan_result': None, 'detection_result': None, 'process_result': None,
    'tmp_dir': None, 'file_configs': [], 'llm': None, 'task_graph': None,
}
for key, default in INIT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("数据预处理 Agent")

# ============ 侧边栏 ============
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
    skip_confirm = st.checkbox("跳过确认，直接处理", value=False,
                               help="勾选后将跳过中间确认步骤，一步到位完成处理")

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
            placeholder="focal_x=fx\nfocal_y=fy\nprincipal_x=cx\nprincipal_y=cy",
            help="标准键名: fx, fy, cx, cy, r_s2b, t_s2b, kc2, kc3, kc4, kc5, sensor_name, is_fisheye")

    with st.expander("标定字段映射 (JSON)", expanded=False):
        st.caption("格式: 你的键名=标准键名，每行一个")
        calib_json_mapping = st.text_area(
            "JSON标定映射", value="", height=100,
            placeholder="camera_intrinsic=intrinsic\ncamera_extrinsic=extrinsic",
            help="标准键名: intrinsic, extrinsic, distortion, translation")

    with st.expander("位姿字段映射", expanded=False):
        st.caption("TXT格式 (无需映射): 每行格式为 帧号,时间戳,x,y,z,qx,qy,qz,qw")
        st.caption("JSON格式: 格式 你的键名=标准键名，每行一个")
        pose_json_mapping = st.text_area(
            "JSON位姿映射", value="", height=100,
            placeholder="ori=orientation\nquat=quaternion_local\npos=position",
            help="标准键名: orientation, quaternion_local, position")

    # COS 配置
    with st.expander("COS 配置", expanded=False):
        cos_secret_id = st.text_input("Secret ID", value=COS_CONFIG.get('secret_id', ''))
        cos_secret_key = st.text_input("Secret Key", type="password", value=COS_CONFIG.get('secret_key', ''))
        cos_region = st.text_input("Region", value=COS_CONFIG.get('region', 'ap-beijing'))
        cos_bucket = st.text_input("Bucket", value=COS_CONFIG.get('bucket', ''))


# ============ 工具函数 ============
def parse_mapping(text):
    mapping = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            mapping[k.strip()] = v.strip().split()[0]  # 取等号后第一个词作为标准键名
    return mapping


def reset_all():
    for key, default in INIT_STATE.items():
        st.session_state[key] = default


def init_llm():
    """初始化LLM，缓存到session_state"""
    if st.session_state.llm is not None:
        return st.session_state.llm
    if not api_key:
        st.warning("请输入 API Key")
        st.stop()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    llm = init_chat_model(model, model_provider="openai", **kwargs)
    st.session_state.llm = llm
    return llm


def get_task_graph():
    if st.session_state.task_graph is not None:
        return st.session_state.task_graph
    llm = init_llm()
    tg = TaskGraph(llm)
    st.session_state.task_graph = tg
    return tg


# ============ 数据源选择 ============
st.subheader("数据来源")
source_tab_local, source_tab_cos, source_tab_url = st.tabs(["本地上传", "COS对象存储", "链接下载"])

input_ready = False  # 标识是否有有效输入
source_type = None

with source_tab_local:
    uploaded_files = st.file_uploader("上传数据压缩包", type=["zip", "tar", "gz", "7z"], accept_multiple_files=True)
    if uploaded_files:
        input_ready = True
        source_type = "local"

with source_tab_cos:
    cos_path = st.text_input("COS 对象路径", placeholder="data/dataset_v1.zip 或 data/my_dataset/")
    if cos_path:
        input_ready = True
        source_type = "cos"

with source_tab_url:
    download_url = st.text_input("下载链接", placeholder="https://example.com/dataset.zip")
    if download_url:
        input_ready = True
        source_type = "url"


# ============ 处理逻辑 ============

# 解析映射
yaml_map = parse_mapping(calib_yaml_mapping) if calib_yaml_mapping else {}
json_map = parse_mapping(calib_json_mapping) if calib_json_mapping else {}
pose_map = parse_mapping(pose_json_mapping) if pose_json_mapping else {}


def prepare_files(tmp_path):
    """根据数据源类型准备文件到本地临时目录，返回 file_configs 列表"""
    configs = []

    if source_type == "local":
        for idx, uploaded in enumerate(uploaded_files):
            file_work_dir = tmp_path / f"work_{idx}"
            file_work_dir.mkdir()
            file_output_dir = tmp_path / f"output_{idx}"
            file_output_dir.mkdir()
            input_path = tmp_path / uploaded.name
            input_path.write_bytes(uploaded.getbuffer())
            configs.append({
                "name": uploaded.name,
                "input_path": input_path.resolve().as_posix(),
                "work_dir": file_work_dir.resolve().as_posix(),
                "output_dir": file_output_dir.resolve().as_posix(),
                "local_output": file_output_dir
            })

    elif source_type == "cos":
        result_str = download_from_cos.invoke({
            "cos_path": cos_path, "output_dir": str(tmp_path / "cos_download"),
            "secret_id": cos_secret_id, "secret_key": cos_secret_key,
            "region": cos_region, "bucket": cos_bucket
        })
        result = json.loads(result_str)
        if "error" in result:
            st.error(f"COS下载失败: {result['error']}")
            return []
        # 可能是单文件或多文件
        files = [result["file"]] if "file" in result else result.get("files", [])
        for idx, fpath in enumerate(files):
            fp = Path(fpath)
            file_work_dir = tmp_path / f"work_{idx}"
            file_work_dir.mkdir()
            file_output_dir = tmp_path / f"output_{idx}"
            file_output_dir.mkdir()
            configs.append({
                "name": fp.name,
                "input_path": fp.resolve().as_posix(),
                "work_dir": file_work_dir.resolve().as_posix(),
                "output_dir": file_output_dir.resolve().as_posix(),
                "local_output": file_output_dir
            })

    elif source_type == "url":
        result_str = download_from_url.invoke({"url": download_url, "output_dir": str(tmp_path / "url_download")})
        result = json.loads(result_str)
        if "error" in result:
            st.error(f"链接下载失败: {result['error']}")
            return []
        fp = Path(result["file"])
        file_work_dir = tmp_path / "work_0"
        file_work_dir.mkdir()
        file_output_dir = tmp_path / "output_0"
        file_output_dir.mkdir()
        configs.append({
            "name": fp.name,
            "input_path": fp.resolve().as_posix(),
            "work_dir": file_work_dir.resolve().as_posix(),
            "output_dir": file_output_dir.resolve().as_posix(),
            "local_output": file_output_dir
        })
    return configs


def collect_and_package(results_list, file_configs, tmp_path):
    """收集处理结果并打包，返回 individual_zips 列表"""
    individual_zips = []
    for res in results_list:
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

        dataset_name = Path(cfg['name']).stem
        if cfg['local_output'].exists() and any(cfg['local_output'].rglob('*')):
            zip_path = tmp_path / f"{dataset_name}.zip"
            shutil.make_archive(str(tmp_path / dataset_name), 'zip', str(cfg['local_output']))
            individual_zips.append({"name": dataset_name, "data": zip_path.read_bytes()})
    return individual_zips


def show_download_buttons(individual_zips, tmp_path):
    """显示下载按钮"""
    if individual_zips:
        st.session_state.processed = True
        st.success(f"处理完成! 共处理 {len(individual_zips)} 个文件")
        st.markdown("### 下载结果")
        cols = st.columns(min(3, len(individual_zips)))
        for idx, zip_info in enumerate(individual_zips):
            with cols[idx % 3]:
                st.download_button(f"{zip_info['name']}", zip_info['data'],
                                   f"{zip_info['name']}.zip", "application/zip", key=f"dl_{idx}")
        if len(individual_zips) > 1:
            all_dir = tmp_path / "all_output"
            all_dir.mkdir(exist_ok=True)
            for zi in individual_zips:
                (all_dir / f"{zi['name']}.zip").write_bytes(zi['data'])
            shutil.make_archive(str(tmp_path / "all_results"), 'zip', str(all_dir))
            st.download_button("下载全部", (tmp_path / "all_results.zip").read_bytes(),
                               "all_results.zip", "application/zip", key="dl_all")
    else:
        st.error("输出为空")


# ============ 跳过确认: 直接处理 ============
if input_ready and skip_confirm and st.button("开始处理", on_click=reset_all):
    llm = init_llm()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with st.spinner("准备文件..."):
            file_configs = prepare_files(tmp_path)
        if not file_configs:
            st.stop()

        def process_file(cfg):
            agent = TaskGraph(llm)
            return agent.run(
                archive_path=cfg["input_path"], work_dir=cfg["work_dir"], output_dir=cfg["output_dir"],
                task_mode=task_mode, manual_calib=manual_calib, manual_pose=manual_pose,
                calib_yaml_mapping=yaml_map, calib_json_mapping=json_map, pose_json_mapping=pose_map
            ), cfg

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with st.spinner(f"并行处理 {len(file_configs)} 个文件..."):
            with ThreadPoolExecutor(max_workers=min(4, len(file_configs))) as executor:
                futures = {executor.submit(process_file, cfg): cfg for cfg in file_configs}
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(e)
        individual_zips = collect_and_package(results, file_configs, tmp_path)
        show_download_buttons(individual_zips, tmp_path)


# ============ 分步确认流程 ============
if input_ready and not skip_confirm:

    # --- 步骤0: 开始 ---
    if st.session_state.current_step == 0:
        if st.button("开始处理 (分步确认)", on_click=reset_all, key="btn_start"):
            st.session_state.current_step = 0  # 重置后会被下面逻辑推进
            llm = init_llm()
            tmp = tempfile.mkdtemp()
            st.session_state.tmp_dir = tmp
            tmp_path = Path(tmp)

            with st.spinner("准备文件..."):
                file_configs = prepare_files(tmp_path)
            if not file_configs:
                st.stop()
            st.session_state.file_configs = file_configs

            # 执行步骤1: 解压并扫描
            tg = get_task_graph()
            scan_results = []
            with st.spinner("解压并扫描目录..."):
                for cfg in file_configs:
                    scan = tg.run_step_extract(cfg["input_path"], cfg["work_dir"])
                    scan_results.append({"name": cfg["name"], "scan": scan})
            st.session_state.scan_result = scan_results
            st.session_state.current_step = 1
            st.rerun()

    # --- 步骤1: 确认目录结构 ---
    if st.session_state.current_step == 1 and st.session_state.scan_result:
        st.subheader("步骤 1/4: 目录结构确认")
        for item in st.session_state.scan_result:
            with st.expander(f"{item['name']}", expanded=True):
                scan = item["scan"]
                st.write(f"**文件总数**: {scan.get('total_files', 0)}")

                # 文件类型统计
                stats = scan.get("file_stats", {})
                if stats:
                    st.write("**文件类型分布**:")
                    cols = st.columns(min(4, len(stats)))
                    for i, (ext, cnt) in enumerate(sorted(stats.items(), key=lambda x: -x[1])):
                        with cols[i % 4]:
                            st.metric(ext, cnt)

                # 目录结构
                dirs = scan.get("dir_tree", [])
                if dirs:
                    st.write("**目录结构** (前20个):")
                    for d in dirs[:20]:
                        st.text(f"  {d['path']}/ ({d['children']} items)")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认并继续", key="confirm_step1"):
                # 执行步骤2: 检测文件类型
                tg = get_task_graph()
                detection_results = []
                with st.spinner("检测文件类型..."):
                    for cfg in st.session_state.file_configs:
                        det = tg.run_step_detect(cfg["work_dir"], task_mode, manual_calib, manual_pose)
                        detection_results.append({"name": cfg["name"], "detection": det, "cfg": cfg})
                st.session_state.detection_result = detection_results
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("重新开始", key="reset_step1"):
                reset_all()
                st.rerun()

    # --- 步骤2: 确认文件类型检测结果 ---
    if st.session_state.current_step == 2 and st.session_state.detection_result:
        st.subheader("步骤 2/4: 文件类型检测确认")
        for item in st.session_state.detection_result:
            with st.expander(f"{item['name']}", expanded=True):
                det = item["detection"]
                mode = det.get("effective_mode", task_mode)
                st.info(f"检测模式: **{det.get('task_mode')}** -> 实际模式: **{mode}**")

                c1, c2 = st.columns(2)
                with c1:
                    img_dirs = det.get("image_dirs", [])
                    st.write(f"**图片目录** ({len(img_dirs)}个):")
                    for d in img_dirs:
                        st.text(f"  {Path(d).name}/")

                    pc3 = det.get("pointcloud_3d_dirs", [])
                    if pc3:
                        st.write(f"**3D点云目录** ({len(pc3)}个):")
                        for d in pc3:
                            st.text(f"  {Path(d).name}/")

                    pc4 = det.get("pointcloud_4d_dirs", [])
                    if pc4:
                        st.write(f"**4D点云目录** ({len(pc4)}个):")
                        for d in pc4:
                            st.text(f"  {Path(d).name}/")

                with c2:
                    pf = det.get("pose_file")
                    pd = det.get("egopose_dir")
                    if pf:
                        st.write(f"**位姿文件**: {Path(pf).name} ({det.get('pose_file_type')})")
                    elif pd:
                        st.write(f"**位姿目录**: {Path(pd).name}/ (json)")
                    else:
                        st.warning("未检测到位姿数据")

                    cf = det.get("calib_file")
                    cd = det.get("calib_dir")
                    if cf:
                        st.write(f"**标定文件**: {Path(cf).name} ({det.get('calib_file_type')})")
                    elif cd:
                        st.write(f"**标定目录**: {Path(cd).name}/ (yaml)")
                    else:
                        st.warning("未检测到标定数据")

                    # 标定预览
                    preview = det.get("calib_preview", {})
                    if preview:
                        st.write("**标定内容摘要**:")
                        for name, content in preview.items():
                            if isinstance(content, list):
                                st.text(f"  {name}: 字段={content}")
                            else:
                                st.code(str(content)[:200], language="yaml")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认并执行处理", key="confirm_step2"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("重新开始", key="reset_step2"):
                reset_all()
                st.rerun()

    # --- 步骤3: 执行处理 ---
    if st.session_state.current_step == 3 and st.session_state.detection_result:
        st.subheader("步骤 3/4: 执行处理")
        tg = get_task_graph()
        tmp_path = Path(st.session_state.tmp_dir)

        process_results = []
        with st.spinner("并发处理中..."):
            for item in st.session_state.detection_result:
                cfg = item["cfg"]
                det = item["detection"]
                try:
                    proc = tg.run_step_process(det, cfg["output_dir"],
                                                yaml_map, json_map, pose_map)
                    tg.run_step_finalize(cfg["output_dir"], cfg["input_path"], task_mode)
                    result = {
                        "detection": det,
                        "agents_launched": proc.get("agents_launched", []),
                        "errors": proc.get("errors", []),
                        "effective_mode": det.get("effective_mode")
                    }
                    process_results.append((result, cfg))
                except Exception as e:
                    process_results.append(e)

        individual_zips = collect_and_package(process_results, st.session_state.file_configs, tmp_path)
        st.session_state.process_result = individual_zips
        show_download_buttons(individual_zips, tmp_path)
        st.session_state.current_step = 4

    # --- 步骤4: 完成 ---
    if st.session_state.current_step == 4:
        # 下载按钮已在步骤3中显示
        if st.button("处理新文件", key="btn_new"):
            # 清理临时目录
            if st.session_state.tmp_dir:
                shutil.rmtree(st.session_state.tmp_dir, ignore_errors=True)
            reset_all()
            st.rerun()


# ============ 结果显示 ============
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

if not input_ready:
    st.info("请选择数据来源并提供数据")
