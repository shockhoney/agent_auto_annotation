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
    provider = st.radio("模型", ["通义千问", "DeepSeek", "其他"])
    model = st.text_input("Model", value=LLM_CONFIG.get('model', ''))
    api_key = st.text_input("API Key", type="password", value=LLM_CONFIG.get('api_key', ''))
    base_url = st.text_input("Base URL", value=LLM_CONFIG.get('base_url', ''))

# 上传
uploaded = st.file_uploader("上传数据压缩包", type=["zip", "tar", "gz", "7z"])

def reset():
    for key in ['processed', 'logs', 'summary', 'zip_data', 'errors']:
        st.session_state[key] = [] if key in ['logs', 'errors'] else (False if key == 'processed' else "")

if uploaded and st.button("开始处理", on_click=reset):
    if not api_key:
        st.warning("请输入 API Key")
        st.stop()
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # 保存文件
        input_path = tmp_path / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
        
        # 初始化LLM
        try:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            llm = init_chat_model(model, model_provider="openai", **kwargs)
        except Exception as e:
            st.error(f"LLM初始化失败: {e}")
            st.stop()
        
        # 创建目录
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with st.spinner("多Agent并发处理中..."):
            try:
                agent = TaskGraph(llm)
                result = agent.run(
                    archive_path=input_path.resolve().as_posix(),
                    work_dir=work_dir.resolve().as_posix(),
                    output_dir=output_dir.resolve().as_posix()
                )
                
                # 收集日志
                if "detection" in result:
                    d = result["detection"]
                    st.session_state.logs.append(f"[检测] 图片目录: {len(d.get('image_dirs', []))} 个")
                    st.session_state.logs.append(f"[检测] 点云目录: {len(d.get('pointcloud_dirs', []))} 个")
                    st.session_state.logs.append(f"[检测] 位姿目录: {d.get('egopose_dir', '无')}")
                    st.session_state.logs.append(f"[检测] 标定目录: {d.get('calib_dir', '无')}")
                
                if "agents_launched" in result:
                    for a in result["agents_launched"]:
                        st.session_state.logs.append(f"[Agent] {a}")
                
                if "errors" in result:
                    st.session_state.errors = result["errors"]
                
                if "messages" in result and result["messages"]:
                    st.session_state.summary = result["messages"][-1].content
                
                # 打包结果
                files = sum(1 for _ in output_dir.rglob('*') if _.is_file())
                st.session_state.logs.append(f"输出文件数: {files}")
                
                if files > 0:
                    shutil.make_archive(str(tmp_path / "result"), 'zip', str(output_dir))
                    st.session_state.zip_data = (tmp_path / "result.zip").read_bytes()
                    st.session_state.processed = True
                    st.success("处理完成!")
                else:
                    st.error("输出为空")
                    
            except Exception as e:
                st.error(f"处理失败: {e}")
                import traceback
                st.code(traceback.format_exc())

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

if not uploaded:
    st.info("请上传数据文件")
