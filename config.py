# 支持的文件格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
POINTCLOUD_EXTENSIONS = {'.pcd', '.ply', '.xyz', '.las', '.laz'}

# 压缩格式
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.gz', '.tgz', '.7z', '.rar'}

# 标定文件格式
CALIBRATION_EXTENSIONS = {'.yaml', '.yml', '.json'}

# LLM配置
LLM_CONFIG = {
    'provider': '通义千问',  # openai, anthropic, 或其他
    'api_key': '',         # 在这里配置您的API密钥
    'model': 'qwen-turbo',      # 或 claude-3-5-sonnet-20241022
    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',      # 可选: 自定义端点
}
