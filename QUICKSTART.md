# 快速开始指南

## 安装

1. **安装依赖:**
   ```bash
   pip install -r requirements.txt
   ```

2. **验证安装:**
   ```bash
   python main.py --help
   ```

## 基本使用

### 示例1: 自动检测
```bash
python main.py --input ./my_raw_data --output ./converted_data
```

系统将自动：
1. 扫描您的数据目录
2. 检测数据类型（2D/3D/4D）
3. 转换为合适的格式
4. 生成配置文件
5. 验证输出结果

### 示例2: 强制指定格式
```bash
# 强制2D COCO转换
python main.py --input ./images --output ./coco_data --format 2d

# 强制3D KITTI转换
python main.py --input ./pointclouds --output ./kitti_data --format 3d
```

### 示例3: 自定义划分比例
```bash
# 70%训练集, 30%验证集
python main.py --input ./data --output ./output --train-split 0.7
```

## 预期输出

运行后，您将得到：

```
output/
├── pre_labeling_2d/  (或 pre_labeling_3d/ 或 pre_labeling_4d/)
│   ├── images/  (3D为 velodyne/)
│   ├── annotations.json
│   ├── metadata.json
│   ├── data_config.yaml
│   ├── dataset_manifest.json
│   ├── train.txt
│   └── val.txt
└── inspection_report.json
```

## 编程方式使用

```python
from format_converter import FormatConverter

# 创建转换器
converter = FormatConverter(
    input_dir="./raw_data",
    output_dir="./converted_data"
)

# 运行转换
result = converter.convert()

print(f"转换了 {result['conversion']['total_images']} 张图像")
```

## 常见问题

**问题:** "No module named 'open3d'"
- **解决方案:** 安装Open3D: `pip install open3d`

**问题:** "Permission denied"（权限被拒绝）
- **解决方案:** 使用适当的权限运行或更改输出目录

**问题:** "Unknown data type"（未知数据类型）
- **解决方案:** 使用 `--format` 强制指定格式

## 需要帮助？

- 查看 [README.md](file:///c:/Users/EDY/agent_auto_annotation/README.md) 获取详细文档
- 参考 [examples/example_usage.py](file:///c:/Users/EDY/agent_auto_annotation/examples/example_usage.py) 查看代码示例
- 运行测试: `python -m pytest tests/`
