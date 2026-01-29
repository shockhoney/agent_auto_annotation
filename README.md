# Agent 智能化数据标注系统

基于智能Agent的自动数据标注系统，可自动检测、转换和标准化各种数据标注格式（2D图像、3D点云、4D序列），转换为深度学习框架兼容的标准格式。

## ✨ 核心特性

- 🔍 **智能检测**: 递归扫描目录，自动识别数据类型
- 🔄 **格式转换**: 转换为标准格式（COCO、KITTI、Waymo风格）
- ⚙️ **配置生成**: 自动生成模型集成所需的配置文件
- ✅ **数据验证**: 验证转换后的数据可成功加载
- 🤖 **Agent智能**: 自动推断数据结构，智能决策转换策略

## 🎯 Agent特性说明

本系统是一个**智能Agent系统**，具有以下Agent特征：

1. **自主感知**: 通过递归探测工具自动构建文件系统的"心智模型"
2. **智能决策**: 基于检测结果自动选择最合适的转换策略（2D→COCO, 3D→KITTI, 4D→Waymo）
3. **自主行动**: 自动执行数据转换、配置生成和验证流程
4. **自我验证**: 通过验证脚本确保转换结果的正确性

## 📋 支持的格式

### 输入格式
- **2D图像**: JPG、PNG、BMP及各种标注格式
- **3D点云**: PCD、BIN、PLY、LAS
- **标注文件**: JSON、XML、TXT、CSV

### 输出格式
- **2D**: COCO JSON格式
- **3D**: KITTI格式（velodyne/、calib/）
- **4D**: Waymo风格（基于时间戳的序列组织）

## 🚀 安装

```bash
pip install -r requirements.txt
```

## 💻 使用方法

### 基础使用（自动检测）

```bash
python main.py --input ./raw_data --output ./standardized_data
```

### 指定格式转换

```bash
# 2D图像数据转COCO
python main.py --input ./images --output ./coco_output --format 2d

# 3D点云数据转KITTI
python main.py --input ./pointclouds --output ./kitti_output --format 3d

# 4D序列数据转Waymo风格
python main.py --input ./sequences --output ./waymo_output --format 4d
```

### 自定义训练集划分

```bash
# 70%训练集，30%验证集
python main.py --input ./data --output ./output --train-split 0.7
```

## 📁 项目结构

```
agent_auto_annotation/
├── src/
│   ├── converters/          # 格式转换器
│   │   ├── coco_converter.py      # COCO转换器
│   │   ├── kitti_converter.py     # KITTI转换器
│   │   └── waymo_converter.py     # Waymo转换器
│   ├── utils/               # 工具函数
│   │   ├── file_utils.py          # 文件操作工具
│   │   ├── coordinate_utils.py    # 坐标变换工具
│   │   └── logger.py              # 日志工具
│   ├── inspect_data.py      # 数据检测模块（Agent感知层）
│   ├── format_converter.py  # 格式转换编排器（Agent决策层）
│   ├── config_generator.py  # 配置文件生成器
│   └── validator.py         # 数据验证模块（Agent验证层）
├── tests/                   # 单元测试
├── examples/                # 示例代码
├── main.py                  # 程序入口
└── requirements.txt         # 依赖包
```

## 📦 输出结构

### 2D（COCO格式）
```
pre_labeling_2d/
├── images/
│   ├── batch_001_0001.png
│   └── batch_001_0002.png
├── annotations.json
└── metadata.json
```

### 3D（KITTI格式）
```
pre_labeling_3d/
├── velodyne/
│   ├── 000001.bin
│   └── 000002.bin
└── calib/
    ├── 000001.txt
    └── 000002.txt
```

### 4D（Waymo风格）
```
pre_labeling_4d/
└── sequence_A/
    ├── timestamp_1672531200/
    │   ├── lidar.bin
    │   ├── camera_front.jpg
    │   └── ego_pose.json
    └── timestamp_1672531201/
        └── ...
```

## 📖 更多文档

- [快速开始指南](QUICKSTART.md) - 快速上手教程
- [示例代码](examples/example_usage.py) - 编程接口示例

## 📄 许可证

MIT License
