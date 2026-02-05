# Agent 智能化数据预处理方案

## 架构概述

本系统采用**多Agent并发架构**，通过协调器（OrchestratorAgent）分发任务给四个专用Agent并发执行，实现自动驾驶数据的标准化处理。

```
┌─────────────────────────────────────────────────────────────┐
│                      用户上传压缩包                            │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  OrchestratorAgent (协调器)                   │
│  1. 解压文件                                                  │
│  2. 扫描目录结构                                              │
│  3. 检测数据类型（图片/点云/位姿/标定）                         │
│  4. 创建输出目录                                              │
└─────────────────────┬───────────────────────────────────────┘
                      ▼ asyncio.gather (并发执行)
     ┌────────────────┼────────────────┬────────────────┐
     ▼                ▼                ▼                ▼
┌─────────┐    ┌─────────────┐   ┌──────────┐    ┌────────────┐
│ Image   │    │ PointCloud  │   │ EgoPose  │    │Calibration │
│ Agent   │    │   Agent     │   │  Agent   │    │   Agent    │
├─────────┤    ├─────────────┤   ├──────────┤    ├────────────┤
│转换为JPG│    │ 转换为PCD   │   │四元数→   │    │旋转向量→   │
│         │    │             │   │旋转矩阵  │    │旋转矩阵    │
└────┬────┘    └──────┬──────┘   └────┬─────┘    └─────┬──────┘
     │                │               │                │
     ▼                ▼               ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                       输出目录结构                            │
└─────────────────────────────────────────────────────────────┘
```

## 目录检测逻辑

协调器通过**代码逻辑**（非LLM决策）自动检测数据类型：

| 数据类型 | 检测规则 |
|---------|---------|
| 图片目录 | 包含 `.jpg`、`.png` 等图片文件 |
| 点云目录 | 包含 `.pcd`、`.bin`、`.ply` 文件 |
| 位姿目录 | 包含10+个JSON文件，且内容含 `orientation`、`position` 等关键词 |
| 标定目录 | 包含 `.yaml`、`.yml` 文件 |

## 工具集

| 工具函数 | 功能 |
|---------|------|
| `extract_archive` | 解压 zip/tar/7z 压缩包 |
| `scan_directory` | 扫描目录，统计文件类型分布 |
| `batch_convert_images` | 批量转换图片为 JPG 格式 |
| `batch_convert_pointclouds` | 批量转换点云为 PCD 格式 |
| `batch_process_egopose` | 处理位姿文件，将四元数转为旋转矩阵 |
| `batch_process_all_calibrations` | 处理标定文件，输出完整标定信息 |
| `create_output_structure` | 创建标准输出目录 |
| `generate_dataset_info` | 生成数据集统计信息 |

## 输出格式

### 目录结构

```
output/
├── egopose/              # 车身位姿信息
│   ├── 1744693431_33.json
│   └── ...
├── img/                  # 图片数据
│   ├── camera_front/
│   └── camera_left/
├── pointclouds/          # 点云数据
│   └── lidar_top/
├── sensors/              # 传感器标定信息
│   ├── camera_front_far.json
│   └── ...
└── dataset_info.json     # 数据集统计
```

### EgoPose 输出格式

保留原始数据，添加旋转矩阵：

```json
{
  "orientation": { ... },           // 原始四元数
  "position": { ... },              // 原始位置
  "velocity": { ... },              // 原始速度
  "rotation_matrix_global": [...],  // 新增：全局旋转矩阵
  "rotation_matrix_local": [...]    // 新增：局部旋转矩阵
}
```

### 传感器标定输出格式

```json
{
  "sensor_name": "camera_front_far",
  "sensor_type": "camera",
  "rotation_vector": [-1.17, 1.14, -1.20],      // 原始旋转向量
  "translation_vector": [4.28, 0.02, 1.95],     // 原始平移向量
  "rotation_matrix": [[...], [...], [...]],     // 3x3旋转矩阵
  "camera_intrinsic": [[fx,0,cx], [0,fy,cy], [0,0,1]],
  "distortion_coefficients": {"kc2":..., "kc3":..., ...},
  "image_size": {"width": 3840, "height": 2160},
  "is_fisheye": false,
  "camera_model": "polyn"
}
```

## 技术栈

- **框架**: LangChain + LangGraph
- **并发**: Python asyncio
- **图片处理**: Pillow
- **点云处理**: Open3D
- **UI界面**: Streamlit

## 使用方式

1. 运行 `streamlit run app.py`
2. 上传数据压缩包
3. 配置 LLM API（通义千问/DeepSeek）
4. 点击"开始处理"
5. 下载处理后的结果