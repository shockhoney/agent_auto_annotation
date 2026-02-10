# Agent 智能化数据预处理方案

## 架构概述

本系统采用**多Agent并发架构**，通过协调器（OrchestratorAgent）自动检测数据类型并分发任务给专用Agent并发执行，实现自动驾驶数据的标准化处理。

```
┌─────────────────────────────────────────────────────────────┐
│                      用户上传压缩包                            │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  OrchestratorAgent (协调器)                   │
│  1. 解压文件                                                  │
│  2. 扫描目录结构                                              │
│  3. 自动检测数据类型 (图片/点云/位姿/标定)                      │
│  4. 手动路径回退 (自动检测失败时)                               │
│  5. 创建输出目录                                              │
└─────────────────────┬───────────────────────────────────────┘
                      ▼ asyncio.gather (并发执行)
     ┌────────────────┼────────────────┬────────────────┐
     ▼                ▼                ▼                ▼
┌─────────┐    ┌─────────────┐   ┌──────────┐    ┌────────────┐
│ Image   │    │ PointCloud  │   │  Pose    │    │Calibration │
│ Agent   │    │   Agent     │   │  Agent   │    │   Agent    │
├─────────┤    ├─────────────┤   ├──────────┤    ├────────────┤
│转换为JPG│    │转换为PCD    │   │四元数->  │    │提取内外参  │
│         │    │(3D/4D)      │   │旋转矩阵  │    │+旋转矩阵   │
└────┬────┘    └──────┬──────┘   └────┬─────┘    └─────┬──────┘
     │                │               │                │
     ▼                ▼               ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    generate_dataset_info                     │
│                  生成 dataset_info.json                       │
└─────────────────────────────────────────────────────────────┘
```

## 任务模式

| 模式 | 处理内容 |
|------|---------|
| **Auto** | 自动检测数据类型并决定处理范围 |
| **2D** | 仅处理图片 |
| **3D** | 图片 + 3D点云 + 标定 + 位姿 |
| **4D** | 图片 + 3D点云 + 4D点云 + 标定 + 位姿 |

## 数据检测逻辑

协调器通过**代码逻辑**（非LLM决策）自动检测数据类型：

| 数据类型 | 检测规则 |
|---------|---------| 
| 图片目录 | 包含 `.jpg`、`.png` 等图片文件的目录 |
| 3D点云目录 | 包含 `.pcd`、`.bin`、`.ply` 文件 (<=4列) |
| 4D点云目录 | 包含 `.pcd`、`.bin`、`.ply` 文件 (>4列，含时间/强度维度) |
| 位姿文件 | JSON目录 (含 `orientation`/`position`) 或 TXT文件 |
| 标定文件 | JSON文件 (含 `intrinsic`/`extrinsic`) 或 YAML目录 |

**手动路径回退**：自动检测失败时，可在侧边栏手动指定文件相对路径。

## 手动指定 (可选)

当自动检测失败时，以下功能提供手动回退：

### 文件路径指定

在侧边栏输入文件或文件夹相对路径（相对于压缩包根目录）：

- **标定文件路径**：如 `metadata/calib_anno/params.json` 或 `calib` (目录)
- **位姿文件路径**：如 `slam_results/pose.txt` 或 `egopose` (目录)

### 字段映射

当数据文件中的键名与标准键名不一致时，可通过字段映射解决：

**YAML标定映射** (每行格式: `你的键名=标准键名`)
```
focal_x=fx
focal_y=fy
principal_x=cx
principal_y=cy
rot_vec=r_s2b
trans_vec=t_s2b
```
标准键名: `fx`, `fy`, `cx`, `cy`, `r_s2b`, `t_s2b`, `kc2`, `kc3`, `kc4`, `kc5`, `sensor_name`, `is_fisheye`

**JSON标定映射**
```
waican=extrinsic
neican=intrinsic
jibian=distortion
```
标准键名: `intrinsic`, `extrinsic`, `distortion`, `translation`

**JSON位姿映射**
```
ori=orientation
quat=quaternion_local
pos=position
```
标准键名: `orientation`, `quaternion_local`, `position`

**TXT位姿格式** (固定格式，无需映射): `帧号,时间戳,x,y,z,qx,qy,qz,qw`

## 工具集

| 工具函数 | 功能 |
|---------|------|
| `extract_archive` | 解压 zip/tar/7z 压缩包 |
| `scan_directory` | 扫描目录，按父目录统计文件类型分布 |
| `batch_convert_images` | 批量转换图片为 JPG 格式 |
| `batch_convert_pointclouds` | 批量转换3D点云为 PCD 格式 |
| `batch_convert_4d_pointclouds` | 批量转换4D点云为 PCD 格式 |
| `batch_process_egopose` | 处理JSON位姿，四元数转旋转矩阵 (支持字段映射) |
| `batch_process_pose_txt` | 处理TXT位姿，每帧输出JSON |
| `batch_process_all_calibrations` | 处理YAML标定文件 (支持字段映射) |
| `batch_process_calibration_json` | 处理JSON标定文件 (支持字段映射) |
| `create_output_structure` | 根据任务模式创建输出目录 |
| `generate_dataset_info` | 生成 dataset_info.json 统计信息 |

## 输出格式

### 目录结构

```
output/
├── calibration/          # 传感器标定信息
│   ├── camera_front.json
│   └── fisheye-left.json
├── camera/               # 图片数据
│   ├── camera_front/
│   └── fisheye-left/
├── egopose/              # 车身位姿信息
│   ├── 1744693431_33.json
│   └── ...
├── lidar/                # 3D点云数据
│   └── lidar_top/
├── lidar_4d/             # 4D点云数据
│   └── map.pcd
└── dataset_info.json     # 数据集统计
```

### EgoPose 输出格式

保留原始数据，添加局部旋转矩阵：

```json
{
  "orientation": { ... },
  "position": { ... },
  "velocity": { ... },
  "rotation_matrix_local": [[...], [...], [...]]
}
```

### 标定输出格式 (JSON源)

```json
{
  "sensor_name": "fisheye-front",
  "rotation_matrix": [[...], [...], [...]],
  "translation_vector": [...],
  "camera_intrinsic": [[fx,0,cx], [0,fy,cy], [0,0,1]],
  "distortion_coefficients": [...],
  "is_fisheye": true
}
```

### 标定输出格式 (YAML源)

```json
{
  "sensor_name": "camera_front_far",
  "rotation_vector": [-1.17, 1.14, -1.20],
  "translation_vector": [4.28, 0.02, 1.95],
  "rotation_matrix": [[...], [...], [...]],
  "camera_intrinsic": [[fx,0,cx], [0,fy,cy], [0,0,1]],
  "distortion_coefficients": {"kc2": ..., "kc3": ...},
  "image_size": {"width": 3840, "height": 2160},
  "is_fisheye": false,
  "camera_model": "polyn"
}
```

### TXT位姿输出格式

```json
{
  "frame_id": "000001",
  "timestamp": 1744693431.33,
  "position": {"x": 1.0, "y": 2.0, "z": 3.0},
  "quaternion": {"qx": 0, "qy": 0, "qz": 0, "qw": 1},
  "rotation_matrix": [[...], [...], [...]],
  "raw_line": "000001,1744693431.33,1.0,2.0,3.0,0,0,0,1"
}
```

## 技术栈

- **框架**: LangChain + LangGraph
- **并发**: Python asyncio + ThreadPoolExecutor
- **图片处理**: Pillow
- **点云处理**: Open3D / NumPy
- **UI界面**: Streamlit

## 使用方式

1. 运行 `streamlit run app.py`
2. 上传数据压缩包（支持多文件并行处理）
3. 选择任务模式（Auto/2D/3D/4D）
4. (可选) 配置手动路径和字段映射
5. 点击"开始处理"
6. 下载处理后的结果