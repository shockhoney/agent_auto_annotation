"""
示例脚本：演示如何以编程方式使用系统
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inspect_data import inspect_dataset
from converters import COCOConverter, KITTIConverter, WaymoConverter
from config_generator import generate_config
from validator import DataValidator
from utils.logger import logger


def example_2d_conversion():
    """示例：将2D图像数据转换为COCO格式"""
    logger.info("=" * 60)
    logger.info("示例：2D图像转COCO格式")
    logger.info("=" * 60)
    
    # 步骤1: 检测数据
    input_dir = "./examples/sample_2d_data"
    inspection = inspect_dataset(input_dir)
    logger.info(f"检测到的类型: {inspection['inferred_type']}")
    
    # 步骤2: 转换
    output_dir = "./examples/output_2d"
    converter = COCOConverter(output_dir)
    stats = converter.convert_from_directory(input_dir)
    converter.save()
    
    logger.info(f"已转换 {stats['total_images']} 张图像")
    
    # 步骤3: 生成配置
    config = generate_config(output_dir, '2d')
    logger.info(f"配置已生成: {config}")
    
    # 步骤4: 验证
    validator = DataValidator(output_dir, '2d')
    validator.validate()


def example_3d_conversion():
    """示例：将3D点云数据转换为KITTI格式"""
    logger.info("=" * 60)
    logger.info("示例：3D点云转KITTI格式")
    logger.info("=" * 60)
    
    # 步骤1: 检测
    input_dir = "./examples/sample_3d_data"
    inspection = inspect_dataset(input_dir)
    
    # 步骤2: 转换
    output_dir = "./examples/output_3d"
    converter = KITTIConverter(output_dir)
    stats = converter.convert_from_directory(input_dir)
    
    logger.info(f"已转换 {stats['total_frames']} 帧")
    
    # 步骤3: 生成配置
    config = generate_config(output_dir, '3d')
    
    # 步骤4: 验证
    validator = DataValidator(output_dir, '3d')
    validator.validate()


def example_programmatic_usage():
    """示例：以编程方式使用转换器"""
    logger.info("=" * 60)
    logger.info("示例：编程API使用方式")
    logger.info("=" * 60)
    
    # 创建COCO转换器
    converter = COCOConverter("./examples/manual_output")
    
    # 手动添加图像和标注
    image_id = converter.add_image("path/to/image.jpg")
    
    # 添加标注: [x, y, width, height]
    converter.add_annotation(
        image_id=image_id,
        category_name="汽车",
        bbox=[100, 100, 200, 150]
    )
    
    # 保存
    converter.save()
    
    logger.info("手动标注创建成功")


if __name__ == '__main__':
    # 运行示例
    print("\n" + "=" * 60)
    print("Agent智能化数据标注系统 - 使用示例")
    print("=" * 60 + "\n")
    
    # 取消注释以运行特定示例:
    # example_2d_conversion()
    # example_3d_conversion()
    example_programmatic_usage()
