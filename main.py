"""
Agent智能化数据标注系统 - 主程序入口
自动编排完整流程: 检测 → 转换 → 配置 → 验证
"""

import argparse
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from format_converter import FormatConverter
from config_generator import ConfigGenerator
from validator import DataValidator
from utils.logger import logger, setup_logger


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Agent智能化数据标注系统 - 自动转换和标准化标注数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动检测并转换
  python main.py --input ./raw_data --output ./standardized_data
  
  # 强制指定格式
  python main.py --input ./images --output ./coco_output --format 2d
  
  # 跳过验证步骤
  python main.py --input ./data --output ./output --no-validate
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入目录（包含原始标注数据）'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出目录（存放标准化后的数据）'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['2d', '3d', '4d', 'auto'],
        default='auto',
        help='强制指定格式（默认: 自动检测）'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='训练集/验证集划分比例（默认: 0.8）'
    )
    
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='跳过train/val划分（用于推理模式）'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='跳过验证步骤'
    )
    
    parser.add_argument(
        '--log-file',
        help='日志文件路径（可选）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志器
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_file=args.log_file, level=log_level)
    
    logger.info("🤖 Agent智能化数据标注系统")
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    
    try:
        # 步骤 1 & 2: 检测和转换
        force_type = None if args.format == 'auto' else args.format
        converter = FormatConverter(args.input, args.output, force_type=force_type)
        result = converter.convert()
        
        # 确定输出子目录
        data_type = result['data_type']
        output_subdir = Path(args.output) / f"pre_labeling_{data_type}"
        
        # 步骤 3: 生成配置
        logger.info("\n" + "=" * 60)
        logger.info("生成配置文件")
        logger.info("=" * 60)
        
        config_gen = ConfigGenerator(str(output_subdir), data_type)
        config = config_gen.generate_config(
            train_split=args.train_split,
            skip_split=args.no_split  # Pass skip_split flag
        )
        
        # 步骤 4: 验证（如果未跳过）
        if not args.no_validate:
            logger.info("\n" + "=" * 60)
            logger.info("验证转换后的数据")
            logger.info("=" * 60)
            
            validator = DataValidator(str(output_subdir), data_type)
            validation = validator.validate()
            
            if validation['failed']:
                logger.warning("\n⚠ 验证完成，但存在错误")
                logger.warning("请检查上述问题")
                return 1
        
        # 成功
        logger.info("\n" + "🎉 " * 20)
        logger.info("✓ 处理流程成功完成！")
        logger.info(f"✓ 输出目录: {output_subdir}")
        logger.info(f"✓ 配置文件: {output_subdir / 'data_config.yaml'}")
        logger.info(f"✓ 数据已准备就绪，可直接用于模型训练")
        logger.info("🎉 " * 20)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
