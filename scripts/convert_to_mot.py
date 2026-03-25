#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO格式标签转换为MOT格式脚本
用于将UA-DETRAC数据集的YOLO格式标签转换为MOT格式，便于使用TrackEval进行评估
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO格式标签转换为MOT格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data-dir', type=str, default='data/UA-DETRAC-G2',
                       help='数据集根目录 (相对路径或绝对路径)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='数据集划分：train或val')
    parser.add_argument('--output-dir', type=str, default='data/mot_format',
                       help='MOT格式输出目录 (相对路径或绝对路径)')
    parser.add_argument('--config', type=str, default='configs/UA-DETRAC.yaml',
                       help='数据集配置文件路径 (相对路径或绝对路径)')
    parser.add_argument('--frame-rate', type=int, default=30,
                       help='视频帧率（用于生成seqinfo.ini）')
    parser.add_argument('--image-width', type=int, default=960,
                       help='图像宽度（用于生成seqinfo.ini）')
    parser.add_argument('--image-height', type=int, default=540,
                       help='图像高度（用于生成seqinfo.ini）')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    加载数据集配置文件，并将配置文件中的相对路径转换为基于当前工作目录的绝对路径

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 如果配置文件中有路径配置，将其转换为基于当前工作目录的绝对路径
        current_dir = os.getcwd()
        config_dir = Path(config_path).parent

        # 处理常见的路径字段
        if 'path' in config and config['path']:
            path_value = config['path']
            if isinstance(path_value, str) and not os.path.isabs(path_value):
                # 如果路径以 ./ 开头，优先基于当前工作目录解析
                if path_value.startswith('./') or path_value.startswith('.\\'):
                    config['path'] = os.path.abspath(os.path.join(current_dir, path_value))
                else:
                    # 否则尝试基于配置文件所在目录解析
                    config['path'] = os.path.abspath(os.path.join(config_dir, path_value))
                    # 如果目录不存在，再尝试基于当前工作目录解析
                    if not os.path.exists(config['path']):
                        config['path'] = os.path.abspath(os.path.join(current_dir, path_value))

        return config
    except Exception as e:
        print(f"加载配置文件时出错 {config_path}: {e}")
        return {}


def get_image_size(image_dir: str, split: str) -> Tuple[int, int]:
    """
    获取图像尺寸

    Args:
        image_dir: 图像目录
        split: 数据集划分

    Returns:
        (宽度, 高度)
    """
    try:
        # 确保路径是绝对路径
        if not os.path.isabs(image_dir):
            image_dir = os.path.abspath(image_dir)

        # 查找第一张图像
        split_dir = Path(image_dir) / 'images' / split
        if not split_dir.exists():
            print(f"图像目录不存在: {split_dir}")
            return 960, 540  # 默认尺寸

        image_files = list(split_dir.glob('*.jpg'))
        if not image_files:
            print(f"未找到图像文件: {split_dir}")
            return 960, 540  # 默认尺寸

        # 使用OpenCV获取图像尺寸
        try:
            import cv2
            img_path = str(image_files[0])
            # 使用支持中文路径的方式读取
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                height, width = img.shape[:2]
                return width, height
        except ImportError:
            print("OpenCV未安装，使用默认图像尺寸")

        return 960, 540  # 默认尺寸

    except Exception as e:
        print(f"获取图像尺寸时出错: {e}")
        return 960, 540  # 默认尺寸


def parse_yolo_label(label_line: str, img_width: int, img_height: int) -> Optional[Dict]:
    """
    解析YOLO格式标签行

    Args:
        label_line: YOLO格式标签行
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        解析后的标签字典，格式错误返回None
    """
    try:
        parts = label_line.strip().split()
        if len(parts) < 5:
            return None

        # YOLO格式: class_id x_center y_center width height (confidence)
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0

        # 转换为像素坐标
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height

        # 计算左上角坐标
        x1 = x_center_pixel - width_pixel / 2
        y1 = y_center_pixel - height_pixel / 2

        return {
            'class_id': class_id,
            'bbox': [x1, y1, width_pixel, height_pixel],
            'confidence': confidence,
        }

    except (ValueError, IndexError) as e:
        if 'verbose' in locals() and verbose:
            print(f"解析YOLO标签行时出错: {label_line}, 错误: {e}")
        return None


def convert_yolo_to_mot(yolo_labels_dir: str, output_dir: str, split: str,
                       class_names: List[str], img_width: int, img_height: int,
                       verbose: bool = False) -> Dict:
    """
    将YOLO格式标签转换为MOT格式

    Args:
        yolo_labels_dir: YOLO标签目录
        output_dir: 输出目录
        split: 数据集划分
        class_names: 类别名称列表
        img_width: 图像宽度
        img_height: 图像高度
        verbose: 是否显示详细输出

    Returns:
        转换统计信息
    """
    stats = {
        'total_frames': 0,
        'total_objects': 0,
        'converted_objects': 0,
        'class_distribution': {},
        'errors': 0,
    }

    try:
        # 创建输出目录
        mot_seq_dir = Path(output_dir) / f'UA-DETRAC-{split}' / 'gt'
        mot_seq_dir.mkdir(parents=True, exist_ok=True)

        # MOT格式文件
        mot_file = mot_seq_dir / 'gt.txt'

        # 获取所有标签文件
        yolo_dir = Path(yolo_labels_dir) / split
        if not yolo_dir.exists():
            print(f"YOLO标签目录不存在: {yolo_dir}")
            return stats

        label_files = sorted(list(yolo_dir.glob('*.txt')))
        if not label_files:
            print(f"未找到YOLO标签文件: {yolo_dir}")
            return stats

        print(f"找到 {len(label_files)} 个标签文件")

        with open(mot_file, 'w', encoding='utf-8') as f_out:
            object_id = 1  # MOT格式中每个对象有唯一ID

            for frame_idx, label_file in enumerate(label_files, start=1):
                if verbose and frame_idx % 100 == 0:
                    print(f"处理第 {frame_idx}/{len(label_files)} 帧...")

                try:
                    with open(label_file, 'r', encoding='utf-8') as f_in:
                        lines = f_in.readlines()

                    for line in lines:
                        label = parse_yolo_label(line, img_width, img_height)
                        if label is None:
                            stats['errors'] += 1
                            continue

                        # MOT格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
                        frame = frame_idx
                        obj_id = object_id
                        x1, y1, w, h = label['bbox']
                        conf = label['confidence']
                        class_id = label['class_id'] + 1  # MOT格式中类别从1开始

                        # 写入MOT格式行
                        mot_line = f"{frame},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},{class_id},1,1\n"
                        f_out.write(mot_line)

                        # 更新统计信息
                        stats['total_objects'] += 1
                        stats['converted_objects'] += 1
                        object_id += 1

                        # 更新类别分布
                        class_name = class_names[label['class_id']] if label['class_id'] < len(class_names) else f'class_{label["class_id"]}'
                        stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1

                except Exception as e:
                    if verbose:
                        print(f"处理文件 {label_file} 时出错: {e}")
                    stats['errors'] += 1

        stats['total_frames'] = len(label_files)

        print(f"MOT格式标签已保存到: {mot_file}")
        return stats

    except Exception as e:
        print(f"转换YOLO到MOT格式时出错: {e}")
        return stats


def create_seqinfo_ini(output_dir: str, split: str, seq_name: str,
                      img_width: int, img_height: int, frame_rate: int,
                      seq_length: int) -> None:
    """
    创建seqinfo.ini文件

    Args:
        output_dir: 输出目录
        split: 数据集划分
        seq_name: 序列名称
        img_width: 图像宽度
        img_height: 图像高度
        frame_rate: 帧率
        seq_length: 序列长度（帧数）
    """
    try:
        seq_dir = Path(output_dir) / f'UA-DETRAC-{split}'
        seq_dir.mkdir(parents=True, exist_ok=True)

        ini_file = seq_dir / 'seqinfo.ini'

        content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate={frame_rate}
seqLength={seq_length}
imWidth={img_width}
imHeight={img_height}
imExt=.jpg
"""

        with open(ini_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"seqinfo.ini已创建: {ini_file}")

    except Exception as e:
        print(f"创建seqinfo.ini时出错: {e}")


def create_trackeval_structure(output_dir: str, split: str) -> None:
    """
    创建TrackEval所需的目录结构

    Args:
        output_dir: 输出目录
        split: 数据集划分
    """
    try:
        # 创建目录结构
        seq_dir = Path(output_dir) / f'UA-DETRAC-{split}'

        # gt目录（已由convert_yolo_to_mot创建）
        gt_dir = seq_dir / 'gt'
        gt_dir.mkdir(exist_ok=True)

        # img1目录（图像软链接或占位符）
        img_dir = seq_dir / 'img1'
        img_dir.mkdir(exist_ok=True)

        # 创建说明文件
        readme_file = seq_dir / 'README.txt'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"""UA-DETRAC {split} set for MOT evaluation

Directory structure:
- gt/gt.txt: Ground truth annotations in MOT format
- img1/: Image frames (optional, can be symlinks to original images)
- seqinfo.ini: Sequence information

Generated by convert_to_mot.py
""")

        print(f"TrackEval目录结构已创建: {seq_dir}")

    except Exception as e:
        print(f"创建TrackEval目录结构时出错: {e}")


def print_conversion_stats(stats: Dict, class_names: List[str]) -> None:
    """
    打印转换统计信息

    Args:
        stats: 统计信息字典
        class_names: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("📊 转换统计信息")
    print("=" * 60)

    print(f"总帧数: {stats['total_frames']}")
    print(f"总对象数: {stats['total_objects']}")
    print(f"成功转换: {stats['converted_objects']}")
    print(f"转换错误: {stats['errors']}")

    if stats['converted_objects'] > 0:
        success_rate = (stats['converted_objects'] / stats['total_objects']) * 100
        print(f"转换成功率: {success_rate:.1f}%")

    print("\n📈 类别分布:")
    for class_name, count in stats['class_distribution'].items():
        percentage = (count / stats['converted_objects']) * 100 if stats['converted_objects'] > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()

    # 将所有路径转换为基于当前工作目录的绝对路径
    current_dir = os.getcwd()

    # 处理数据目录路径
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(current_dir, args.data_dir))

    # 处理输出目录路径
    if args.output_dir and not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(current_dir, args.output_dir))

    # 处理配置文件路径
    if args.config and not os.path.isabs(args.config):
        args.config = os.path.abspath(os.path.join(current_dir, args.config))

    print("=" * 60)
    print("🔄 YOLO格式标签转换为MOT格式")
    print("=" * 60)

    # 加载配置文件
    config = load_config(args.config)
    if not config:
        print("错误: 无法加载配置文件")
        sys.exit(1)

    # 获取类别信息
    class_names = config.get('names', ['car', 'bus', 'van', 'truck'])
    print(f"类别: {class_names}")

    # 获取图像尺寸
    img_width, img_height = get_image_size(args.data_dir, args.split)
    print(f"图像尺寸: {img_width}x{img_height}")

    # 转换YOLO标签到MOT格式
    yolo_labels_dir = Path(args.data_dir) / 'labels'
    stats = convert_yolo_to_mot(
        str(yolo_labels_dir),
        args.output_dir,
        args.split,
        class_names,
        img_width,
        img_height,
        args.verbose
    )

    # 打印统计信息
    print_conversion_stats(stats, class_names)

    if stats['converted_objects'] == 0:
        print("警告: 未转换任何对象，请检查输入数据")
        return

    # 创建seqinfo.ini
    seq_name = f"UA-DETRAC-{args.split}"
    create_seqinfo_ini(
        args.output_dir,
        args.split,
        seq_name,
        img_width,
        img_height,
        args.frame_rate,
        stats['total_frames']
    )

    # 创建TrackEval目录结构
    create_trackeval_structure(args.output_dir, args.split)

    # 保存转换配置
    config_file = Path(args.output_dir) / f'UA-DETRAC-{args.split}' / 'conversion_config.json'
    config_data = {
        'data_dir': args.data_dir,
        'split': args.split,
        'output_dir': args.output_dir,
        'config_file': args.config,
        'image_size': [img_width, img_height],
        'frame_rate': args.frame_rate,
        'class_names': class_names,
        'conversion_stats': stats,
        'conversion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"转换配置已保存: {config_file}")
    except Exception as e:
        print(f"保存转换配置时出错: {e}")

    print("\n✅ 转换完成!")
    print(f"MOT格式数据已保存到: {Path(args.output_dir) / f'UA-DETRAC-{args.split}'}")


if __name__ == '__main__':
    main()