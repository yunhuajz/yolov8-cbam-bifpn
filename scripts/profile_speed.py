#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 深度耗时分析脚本
对 expA/expB/expC/expD 四个模型进行性能剖析
分析各模型的推理耗时分布

Usage:
    python scripts/profile_speed.py
    python scripts/profile_speed.py --num_frames 100 --device 0
"""

import os
import sys
import argparse
import time
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    import numpy as np
    import torch
except ImportError as e:
    print(f"导入依赖库失败: {e}")
    print("请安装所需依赖: pip install ultralytics numpy torch")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 深度耗时分析 - 性能剖析脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 模型配置
    parser.add_argument('--expA', type=str, default='runs/train/expA_baseline/weights/best.pt',
                       help='实验A模型路径')
    parser.add_argument('--expB', type=str, default='runs/train/expB_cbam/weights/best.pt',
                       help='实验B模型路径')
    parser.add_argument('--expC', type=str, default='runs/train/expC_bifpn/weights/best.pt',
                       help='实验C模型路径')
    parser.add_argument('--expD', type=str, default='runs/train/expD_combined/weights/best.pt',
                       help='实验D模型路径')

    # 测试数据配置
    parser.add_argument('--data', type=str, default='data/UA-DETRAC-G2/images/val',
                       help='验证集图像目录')
    parser.add_argument('--num_frames', type=int, default=100,
                       help='测试帧数（取前N帧）')

    # 推理参数
    parser.add_argument('--device', type=str, default='0',
                       help='推理设备，如: 0 (GPU0), cpu')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IOU阈值')

    # 输出配置
    parser.add_argument('--output', type=str, default='metrics/detailed_speed_analysis.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--warmup', type=int, default=5,
                       help='预热帧数（不计入统计）')

    return parser.parse_args()


def resolve_path(path: str, current_dir: str) -> str:
    """
    将相对路径转换为绝对路径

    Args:
        path: 输入路径
        current_dir: 当前工作目录

    Returns:
        绝对路径
    """
    if path and not os.path.isabs(path):
        return os.path.abspath(os.path.join(current_dir, path))
    return path


def get_image_list(data_dir: str, num_frames: int) -> List[str]:
    """
    获取图像列表

    Args:
        data_dir: 图像目录
        num_frames: 需要的图像数量

    Returns:
        图像路径列表
    """
    # 支持的图像格式
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []

    for ext in image_exts:
        pattern = os.path.join(data_dir, ext)
        image_files.extend(glob.glob(pattern))
        # 同时检查大写扩展名
        pattern_upper = os.path.join(data_dir, ext.upper())
        image_files.extend(glob.glob(pattern_upper))

    # 去重并排序
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"警告: 在 {data_dir} 中未找到图像文件")
        return []

    # 取前N帧
    if len(image_files) > num_frames:
        image_files = image_files[:num_frames]

    print(f"找到 {len(image_files)} 个图像文件用于测试")
    return image_files


def load_model(model_path: str, device: str) -> Optional[YOLO]:
    """
    加载YOLOv8模型

    Args:
        model_path: 模型文件路径
        device: 设备

    Returns:
        YOLO模型对象，失败返回None
    """
    try:
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return None

        print(f"  加载模型: {model_path}")
        model = YOLO(model_path)

        # 预热模型
        print(f"  预热模型...")
        # 修复设备字符串处理
        device_str = f'cuda:{device}' if device != 'cpu' and device.isdigit() else device
        dummy_input = torch.zeros(1, 3, 640, 640).to(device_str)
        with torch.no_grad():
            for _ in range(3):
                _ = model.predict(source=dummy_input, device=device_str, verbose=False)

        return model

    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def profile_model(model: YOLO, image_list: List[str], device: str,
                  imgsz: int, conf: float, iou: float,
                  warmup: int = 5) -> Dict:
    """
    对单个模型进行性能剖析

    Args:
        model: YOLO模型
        image_list: 图像路径列表
        device: 设备
        imgsz: 输入尺寸
        conf: 置信度阈值
        iou: NMS IOU阈值
        warmup: 预热帧数

    Returns:
        性能指标字典
    """
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []
    num_boxes_list = []

    print(f"  开始推理 ({len(image_list)} 帧)...")

    for idx, image_path in enumerate(image_list):
        try:
            # 执行推理
            results = model.predict(
                source=image_path,
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

            # 提取速度信息
            if results and len(results) > 0:
                speed = results[0].speed  # {'preprocess': float, 'inference': float, 'postprocess': float}

                # 预热帧不计入统计
                if idx >= warmup:
                    preprocess_times.append(speed['preprocess'])
                    inference_times.append(speed['inference'])
                    postprocess_times.append(speed['postprocess'])
                    total_times.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])

                    # 记录检测框数量
                    num_boxes = len(results[0].boxes) if results[0].boxes is not None else 0
                    num_boxes_list.append(num_boxes)

        except Exception as e:
            print(f"    处理图像时出错 {image_path}: {e}")
            continue

    # 计算平均值
    num_valid_frames = len(preprocess_times)
    if num_valid_frames == 0:
        print("  警告: 没有有效的推理结果")
        return {}

    results_dict = {
        'avg_preprocess_ms': np.mean(preprocess_times),
        'avg_inference_ms': np.mean(inference_times),
        'avg_postprocess_ms': np.mean(postprocess_times),
        'avg_total_ms': np.mean(total_times),
        'std_preprocess_ms': np.std(preprocess_times),
        'std_inference_ms': np.std(inference_times),
        'std_postprocess_ms': np.std(postprocess_times),
        'std_total_ms': np.std(total_times),
        'avg_boxes': np.mean(num_boxes_list),
        'num_frames': num_valid_frames
    }

    return results_dict


def save_results(results_list: List[Dict], output_path: str):
    """
    保存结果到CSV

    Args:
        results_list: 结果列表
        output_path: 输出文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 写入CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not results_list:
                return

            fieldnames = results_list[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\n结果已保存到: {output_path}")

    except Exception as e:
        print(f"保存结果时出错: {e}")


def print_summary(results_list: List[Dict]):
    """
    打印结果摘要

    Args:
        results_list: 结果列表
    """
    print("\n" + "=" * 80)
    print("📊 性能剖析结果摘要")
    print("=" * 80)

    for result in results_list:
        exp = result['experiment']
        print(f"\n实验 {exp}:")
        print(f"  平均预处理时间:  {result['avg_preprocess_ms']:.2f} ± {result['std_preprocess_ms']:.2f} ms")
        print(f"  平均推理时间:    {result['avg_inference_ms']:.2f} ± {result['std_inference_ms']:.2f} ms")
        print(f"  平均后处理时间:  {result['avg_postprocess_ms']:.2f} ± {result['std_postprocess_ms']:.2f} ms")
        print(f"  平均总时间:      {result['avg_total_ms']:.2f} ± {result['std_total_ms']:.2f} ms")
        print(f"  平均检测框数量:  {result['avg_boxes']:.2f}")
        print(f"  测试帧数:        {result['num_frames']}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    args = parse_args()

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 解析路径
    args.expA = resolve_path(args.expA, current_dir)
    args.expB = resolve_path(args.expB, current_dir)
    args.expC = resolve_path(args.expC, current_dir)
    args.expD = resolve_path(args.expD, current_dir)
    args.data = resolve_path(args.data, current_dir)
    args.output = resolve_path(args.output, current_dir)

    print("=" * 80)
    print("🚀 YOLOv8 深度耗时分析 - 性能剖析")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  实验A模型: {args.expA}")
    print(f"  实验B模型: {args.expB}")
    print(f"  实验C模型: {args.expC}")
    print(f"  实验D模型: {args.expD}")
    print(f"  测试数据: {args.data}")
    print(f"  测试帧数: {args.num_frames}")
    print(f"  预热帧数: {args.warmup}")
    print(f"  设备: {args.device}")
    print(f"  输出文件: {args.output}")

    # 获取图像列表
    image_list = get_image_list(args.data, args.num_frames)
    if not image_list:
        print("错误: 无法获取测试图像")
        sys.exit(1)

    # 定义实验配置
    experiments = [
        ('A', args.expA, '基线 (PANet)'),
        ('B', args.expB, 'CBAM注意力'),
        ('C', args.expC, 'BiFPN'),
        ('D', args.expD, '联合 (CBAM+BiFPN)')
    ]

    # 存储结果
    all_results = []

    # 依次测试每个实验
    for exp_name, model_path, exp_desc in experiments:
        print(f"\n{'-' * 80}")
        print(f"🔬 实验 {exp_name}: {exp_desc}")
        print(f"{'-' * 80}")

        # 检查模型是否存在
        if not os.path.exists(model_path):
            print(f"  跳过: 模型文件不存在: {model_path}")
            continue

        # 加载模型
        model = load_model(model_path, args.device)
        if model is None:
            print(f"  跳过: 无法加载模型")
            continue

        # 性能剖析
        profile_results = profile_model(
            model, image_list, args.device,
            args.imgsz, args.conf, args.iou, args.warmup
        )

        if profile_results:
            profile_results['experiment'] = exp_name
            profile_results['description'] = exp_desc
            all_results.append(profile_results)

        # 释放GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印结果摘要
    if all_results:
        print_summary(all_results)

        # 保存结果
        save_results(all_results, args.output)
        print("\n✅ 性能剖析完成!")
    else:
        print("\n❌ 没有有效的测试结果")
        sys.exit(1)


if __name__ == '__main__':
    main()
