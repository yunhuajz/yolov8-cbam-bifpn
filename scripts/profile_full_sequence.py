#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 全量视频序列性能剖析脚本
================================

对 ExpA, ExpB, ExpC, ExpD 四个模型进行最严谨的性能评估。
处理完整视频序列，使用 ByteTrack 进行跟踪，精确计时各阶段耗时。

Usage:
    python scripts/profile_full_sequence.py
    python scripts/profile_full_sequence.py --sequence MVI_40212 --device 0

Output:
    metrics/full_sequence_perf_details.csv - 详细性能数据
"""

import os
import sys
import argparse
import time
import glob
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 注册自定义模块 (CBAM, BiFPN)
try:
    from ultralytics.nn.modules import CBAM, BiFPN
    import ultralytics.nn.tasks as tasks

    if not hasattr(tasks, 'CBAM'):
        tasks.CBAM = CBAM
    if not hasattr(tasks, 'BiFPN'):
        tasks.BiFPN = BiFPN
except ImportError:
    pass

try:
    from ultralytics import YOLO
    import numpy as np
    import torch
    import cv2
except ImportError as e:
    print(f"导入依赖库失败: {e}")
    print("请安装所需依赖: pip install ultralytics opencv-python numpy torch")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 全量视频序列性能剖析 - 深度耗时分析',
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
    parser.add_argument('--sequence', type=str, default='MVI_40212',
                       help='验证集视频序列名称')
    parser.add_argument('--data-root', type=str, default='data/UA-DETRAC-G2/images/val',
                       help='验证集根目录')

    # 推理参数
    parser.add_argument('--device', type=str, default='0',
                       help='推理设备，如: 0 (GPU0), cpu')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IOU阈值')

    # 跟踪器配置
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                       help='跟踪器配置名称')

    # 输出配置
    parser.add_argument('--output', type=str, default='metrics/full_sequence_perf_details.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--warmup', type=int, default=3,
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


def get_sequence_images(data_root: str, sequence_name: str = None) -> List[str]:
    """
    获取指定视频序列的所有图像

    Args:
        data_root: 验证集根目录
        sequence_name: 视频序列名称（可选，如提供则查找子文件夹）

    Returns:
        图像路径列表（按文件名排序）
    """
    # 如果指定了序列名，先尝试查找子文件夹
    if sequence_name:
        sequence_path = os.path.join(data_root, sequence_name)
        if os.path.exists(sequence_path) and os.path.isdir(sequence_path):
            search_path = sequence_path
        else:
            # 序列文件夹不存在，直接使用数据根目录
            search_path = data_root
            print(f"注意: 序列文件夹 {sequence_name} 不存在，使用 {data_root} 中的所有图像")
    else:
        search_path = data_root

    if not os.path.exists(search_path):
        print(f"错误: 路径不存在: {search_path}")
        return []

    # 支持的图像格式
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for ext in image_exts:
        pattern = os.path.join(search_path, ext)
        image_files.extend(glob.glob(pattern))
        pattern_upper = os.path.join(search_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper))

    # 去重并排序
    image_files = sorted(list(set(image_files)))

    print(f"找到 {len(image_files)} 帧图像用于测试")
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


def profile_sequence_tracking(
    model: YOLO,
    image_list: List[str],
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    tracker: str,
    warmup: int = 3
) -> Tuple[Dict, List[Dict]]:
    """
    对单个模型进行完整序列跟踪性能剖析

    Args:
        model: YOLO模型
        image_list: 图像路径列表
        device: 设备
        imgsz: 输入尺寸
        conf: 置信度阈值
        iou: NMS IOU阈值
        tracker: 跟踪器配置名称
        warmup: 预热帧数

    Returns:
        (统计摘要字典, 每帧详细数据列表)
    """
    frame_details = []

    # 计时数据
    total_track_times = []  # 使用 perf_counter 测量的总时间
    preprocess_times = []
    inference_times = []
    postprocess_times = []  # 这是 NMS + 其他后处理时间
    association_times = []  # Association_plus_NMS = Total - preprocess - inference

    # 统计信息
    num_candidates_list = []  # 原始候选框数量
    num_tracks_list = []      # 最终跟踪目标数

    print(f"  开始处理序列 ({len(image_list)} 帧)...")

    for idx, image_path in enumerate(image_list):
        try:
            # 使用 perf_counter 高精度计时
            start_time = time.perf_counter()

            # 执行跟踪
            results = model.track(
                source=image_path,
                tracker=tracker,
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

            end_time = time.perf_counter()
            total_track_time = (end_time - start_time) * 1000  # 转换为毫秒

            # 提取速度信息和统计
            if results and len(results) > 0:
                result = results[0]
                speed = result.speed  # {'preprocess': float, 'inference': float, 'postprocess': float}

                # 获取候选框和跟踪数量
                num_candidates = len(result.boxes) if result.boxes is not None else 0
                num_tracks = 0
                if result.boxes is not None and result.boxes.id is not None:
                    num_tracks = len(result.boxes.id)

                # 预热帧不计入统计
                if idx >= warmup:
                    total_track_times.append(total_track_time)
                    preprocess_times.append(speed['preprocess'])
                    inference_times.append(speed['inference'])
                    postprocess_times.append(speed['postprocess'])

                    # 计算 Association_plus_NMS
                    # 注意: speed['postprocess'] 已经包含 NMS
                    # Association_plus_NMS = Total_Track_Time - preprocess - inference
                    association_time = total_track_time - speed['preprocess'] - speed['inference']
                    association_times.append(max(0, association_time))  # 确保非负

                    num_candidates_list.append(num_candidates)
                    num_tracks_list.append(num_tracks)

                # 记录每帧详细信息
                frame_details.append({
                    'frame_idx': idx,
                    'image_name': os.path.basename(image_path),
                    'total_track_ms': total_track_time,
                    'preprocess_ms': speed['preprocess'],
                    'inference_ms': speed['inference'],
                    'postprocess_ms': speed['postprocess'],
                    'association_plus_nms_ms': max(0, total_track_time - speed['preprocess'] - speed['inference']),
                    'num_candidates': num_candidates,
                    'num_tracks': num_tracks
                })

        except Exception as e:
            print(f"    处理图像时出错 {image_path}: {e}")
            continue

    # 计算统计摘要
    num_valid_frames = len(total_track_times)
    if num_valid_frames == 0:
        print("  警告: 没有有效的跟踪结果")
        return {}, frame_details

    summary = {
        'num_frames': num_valid_frames,
        'avg_total_track_ms': np.mean(total_track_times),
        'std_total_track_ms': np.std(total_track_times),
        'avg_preprocess_ms': np.mean(preprocess_times),
        'std_preprocess_ms': np.std(preprocess_times),
        'avg_inference_ms': np.mean(inference_times),
        'std_inference_ms': np.std(inference_times),
        'avg_postprocess_ms': np.mean(postprocess_times),
        'std_postprocess_ms': np.std(postprocess_times),
        'avg_association_plus_nms_ms': np.mean(association_times),
        'std_association_plus_nms_ms': np.std(association_times),
        'avg_candidates': np.mean(num_candidates_list),
        'avg_tracks': np.mean(num_tracks_list),
    }

    return summary, frame_details


def save_detailed_results(
    all_frame_data: List[Dict],
    summary_data: List[Dict],
    output_path: str
):
    """
    保存详细结果到CSV

    Args:
        all_frame_data: 所有帧的详细数据
        summary_data: 实验摘要数据
        output_path: 输出文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存每帧详细数据
        detail_path = output_path.replace('.csv', '_frames.csv')
        with open(detail_path, 'w', newline='', encoding='utf-8') as f:
            if all_frame_data:
                fieldnames = all_frame_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_frame_data)

        # 保存摘要数据
        summary_path = output_path.replace('.csv', '_summary.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            if summary_data:
                fieldnames = summary_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)

        # 同时保存合并的详细数据（包含实验标识）
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if all_frame_data:
                fieldnames = all_frame_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_frame_data)

        print(f"\n详细结果已保存到: {output_path}")
        print(f"摘要结果已保存到: {summary_path}")
        print(f"每帧数据已保存到: {detail_path}")

    except Exception as e:
        print(f"保存结果时出错: {e}")


def print_summary(summary_data: List[Dict]):
    """
    打印结果摘要

    Args:
        summary_data: 实验摘要数据列表
    """
    print("\n" + "=" * 100)
    print("📊 全量视频序列性能剖析结果摘要")
    print("=" * 100)

    for data in summary_data:
        exp = data['experiment']
        desc = data['description']
        print(f"\n实验 {exp} ({desc}):")
        print(f"  测试帧数:              {data['num_frames']}")
        print(f"  平均总跟踪时间:        {data['avg_total_track_ms']:.2f} ± {data['std_total_track_ms']:.2f} ms")
        print(f"  ├─ 预处理时间:         {data['avg_preprocess_ms']:.2f} ± {data['std_preprocess_ms']:.2f} ms")
        print(f"  ├─ 推理时间:           {data['avg_inference_ms']:.2f} ± {data['std_inference_ms']:.2f} ms")
        print(f"  └─ Association+NMP:    {data['avg_association_plus_nms_ms']:.2f} ± {data['std_association_plus_nms_ms']:.2f} ms")
        print(f"  平均候选框数量:        {data['avg_candidates']:.2f}")
        print(f"  平均跟踪目标数:        {data['avg_tracks']:.2f}")

    # 计算对比
    if len(summary_data) >= 2:
        print("\n" + "-" * 100)
        print("📈 关键对比 (A vs B):")
        baseline = next((d for d in summary_data if d['experiment'] == 'A'), None)
        cbam = next((d for d in summary_data if d['experiment'] == 'B'), None)

        if baseline and cbam:
            assoc_reduction = baseline['avg_association_plus_nms_ms'] - cbam['avg_association_plus_nms_ms']
            assoc_reduction_pct = (assoc_reduction / baseline['avg_association_plus_nms_ms']) * 100
            print(f"  Association+NMS 减少: {assoc_reduction:.2f} ms ({assoc_reduction_pct:.1f}%)")

    print("\n" + "=" * 100)


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
    args.data_root = resolve_path(args.data_root, current_dir)
    args.output = resolve_path(args.output, current_dir)

    print("=" * 100)
    print("🚀 YOLOv8 全量视频序列性能剖析")
    print("=" * 100)
    print(f"\n配置信息:")
    print(f"  实验A模型: {args.expA}")
    print(f"  实验B模型: {args.expB}")
    print(f"  实验C模型: {args.expC}")
    print(f"  实验D模型: {args.expD}")
    print(f"  目标序列: {args.sequence}")
    print(f"  数据根目录: {args.data_root}")
    print(f"  跟踪器: {args.tracker}")
    print(f"  设备: {args.device}")
    print(f"  预热帧数: {args.warmup}")
    print(f"  输出文件: {args.output}")

    # 获取序列图像
    image_list = get_sequence_images(args.data_root, args.sequence)
    if not image_list:
        print("错误: 无法获取序列图像")
        sys.exit(1)

    # 定义实验配置
    experiments = [
        ('A', args.expA, '基线 (PANet)'),
        ('B', args.expB, 'CBAM注意力'),
        ('C', args.expC, 'BiFPN'),
        ('D', args.expD, '联合 (CBAM+BiFPN)')
    ]

    # 存储结果
    all_summary = []
    all_frame_data = []

    # 依次测试每个实验
    for exp_name, model_path, exp_desc in experiments:
        print(f"\n{'-' * 100}")
        print(f"🔬 实验 {exp_name}: {exp_desc}")
        print(f"{'-' * 100}")

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
        summary, frame_details = profile_sequence_tracking(
            model, image_list, args.device,
            args.imgsz, args.conf, args.iou,
            args.tracker, args.warmup
        )

        if summary:
            # 添加实验信息
            summary['experiment'] = exp_name
            summary['description'] = exp_desc
            all_summary.append(summary)

            # 添加实验信息到每帧数据
            for frame_data in frame_details:
                frame_data['experiment'] = exp_name
                frame_data['description'] = exp_desc
            all_frame_data.extend(frame_details)

        # 释放GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印结果摘要
    if all_summary:
        print_summary(all_summary)

        # 保存结果
        save_detailed_results(all_frame_data, all_summary, args.output)
        print("\n✅ 全量视频序列性能剖析完成!")
    else:
        print("\n❌ 没有有效的测试结果")
        sys.exit(1)


if __name__ == '__main__':
    main()
