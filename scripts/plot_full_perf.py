#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 全量序列性能图表生成脚本
================================

读取性能数据并生成可视化图表:
1. 堆叠柱状图: 展示 A, B, C, D 的总耗时拆解
2. 趋势图: 对比 A 和 B 的关联耗时随帧数变化

Usage:
    python scripts/plot_full_perf.py
    python scripts/plot_full_perf.py --input metrics/full_sequence_perf_details.csv

Output:
    picture/full_sequence_stacked_bar.png - 堆叠柱状图
    picture/association_trend_comparison.png - 关联耗时趋势图
"""

import os
import sys
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 全量序列性能图表生成',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str,
                       default='metrics/full_sequence_perf_details_frames.csv',
                       help='输入CSV文件路径（每帧详细数据）')
    parser.add_argument('--summary', type=str,
                       default='metrics/full_sequence_perf_details_summary.csv',
                       help='输入摘要CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='picture',
                       help='输出图表目录')

    return parser.parse_args()


def load_data(input_path: str, summary_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载性能数据

    Args:
        input_path: 详细数据CSV路径
        summary_path: 摘要数据CSV路径

    Returns:
        (详细数据DataFrame, 摘要数据DataFrame)
    """
    try:
        # 尝试加载详细数据
        if os.path.exists(input_path):
            df_frames = pd.read_csv(input_path)
            print(f"已加载详细数据: {len(df_frames)} 条记录")
        else:
            # 尝试加载主输出文件
            alt_path = input_path.replace('_frames.csv', '.csv')
            if os.path.exists(alt_path):
                df_frames = pd.read_csv(alt_path)
                print(f"已加载详细数据: {len(df_frames)} 条记录")
            else:
                print(f"错误: 找不到数据文件: {input_path}")
                return None, None

        # 加载摘要数据
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            print(f"已加载摘要数据: {len(df_summary)} 个实验")
        else:
            df_summary = None

        return df_frames, df_summary

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None


def plot_stacked_bar(df_summary: pd.DataFrame, output_dir: str):
    """
    生成堆叠柱状图: 展示各实验的总耗时拆解

    Args:
        df_summary: 摘要数据DataFrame
        output_dir: 输出目录
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        # 准备数据
        experiments = df_summary['experiment'].tolist()
        descriptions = df_summary['description'].tolist()

        # 数据 (单位: ms)
        inference_times = df_summary['avg_inference_ms'].values
        association_times = df_summary['avg_association_plus_nms_ms'].values

        # 创建标签
        labels = [f"Exp {exp}\n({desc})" for exp, desc in zip(experiments, descriptions)]

        # 绘制堆叠柱状图
        x = np.arange(len(experiments))
        width = 0.6

        # 底部: Inference (GPU)
        bars1 = ax.bar(x, inference_times, width, label='Inference (GPU)', color='#3498db', edgecolor='black')

        # 顶部: Association + NMS (CPU)
        bars2 = ax.bar(x, association_times, width, bottom=inference_times,
                      label='Post-process & Association (CPU)', color='#e74c3c', edgecolor='black')

        # 添加总时间标签
        for i, (inf, assoc) in enumerate(zip(inference_times, association_times)):
            total = inf + assoc
            ax.text(i, total + 1, f'{total:.1f} ms', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

        # 设置图表属性
        ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
        ax.set_title('YOLOv8 Full Sequence Performance Breakdown\n(Average Time per Frame)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加注释说明
        ax.text(0.02, 0.98, 'Lower is better', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', color='green',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图表
        output_path = os.path.join(output_dir, 'full_sequence_stacked_bar.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"堆叠柱状图已保存: {output_path}")

        plt.close()

    except Exception as e:
        print(f"生成堆叠柱状图时出错: {e}")
        import traceback
        traceback.print_exc()


def plot_association_trend(df_frames: pd.DataFrame, output_dir: str):
    """
    生成趋势图: 对比 A 和 B 的关联耗时随帧数变化

    Args:
        df_frames: 每帧详细数据DataFrame
        output_dir: 输出目录
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 过滤实验 A 和 B 的数据
        df_a = df_frames[df_frames['experiment'] == 'A'].sort_values('frame_idx')
        df_b = df_frames[df_frames['experiment'] == 'B'].sort_values('frame_idx')

        if len(df_a) == 0 or len(df_b) == 0:
            print("警告: 缺少实验 A 或 B 的数据")
            return

        # 图1: Association + NMS 耗时趋势
        ax1.plot(df_a['frame_idx'], df_a['association_plus_nms_ms'],
                label='Exp A (Baseline PANet)', color='#3498db',
                linewidth=1.5, alpha=0.8)
        ax1.plot(df_b['frame_idx'], df_b['association_plus_nms_ms'],
                label='Exp B (CBAM)', color='#e74c3c',
                linewidth=1.5, alpha=0.8)

        # 添加移动平均线
        window = 10
        if len(df_a) >= window:
            ax1.plot(df_a['frame_idx'], df_a['association_plus_nms_ms'].rolling(window=window).mean(),
                    '--', color='#2980b9', linewidth=2, alpha=0.9,
                    label=f'Exp A ({window}-frame MA)')
        if len(df_b) >= window:
            ax1.plot(df_b['frame_idx'], df_b['association_plus_nms_ms'].rolling(window=window).mean(),
                    '--', color='#c0392b', linewidth=2, alpha=0.9,
                    label=f'Exp B ({window}-frame MA)')

        ax1.set_ylabel('Association + NMS Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Association + NMS Time Trend Comparison\n(Exp A vs Exp B)',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 添加注释说明抖动情况
        std_a = df_a['association_plus_nms_ms'].std()
        std_b = df_b['association_plus_nms_ms'].std()
        ax1.text(0.02, 0.95, f'Exp A Std: {std_a:.2f} ms\nExp B Std: {std_b:.2f} ms',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 图2: 每帧跟踪目标数（表示车辆密度）
        ax2.plot(df_a['frame_idx'], df_a['num_tracks'],
                label='Exp A (Tracked Objects)', color='#3498db',
                linewidth=1.5, alpha=0.8)
        ax2.plot(df_b['frame_idx'], df_b['num_tracks'],
                label='Exp B (Tracked Objects)', color='#e74c3c',
                linewidth=1.5, alpha=0.8)

        ax2.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Tracked Objects', fontsize=12, fontweight='bold')
        ax2.set_title('Tracked Objects per Frame (Vehicle Density)',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 添加注释
        max_tracks_a = df_a['num_tracks'].max()
        max_tracks_b = df_b['num_tracks'].max()
        ax2.text(0.02, 0.95, f'Max Tracks A: {max_tracks_a}\nMax Tracks B: {max_tracks_b}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        # 保存图表
        output_path = os.path.join(output_dir, 'association_trend_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"趋势对比图已保存: {output_path}")

        plt.close()

    except Exception as e:
        print(f"生成趋势图时出错: {e}")
        import traceback
        traceback.print_exc()


def plot_all_experiments_trend(df_frames: pd.DataFrame, output_dir: str):
    """
    生成四实验关联耗时对比趋势图

    Args:
        df_frames: 每帧详细数据DataFrame
        output_dir: 输出目录
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = {'A': '#3498db', 'B': '#e74c3c', 'C': '#2ecc71', 'D': '#9b59b6'}
        labels = {
            'A': 'Exp A (Baseline)',
            'B': 'Exp B (CBAM)',
            'C': 'Exp C (BiFPN)',
            'D': 'Exp D (Combined)'
        }

        for exp in ['A', 'B', 'C', 'D']:
            df_exp = df_frames[df_frames['experiment'] == exp].sort_values('frame_idx')
            if len(df_exp) > 0:
                # 绘制移动平均线以平滑显示
                window = 10
                if len(df_exp) >= window:
                    ma = df_exp['association_plus_nms_ms'].rolling(window=window, min_periods=1).mean()
                    ax.plot(df_exp['frame_idx'], ma,
                           label=labels.get(exp, f'Exp {exp}'),
                           color=colors.get(exp, 'gray'),
                           linewidth=2, alpha=0.9)

        ax.set_xlabel('Frame Index', fontsize=13, fontweight='bold')
        ax.set_ylabel('Association + NMS Time (ms)', fontsize=13, fontweight='bold')
        ax.set_title('Association + NMS Time Comparison (All Experiments)\n(10-frame Moving Average)',
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 保存图表
        output_path = os.path.join(output_dir, 'all_experiments_trend.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"四实验趋势图已保存: {output_path}")

        plt.close()

    except Exception as e:
        print(f"生成四实验趋势图时出错: {e}")
        import traceback
        traceback.print_exc()


def print_statistics(df_frames: pd.DataFrame, df_summary: pd.DataFrame):
    """
    打印统计信息

    Args:
        df_frames: 每帧详细数据DataFrame
        df_summary: 摘要数据DataFrame
    """
    print("\n" + "=" * 80)
    print("性能统计摘要")
    print("=" * 80)

    if df_summary is not None:
        print("\n各实验平均耗时:")
        for _, row in df_summary.iterrows():
            print(f"  {row['experiment']} ({row['description']}):")
            print(f"    - Inference: {row['avg_inference_ms']:.2f} ms")
            print(f"    - Association+NMS: {row['avg_association_plus_nms_ms']:.2f} ms")
            print(f"    - Total: {row['avg_total_track_ms']:.2f} ms")

    # 计算A vs B的对比
    df_a = df_frames[df_frames['experiment'] == 'A']
    df_b = df_frames[df_frames['experiment'] == 'B']

    if len(df_a) > 0 and len(df_b) > 0:
        print("\nExp A vs Exp B 对比:")

        assoc_a = df_a['association_plus_nms_ms'].mean()
        assoc_b = df_b['association_plus_nms_ms'].mean()
        reduction = assoc_a - assoc_b
        reduction_pct = (reduction / assoc_a) * 100 if assoc_a > 0 else 0

        print(f"  Association+NMS 减少: {reduction:.2f} ms ({reduction_pct:.1f}%)")

        std_a = df_a['association_plus_nms_ms'].std()
        std_b = df_b['association_plus_nms_ms'].std()
        stability_improvement = ((std_a - std_b) / std_a) * 100 if std_a > 0 else 0

        print(f"  抖动改善: {stability_improvement:.1f}% (Std: A={std_a:.2f}, B={std_b:.2f})")

        # 高车辆密度场景分析
        high_density_threshold = df_a['num_tracks'].quantile(0.8)
        df_a_high = df_a[df_a['num_tracks'] >= high_density_threshold]
        df_b_high = df_b[df_b['num_tracks'] >= high_density_threshold]

        if len(df_a_high) > 0 and len(df_b_high) > 0:
            assoc_a_high = df_a_high['association_plus_nms_ms'].mean()
            assoc_b_high = df_b_high['association_plus_nms_ms'].mean()
            print(f"\n  高车辆密度场景 (≥{high_density_threshold:.0f} objects):")
            print(f"    Exp A: {assoc_a_high:.2f} ms")
            print(f"    Exp B: {assoc_b_high:.2f} ms")
            print(f"    改善: {((assoc_a_high - assoc_b_high) / assoc_a_high * 100):.1f}%")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    args = parse_args()

    # Set stdout encoding for Windows
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 80)
    print("YOLOv8 全量序列性能图表生成")
    print("=" * 80)

    # 加载数据
    df_frames, df_summary = load_data(args.input, args.summary)
    if df_frames is None:
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成图表
    print("\n生成图表...")

    if df_summary is not None:
        plot_stacked_bar(df_summary, args.output_dir)

    plot_association_trend(df_frames, args.output_dir)
    plot_all_experiments_trend(df_frames, args.output_dir)

    # 打印统计
    print_statistics(df_frames, df_summary)

    print(f"\n所有图表已保存到: {args.output_dir}/")


if __name__ == '__main__':
    from typing import Tuple
    main()
