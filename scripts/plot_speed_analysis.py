#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 速度分析可视化脚本
生成堆叠柱状图展示Inference和Post-process时间对比

Usage:
    python scripts/plot_speed_analysis.py
    python scripts/plot_speed_analysis.py --input metrics/detailed_speed_analysis.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"导入依赖库失败: {e}")
    print("请安装所需依赖: pip install pandas matplotlib numpy")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 速度分析可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, default='metrics/detailed_speed_analysis.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='picture/Speed_Breakdown_Comparison.png',
                       help='输出图像路径')
    parser.add_argument('--dpi', type=int, default=300,
                       help='输出图像DPI')
    parser.add_argument('--show', action='store_true',
                       help='是否显示图像')

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


def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    从CSV文件加载数据

    Args:
        csv_path: CSV文件路径

    Returns:
        DataFrame或None
    """
    try:
        if not os.path.exists(csv_path):
            print(f"错误: 文件不存在: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        print(f"已加载数据: {csv_path}")
        print(f"  行数: {len(df)}")
        print(f"  列: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def plot_speed_breakdown(df: pd.DataFrame, output_path: str, dpi: int = 300, show: bool = False):
    """
    绘制速度分解堆叠柱状图

    Args:
        df: 包含性能数据的DataFrame
        output_path: 输出图像路径
        dpi: 图像DPI
        show: 是否显示图像
    """
    try:
        # 检查必要的列
        required_cols = ['experiment', 'avg_inference_ms', 'avg_postprocess_ms']
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 缺少必要的列: {col}")
                return

        # 按A/B/C/D顺序排序
        exp_order = ['A', 'B', 'C', 'D']
        df['sort_key'] = df['experiment'].apply(lambda x: exp_order.index(x) if x in exp_order else 999)
        df = df.sort_values('sort_key')

        # 准备数据
        experiments = df['experiment'].tolist()
        inference_times = df['avg_inference_ms'].tolist()
        postprocess_times = df['avg_postprocess_ms'].tolist()

        # 获取描述信息（如果有）
        descriptions = []
        if 'description' in df.columns:
            descriptions = df['description'].tolist()
        else:
            # 默认描述
            desc_map = {'A': 'Baseline', 'B': 'CBAM', 'C': 'BiFPN', 'D': 'Combined'}
            descriptions = [desc_map.get(e, e) for e in experiments]

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 设置柱状图位置
        x = np.arange(len(experiments))
        width = 0.6

        # 定义颜色
        color_inference = '#4472C4'  # 蓝色
        color_postprocess = '#ED7D31'  # 橙色

        # 绘制堆叠柱状图
        bars1 = ax.bar(x, inference_times, width, label='Inference', color=color_inference, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, postprocess_times, width, bottom=inference_times, label='Post-process', color=color_postprocess, edgecolor='black', linewidth=0.5)

        # 在柱子上添加数值标签
        for i, (inf_time, post_time) in enumerate(zip(inference_times, postprocess_times)):
            # Inference时间标签
            ax.text(i, inf_time / 2, f'{inf_time:.1f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')

            # Post-process时间标签
            ax.text(i, inf_time + post_time / 2, f'{post_time:.1f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')

            # 总时间标签（在柱子顶部）
            total_time = inf_time + post_time
            ax.text(i, total_time + 0.5, f'{total_time:.1f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='black')

        # 设置标签和标题
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Speed Breakdown Comparison\n(Inference + Post-process Time)', fontsize=14, fontweight='bold')

        # 设置X轴标签（实验名称+描述）
        x_labels = [f'{exp}\n({desc})' for exp, desc in zip(experiments, descriptions)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)

        # 添加图例
        ax.legend(loc='upper left', fontsize=10)

        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # 设置Y轴范围（留出一些空间显示标签）
        max_total = max([inf + post for inf, post in zip(inference_times, postprocess_times)])
        ax.set_ylim(0, max_total * 1.15)

        # 添加注释
        if 'avg_boxes' in df.columns:
            boxes_text = '\n'.join([f'{exp}: {boxes:.1f} boxes' for exp, boxes in zip(experiments, df['avg_boxes'])])
            ax.text(0.98, 0.98, f'Avg Detections:\n{boxes_text}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存图像
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"图像已保存到: {output_path}")

        # 显示图像
        if show:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"绘制图表时出错: {e}")
        import traceback
        traceback.print_exc()


def print_analysis(df: pd.DataFrame):
    """
    打印速度分析结果

    Args:
        df: 包含性能数据的DataFrame
    """
    print("\n" + "=" * 80)
    print("速度分析")
    print("=" * 80)

    # 按A/B/C/D顺序
    exp_order = ['A', 'B', 'C', 'D']

    for exp in exp_order:
        row = df[df['experiment'] == exp]
        if row.empty:
            continue

        row = row.iloc[0]
        inf_time = row['avg_inference_ms']
        post_time = row['avg_postprocess_ms']
        total_time = inf_time + post_time

        print(f"\n实验 {exp}:")
        print(f"  Inference:    {inf_time:.2f} ms ({inf_time/total_time*100:.1f}%)")
        print(f"  Post-process: {post_time:.2f} ms ({post_time/total_time*100:.1f}%)")
        print(f"  Total:        {total_time:.2f} ms")

    # 找出最快和最慢的
    df['total'] = df['avg_inference_ms'] + df['avg_postprocess_ms']
    fastest = df.loc[df['total'].idxmin()]
    slowest = df.loc[df['total'].idxmax()]

    print("\n最快: 实验 {} ({:.2f} ms)".format(fastest['experiment'], fastest['total']))
    print("最慢: 实验 {} ({:.2f} ms)".format(slowest['experiment'], slowest['total']))

    # 分析B组的特点（CBAM注意力）
    row_b = df[df['experiment'] == 'B']
    if not row_b.empty:
        row_b = row_b.iloc[0]
        print("\n实验B (CBAM注意力) 特点分析:")
        print("  - Inference时间较高: {:.2f} ms".format(row_b['avg_inference_ms']))
        print("  - Post-process时间较低: {:.2f} ms".format(row_b['avg_postprocess_ms']))
        print("  - 可能原因: CBAM注意力机制增加了前向计算开销，")
        print("              但特征质量提升减少了NMS后的候选框数量")

    print("=" * 80)


def main():
    """主函数"""
    args = parse_args()

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 解析路径
    args.input = resolve_path(args.input, current_dir)
    args.output = resolve_path(args.output, current_dir)

    print("=" * 80)
    print("YOLOv8 速度分析可视化")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  DPI: {args.dpi}")

    # 加载数据
    df = load_data(args.input)
    if df is None:
        sys.exit(1)

    # 打印分析
    print_analysis(df)

    # 绘制图表
    plot_speed_breakdown(df, args.output, args.dpi, args.show)

    print("\n可视化完成!")


if __name__ == '__main__':
    main()
