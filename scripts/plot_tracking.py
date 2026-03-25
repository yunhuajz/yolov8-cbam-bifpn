#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 跟踪结果可视化脚本
读取跟踪评估结果并生成对比图表
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore')

# ========== 添加项目根目录到 PYTHONPATH ==========
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_absolute_path(file_path: str) -> str:
    """获取绝对路径"""
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    abs_path = project_root / file_path
    if abs_path.exists():
        return str(abs_path)
    return str(Path.cwd() / file_path)


def setup_plot_style():
    """设置绘图样式和中文字体"""
    # 尝试使用seaborn风格
    try:
        plt.style.use('seaborn-v0_8-muted')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_tracking_data(csv_path: str) -> Optional[pd.DataFrame]:
    """加载跟踪评估结果"""
    try:
        abs_path = get_absolute_path(csv_path)
        if not Path(abs_path).exists():
            print(f"数据文件不存在: {abs_path}")
            return None

        df = pd.read_csv(abs_path)
        print(f"已加载数据: {len(df)} 组实验")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def plot_mota_comparison(df: pd.DataFrame, save_dir: Path):
    """
    图1: 四组实验的 MOTA 指标对比柱状图
    """
    print("[*] 正在绘制 MOTA 对比图...")

    # 实验名称映射
    name_map = {
        'expA_baseline': 'Baseline\n(YOLOv8n)',
        'expB_cbam': 'YOLOv8n\n+CBAM *',
        'expC_bifpn': 'YOLOv8n\n+BiFPN',
        'expD_combined': 'YOLOv8n\n+Combined'
    }

    df['DisplayName'] = df['Name'].map(lambda x: name_map.get(x, x))

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色设置（高亮expB）
    colors = ['#bdc3c7', '#e74c3c', '#e67e22', '#9b59b6']  # 灰、红(突出)、橙、紫

    bars = ax.bar(df['DisplayName'], df['MOTA'], color=colors, width=0.55, edgecolor='black', linewidth=1.2)

    # 设置标题和标签
    ax.set_title('不同改进方案的多目标跟踪精度(MOTA)对比', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylabel('MOTA (%)', fontsize=12)
    ax.set_xlabel('实验方案', fontsize=12)

    # 设置Y轴范围（根据数据自动调整）
    min_mota = df['MOTA'].min()
    max_mota = df['MOTA'].max()
    y_min = max(0, min_mota - 5)
    y_max = min(100, max_mota + 5)
    ax.set_ylim(y_min, y_max)

    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标注
    for bar, mota in zip(bars, df['MOTA']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mota:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加最佳标注
    best_idx = df['MOTA'].idxmax()
    best_exp = df.loc[best_idx, 'Name']
    if 'expB' in best_exp:
        ax.text(0.5, 0.95, 'CBAM方案性能最佳', transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # 保存图片
    save_path = save_dir / 'Tracking_MOTA_Comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")

    plt.close()


def plot_id_switches_comparison(df: pd.DataFrame, save_dir: Path):
    """
    图2: 四组实验的 ID Switches 次数对比图
    """
    print("[*] 正在绘制 ID Switches 对比图...")

    # 实验名称映射
    name_map = {
        'expA_baseline': 'Baseline\n(YOLOv8n)',
        'expB_cbam': 'YOLOv8n\n+CBAM *',
        'expC_bifpn': 'YOLOv8n\n+BiFPN',
        'expD_combined': 'YOLOv8n\n+Combined'
    }

    df['DisplayName'] = df['Name'].map(lambda x: name_map.get(x, x))

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色设置（ID Switches越少越好，所以最佳用绿色）
    colors = ['#e74c3c', '#27ae60', '#f39c12', '#9b59b6']  # 红、绿(好)、橙、紫

    bars = ax.bar(df['DisplayName'], df['ID_Switches'], color=colors, width=0.55, edgecolor='black', linewidth=1.2)

    # 设置标题和标签
    ax.set_title('不同改进方案的ID跳变次数对比（越少越好）', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylabel('ID Switches 次数', fontsize=12)
    ax.set_xlabel('实验方案', fontsize=12)

    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标注
    for bar, switches in zip(bars, df['ID_Switches']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(df['ID_Switches'])*0.02,
                f'{int(switches)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 找出ID Switches最少的实验
    best_idx = df['ID_Switches'].idxmin()
    best_exp = df.loc[best_idx, 'Name']
    best_switches = df.loc[best_idx, 'ID_Switches']

    # 添加说明
    ax.text(0.5, 0.95, f'{name_map.get(best_exp, best_exp)} 的ID跳变最少: {int(best_switches)}次',
            transform=ax.transAxes, ha='center', va='top', fontsize=11, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()

    # 保存图片
    save_path = save_dir / 'Tracking_ID_Switches_Comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")

    plt.close()


def plot_comprehensive_comparison(df: pd.DataFrame, save_dir: Path):
    """
    图3: 综合跟踪指标对比（MOTA + IDF1 + FPS）
    """
    print("[*] 正在绘制综合指标对比图...")

    # 实验名称映射（简化版）
    name_map = {
        'expA_baseline': 'Baseline',
        'expB_cbam': 'CBAM *',
        'expC_bifpn': 'BiFPN',
        'expD_combined': 'Combined'
    }

    df['ShortName'] = df['Name'].map(lambda x: name_map.get(x, x))

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#7f8c8d', '#e74c3c', '#f39c12', '#9b59b6']

    # MOTA
    axes[0].bar(df['ShortName'], df['MOTA'], color=colors, edgecolor='black', linewidth=1)
    axes[0].set_title('MOTA (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('百分比', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(df['MOTA']):
        axes[0].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

    # IDF1
    axes[1].bar(df['ShortName'], df['IDF1'], color=colors, edgecolor='black', linewidth=1)
    axes[1].set_title('IDF1 (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('百分比', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(df['IDF1']):
        axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

    # FPS
    axes[2].bar(df['ShortName'], df['FPS'], color=colors, edgecolor='black', linewidth=1)
    axes[2].set_title('处理速度 (FPS)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('帧/秒', fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(df['FPS']):
        axes[2].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('多目标跟踪综合性能对比', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存图片
    save_path = save_dir / 'Tracking_Comprehensive_Comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")

    plt.close()


def plot_mota_vs_idf1(df: pd.DataFrame, save_dir: Path):
    """
    图4: MOTA vs IDF1 散点图
    """
    print("[*] 正在绘制 MOTA-IDF1 关系图...")

    name_map = {
        'expA_baseline': 'Baseline',
        'expB_cbam': 'CBAM *',
        'expC_bifpn': 'BiFPN',
        'expD_combined': 'Combined'
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#7f8c8d', '#e74c3c', '#f39c12', '#9b59b6']
    sizes = [100, 200, 100, 100]  # CBAM更大

    for i, row in df.iterrows():
        ax.scatter(row['MOTA'], row['IDF1'], s=sizes[i], c=colors[i],
                  edgecolors='black', linewidth=2, alpha=0.7,
                  label=name_map.get(row['Name'], row['Name']))

        # 添加标签
        ax.annotate(name_map.get(row['Name'], row['Name']),
                   (row['MOTA'], row['IDF1']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('MOTA (%)', fontsize=12)
    ax.set_ylabel('IDF1 (%)', fontsize=12)
    ax.set_title('MOTA vs IDF1 性能分布', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    # 添加理想区域标注
    ax.axhline(y=df['IDF1'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=df['MOTA'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.text(0.95, 0.95, '理想区域\n(右上)', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    save_path = save_dir / 'Tracking_MOTA_vs_IDF1.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 跟踪结果可视化')
    parser.add_argument('--input', type=str, default='metrics/tracking_results.csv',
                       help='输入的跟踪结果CSV文件路径')
    parser.add_argument('--output', type=str, default='picture',
                       help='输出图片保存目录')
    args = parser.parse_args()

    print("="*60)
    print("YOLOv8 跟踪结果可视化")
    print("="*60)

    # 设置绘图样式
    setup_plot_style()

    # 加载数据
    df = load_tracking_data(args.input)
    if df is None:
        sys.exit(1)

    # 确保输出目录存在
    output_dir = Path(get_absolute_path(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 输出目录: {output_dir}")

    # 生成图表
    print("\n" + "-"*60)
    plot_mota_comparison(df, output_dir)
    plot_id_switches_comparison(df, output_dir)
    plot_comprehensive_comparison(df, output_dir)
    plot_mota_vs_idf1(df, output_dir)
    print("-"*60)

    print(f"\n所有可视化图表已保存到: {output_dir}")
    print("\n生成的图表:")
    print("  1. Tracking_MOTA_Comparison.png - MOTA指标对比")
    print("  2. Tracking_ID_Switches_Comparison.png - ID跳变对比")
    print("  3. Tracking_Comprehensive_Comparison.png - 综合指标对比")
    print("  4. Tracking_MOTA_vs_IDF1.png - MOTA-IDF1散点图")


if __name__ == '__main__':
    main()
