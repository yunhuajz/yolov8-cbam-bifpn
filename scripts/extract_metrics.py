#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_metrics.py - 实验指标提取脚本（改进版）

使用方法:
    # 自动查找实验目录（需要目录名以 expX_xxx 开头）
    python extract_metrics.py --experiment A
    python extract_metrics.py --experiment B
    python extract_metrics.py --experiment C
    python extract_metrics.py --experiment D

    # 手动指定实验目录（推荐，避免自动查找出错）
    python extract_metrics.py --experiment A --dir runs/train/expA_baseline_b80
    python extract_metrics.py --experiment B --dir runs/train/expB_cbam_b80
    ...

功能:
    从指定实验的 results.csv 中提取最后一轮的指标：
    - mAP50
    - mAP50-95
    - Precision (如果存在)
    - Recall (如果存在)
    并将结果追加到汇总文件 results.csv 中。
"""

import argparse
import os
import csv
import sys
from typing import Optional, Tuple, List

EXPERIMENT_MAP = {
    'A': {'name': 'expA_baseline', 'description': '基线 (PANet)'},
    'B': {'name': 'expB_cbam', 'description': 'CBAM注意力'},
    'C': {'name': 'expC_bifpn', 'description': 'BiFPN'},
    'D': {'name': 'expD_combined', 'description': '联合 (CBAM+BiFPN)'},
}

SUMMARY_FILE = 'results.csv'


def list_subdirs(parent_dir: str) -> List[str]:
    """列出 parent_dir 下的所有子目录"""
    if not os.path.isdir(parent_dir):
        return []
    return [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]


def find_experiment_dir(experiment: str) -> Optional[str]:
    """
    改进的目录查找：优先匹配以 exp_name 开头的目录，若失败则回退到包含 exp_name 的目录。
    若找到多个匹配，报错并提示手动指定。
    """
    exp_info = EXPERIMENT_MAP.get(experiment)
    if not exp_info:
        return None

    exp_name = exp_info['name']  # 例如 'expA_baseline'
    train_dir = os.path.join('runs', 'train')

    if not os.path.isdir(train_dir):
        print(f"  ❌ 目录不存在: {train_dir}")
        return None

    candidates = []
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if not os.path.isdir(item_path):
            continue
        results_file = os.path.join(item_path, 'results.csv')
        if not os.path.exists(results_file):
            continue

        # 优先匹配以 exp_name 开头的目录（忽略大小写）
        if item.lower().startswith(exp_name.lower()):
            candidates.append(item_path)
        # 次选匹配包含 exp_name 的目录（作为备选，但记录时标记）
        elif exp_name.lower() in item.lower():
            candidates.append(item_path)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"  ❌ 找到多个匹配目录，请使用 --dir 手动指定：")
        for c in candidates:
            print(f"     {c}")
        return None
    else:
        # 没找到任何匹配，列出当前所有子目录供参考
        all_dirs = list_subdirs(train_dir)
        if all_dirs:
            print(f"  ❌ 未找到匹配实验 {experiment} 的目录。当前 runs/train 下的子目录有：")
            for d in all_dirs:
                print(f"     {d}")
        else:
            print(f"  ❌ runs/train 下没有任何子目录。")
        return None


def read_last_epoch_metrics(results_file: str) -> Optional[dict]:
    """
    读取 results.csv 最后一行，返回包含以下键的字典（如果列存在）：
        - mAP50
        - mAP50-95
        - precision
        - recall
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip() for h in header]

            # 找出所需列的索引
            col_indices = {
                'mAP50': None,
                'mAP50-95': None,
                'precision': None,
                'recall': None,
            }
            for i, col in enumerate(header):
                if col == 'metrics/mAP50(B)':
                    col_indices['mAP50'] = i
                elif col == 'metrics/mAP50-95(B)':
                    col_indices['mAP50-95'] = i
                elif col == 'metrics/precision(B)':
                    col_indices['precision'] = i
                elif col == 'metrics/recall(B)':
                    col_indices['recall'] = i

            # 读取最后一行
            last_row = None
            for row in reader:
                if row:
                    last_row = row

            if last_row is None:
                print(f"  ❌ 文件为空: {results_file}")
                return None

            # 提取指标
            metrics = {}
            for key, idx in col_indices.items():
                if idx is not None and idx < len(last_row):
                    try:
                        metrics[key] = float(last_row[idx])
                    except ValueError:
                        metrics[key] = None
                else:
                    metrics[key] = None

            return metrics

    except Exception as e:
        print(f"  ❌ 读取文件出错: {e}")
        return None


def append_to_summary(experiment: str, metrics: dict) -> bool:
    """
    将指标追加到汇总文件，自动处理表头。
    metrics 字典应包含 'mAP50', 'mAP50-95', 'precision', 'recall' 等键（可能为 None）。
    """
    file_exists = os.path.exists(SUMMARY_FILE)
    try:
        with open(SUMMARY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(['Experiment', 'Description', 'mAP50', 'mAP50-95', 'Precision', 'Recall'])

            desc = EXPERIMENT_MAP[experiment]['description']
            row = [
                experiment,
                desc,
                f"{metrics.get('mAP50', 0):.5f}" if metrics.get('mAP50') is not None else '',
                f"{metrics.get('mAP50-95', 0):.5f}" if metrics.get('mAP50-95') is not None else '',
                f"{metrics.get('precision', 0):.5f}" if metrics.get('precision') is not None else '',
                f"{metrics.get('recall', 0):.5f}" if metrics.get('recall') is not None else '',
            ]
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"  ❌ 写入汇总文件出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='从实验结果中提取指标（改进版）')
    parser.add_argument('--experiment', '-e', required=True, choices=['A', 'B', 'C', 'D'],
                        help='实验标识 (A: 基线, B: CBAM, C: BiFPN, D: 联合)')
    parser.add_argument('--dir', '-d', help='手动指定实验目录路径（跳过自动查找）')
    args = parser.parse_args()
    exp = args.experiment.upper()

    print(f"\n{'='*50}")
    print(f"[INFO] 提取实验 {exp} ({EXPERIMENT_MAP[exp]['description']}) 指标")
    print(f"{'='*50}\n")

    # 确定实验目录
    if args.dir:
        exp_dir = args.dir
        print(f"[INFO] 使用手动指定的目录: {exp_dir}")
    else:
        exp_dir = find_experiment_dir(exp)
        if exp_dir is None:
            print("\n[ERROR] 自动查找失败，请使用 --dir 手动指定目录。")
            sys.exit(1)
        print(f"[OK] 自动找到实验目录: {exp_dir}")

    # 检查 results.csv
    results_file = os.path.join(exp_dir, 'results.csv')
    if not os.path.exists(results_file):
        print(f"[ERROR] 未找到 results.csv 文件: {results_file}")
        sys.exit(1)

    # 读取指标
    metrics = read_last_epoch_metrics(results_file)
    if metrics is None:
        sys.exit(1)

    # 显示提取结果
    print(f"\n  最后一轮指标:")
    print(f"    mAP50   : {metrics.get('mAP50', 'N/A')}")
    print(f"    mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
    if metrics.get('precision') is not None:
        print(f"    Precision: {metrics['precision']:.5f}")
    if metrics.get('recall') is not None:
        print(f"    Recall   : {metrics['recall']:.5f}")

    # 追加到汇总文件
    if append_to_summary(exp, metrics):
        print(f"\n[OK] 指标已追加到汇总文件: {SUMMARY_FILE}")
    else:
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"[OK] 实验 {exp} 提取完成！")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()