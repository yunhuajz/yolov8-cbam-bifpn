#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 训练脚本 - 极简稳健性能优化版
针对 4090D 高性能显卡进行 IO 优化
"""

# ========== 显存碎片管理（必须在最开头）==========
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# ========== 添加项目根目录到 PYTHONPATH ==========
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========== PyTorch 与 cuDNN 设置 ==========
import torch
torch.backends.cudnn.benchmark = False  # 防止 cuDNN 初始化报错

# ========== 导入并注册自定义模块 ==========
from ultralytics import YOLO
from ultralytics.nn.modules import CBAM, BiFPN
import ultralytics.nn.tasks as tasks

# 注入自定义模块到 tasks 命名空间
tasks.CBAM = CBAM
tasks.BiFPN = BiFPN


def parse_args():
    """极简参数解析"""
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--config', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--name', type=str, required=True, help='实验名称')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    return parser.parse_args()


def main():
    args = parse_args()

    # 路径处理（支持空格）
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    # 数据集配置路径（固定）
    data_config = os.path.join(project_root, 'configs', 'UA-DETRAC.yaml')

    print("=" * 60)
    print("YOLOv8 训练启动 (性能优化版)")
    print("=" * 60)
    print(f"配置: {config_path}")
    print(f"实验: {args.name}")
    print(f"批次: {args.batch}")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ========== 模型初始化：结构 + 权重 ==========
    print("\n[1/3] 构建模型结构...")
    model = YOLO(config_path)
    print("   模型结构加载完成")

    print("\n[2/3] 加载预训练权重...")
    try:
        model.load('yolov8n.pt')
        print("   权重加载完成（仅匹配层）")
    except Exception as e:
        print(f"   警告: 权重加载失败 ({e})，使用随机初始化")

    # 打印模型信息
    print("\n[3/3] 模型结构确认:")
    model.info()

    # ========== 训练参数优化 ==========
    train_args = {
        'data': data_config,
        'epochs': 100,
        'batch': args.batch,
        'imgsz': 640,
        'workers': 8,      # 提升帮厨数量，解决 GPU 饥饿
        'device': '0',
        'amp': True,       # 自动混合精度
        'project': 'runs/train',
        'name': args.name,
        'exist_ok': True,
        'patience': 50,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'verbose': False,
        'cache': True,     # 【核心优化】开启内存缓存，彻底消除 IO 瓶颈
    }

    print("\n" + "=" * 60)
    print("开始训练 (IO 缓存已开启)")
    print("=" * 60)

    try:
        results = model.train(**train_args)
        print("\n训练完成!")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()