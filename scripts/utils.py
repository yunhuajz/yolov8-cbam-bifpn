#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 工具函数库
包含数据处理、可视化、评估等工具函数
支持中文路径，包含详细错误处理
"""

import os
import sys
import json
import yaml
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import pandas as pd
import time
import random
import torch
import warnings
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 忽略警告
warnings.filterwarnings('ignore')


def seed_everything(seed: int = 42) -> None:
    """
    设置所有随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置为: {seed}")


def time_sync() -> float:
    """
    同步时间，返回当前时间戳（秒）

    Returns:
        当前时间戳
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def check_file(file_path: str) -> bool:
    """
    检查文件是否存在且可读，支持中文路径和包含空格的路径

    Args:
        file_path: 文件路径

    Returns:
        文件是否存在且可读
    """
    try:
        # 首先尝试使用pathlib处理
        path = Path(file_path)
        if path.exists() and path.is_file():
            return True

        # 如果pathlib失败，尝试使用os.path处理（对包含空格的路径更友好）
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return True

        # 尝试使用绝对路径
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            return True

        return False
    except Exception as e:
        print(f"检查文件时出错 {file_path}: {e}")
        return False


def load_image(image_path: str, mode: str = 'RGB') -> Optional[np.ndarray]:
    """
    加载图像，支持中文路径

    Args:
        image_path: 图像路径
        mode: 图像模式，'RGB'或'BGR'

    Returns:
        图像数组，失败返回None
    """
    try:
        # 使用OpenCV加载，支持中文路径
        if not check_file(image_path):
            print(f"图像文件不存在: {image_path}")
            return None

        # 读取图像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print(f"无法加载图像: {image_path}")
            return None

        # 转换颜色空间
        if mode.upper() == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    except Exception as e:
        print(f"加载图像时出错 {image_path}: {e}")
        return None


def load_video(video_path: str) -> Optional[Tuple[cv2.VideoCapture, Dict]]:
    """
    加载视频文件，支持中文路径

    Args:
        video_path: 视频路径

    Returns:
        (视频捕获对象, 视频信息字典)，失败返回None
    """
    try:
        if not check_file(video_path):
            print(f"视频文件不存在: {video_path}")
            return None

        # 使用OpenCV打开视频
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0
        }

        print(f"视频加载成功: {video_path}")
        print(f"  尺寸: {width}x{height}, FPS: {fps:.2f}, 总帧数: {frame_count}")

        return cap, video_info

    except Exception as e:
        print(f"加载视频时出错 {video_path}: {e}")
        return None


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640),
                     normalize: bool = True) -> np.ndarray:
    """
    预处理图像帧

    Args:
        frame: 输入图像帧
        target_size: 目标尺寸 (width, height)
        normalize: 是否归一化到[0,1]

    Returns:
        预处理后的图像
    """
    try:
        if frame is None:
            raise ValueError("输入帧为空")

        # 调整尺寸
        if frame.shape[:2] != target_size[::-1]:  # OpenCV尺寸是(height, width)
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        # 归一化
        if normalize:
            frame = frame.astype(np.float32) / 255.0

        return frame

    except Exception as e:
        print(f"预处理帧时出错: {e}")
        return frame if frame is not None else np.zeros((*target_size[::-1], 3), dtype=np.float32)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    将边界框从(x_center, y_center, width, height)转换为(x1, y1, x2, y2)

    Args:
        x: 边界框数组，形状为(n, 4)或(4,)

    Returns:
        转换后的边界框
    """
    try:
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != 4:
            raise ValueError(f"输入形状应为(n, 4)，实际为{x.shape}")

        # 转换
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2

        return y.squeeze()

    except Exception as e:
        print(f"xywh2xyxy转换时出错: {e}")
        return x


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """
    将边界框从(x1, y1, x2, y2)转换为(x_center, y_center, width, height)

    Args:
        x: 边界框数组，形状为(n, 4)或(4,)

    Returns:
        转换后的边界框
    """
    try:
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != 4:
            raise ValueError(f"输入形状应为(n, 4)，实际为{x.shape}")

        # 转换
        y = np.zeros_like(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
        y[:, 2] = x[:, 2] - x[:, 0]        # width
        y[:, 3] = x[:, 3] - x[:, 1]        # height

        return y.squeeze()

    except Exception as e:
        print(f"xyxy2xywh转换时出错: {e}")
        return x


def bbox_iou(box1: np.ndarray, box2: np.ndarray, x1y1x2y2: bool = True) -> np.ndarray:
    """
    计算边界框IOU

    Args:
        box1: 边界框1，形状为(n, 4)
        box2: 边界框2，形状为(m, 4)
        x1y1x2y2: 是否为(x1, y1, x2, y2)格式

    Returns:
        IOU矩阵，形状为(n, m)
    """
    try:
        box1 = np.asarray(box1, dtype=np.float32)
        box2 = np.asarray(box2, dtype=np.float32)

        if not x1y1x2y2:
            # 转换为xyxy格式
            box1 = xywh2xyxy(box1)
            box2 = xywh2xyxy(box2)

        # 扩展维度以便广播计算
        box1 = box1[:, None, :]  # (n, 1, 4)
        box2 = box2[None, :, :]  # (1, m, 4)

        # 计算交集
        inter_x1 = np.maximum(box1[..., 0], box2[..., 0])
        inter_y1 = np.maximum(box1[..., 1], box2[..., 1])
        inter_x2 = np.minimum(box1[..., 2], box2[..., 2])
        inter_y2 = np.minimum(box1[..., 3], box2[..., 3])

        inter_w = np.maximum(inter_x2 - inter_x1, 0)
        inter_h = np.maximum(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h

        # 计算并集
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area

        # 计算IOU
        iou = inter_area / (union_area + 1e-7)

        return iou

    except Exception as e:
        print(f"计算IOU时出错: {e}")
        return np.zeros((len(box1), len(box2)))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    非极大值抑制

    Args:
        boxes: 边界框，形状为(n, 4)，格式为(x1, y1, x2, y2)
        scores: 置信度分数，形状为(n,)
        iou_threshold: IOU阈值

    Returns:
        保留的边界框索引列表
    """
    try:
        if len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        # 按分数降序排序
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            # 选择分数最高的框
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算与剩余框的IOU
            iou = bbox_iou(boxes[i:i+1], boxes[order[1:]], x1y1x2y2=True)[0]

            # 保留IOU低于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    except Exception as e:
        print(f"NMS时出错: {e}")
        return list(range(len(boxes)))


def draw_bboxes(image: np.ndarray, boxes: np.ndarray, labels: List[str] = None,
                scores: np.ndarray = None, colors: List[Tuple[int, int, int]] = None,
                line_width: int = 2, font_size: int = 12) -> np.ndarray:
    """
    在图像上绘制边界框

    Args:
        image: 输入图像 (BGR格式)
        boxes: 边界框，形状为(n, 4)，格式为(x1, y1, x2, y2)
        labels: 标签列表，长度为n
        scores: 置信度分数，形状为(n,)
        colors: 颜色列表，每个边界框的颜色
        line_width: 线宽
        font_size: 字体大小

    Returns:
        绘制后的图像
    """
    try:
        if image is None:
            raise ValueError("输入图像为空")

        img = image.copy()
        n = len(boxes)

        if n == 0:
            return img

        # 默认颜色
        if colors is None:
            # 使用不同颜色区分不同类别
            color_palette = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 紫色
                (0, 255, 255),  # 黄色
            ]
            colors = [color_palette[i % len(color_palette)] for i in range(n)]

        # 绘制边界框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors[i]

            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

            # 准备标签文本
            label_text = ""
            if labels is not None and i < len(labels):
                label_text = str(labels[i])

            if scores is not None and i < len(scores):
                if label_text:
                    label_text += f" {scores[i]:.2f}"
                else:
                    label_text = f"{scores[i]:.2f}"

            # 绘制标签背景和文本
            if label_text:
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # 绘制文本背景
                cv2.rectangle(img, (x1, y1 - text_height - 5),
                            (x1 + text_width, y1), color, -1)

                # 绘制文本
                cv2.putText(img, label_text, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    except Exception as e:
        print(f"绘制边界框时出错: {e}")
        return image


def draw_tracks(image: np.ndarray, tracks: List[Dict],
                track_history: Dict[int, List[Tuple[int, int]]] = None,
                max_history: int = 30) -> np.ndarray:
    """
    在图像上绘制跟踪结果

    Args:
        image: 输入图像 (BGR格式)
        tracks: 跟踪结果列表，每个元素包含:
            - 'track_id': 跟踪ID
            - 'bbox': 边界框 [x1, y1, x2, y2]
            - 'class_id': 类别ID
            - 'score': 置信度
        track_history: 轨迹历史字典，key为track_id，value为位置历史列表
        max_history: 最大历史轨迹长度

    Returns:
        绘制后的图像
    """
    try:
        if image is None:
            raise ValueError("输入图像为空")

        img = image.copy()

        if not tracks:
            return img

        # 为每个track_id分配颜色
        track_colors = {}
        for track in tracks:
            track_id = track.get('track_id', 0)
            if track_id not in track_colors:
                # 根据track_id生成颜色
                hue = (track_id * 50) % 180  # 避免红色（0°）和相近颜色
                track_colors[track_id] = tuple(map(int, cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))

        # 绘制轨迹历史
        if track_history:
            for track_id, history in track_history.items():
                if track_id in track_colors and len(history) > 1:
                    color = track_colors[track_id]
                    # 只绘制最近的历史
                    recent_history = history[-max_history:]
                    # 绘制轨迹线
                    for j in range(1, len(recent_history)):
                        pt1 = tuple(map(int, recent_history[j-1]))
                        pt2 = tuple(map(int, recent_history[j]))
                        cv2.line(img, pt1, pt2, color, 2)

        # 绘制当前跟踪框
        for track in tracks:
            track_id = track.get('track_id', 0)
            bbox = track.get('bbox', [0, 0, 0, 0])
            class_id = track.get('class_id', 0)
            score = track.get('score', 0.0)

            if track_id in track_colors:
                color = track_colors[track_id]
            else:
                color = (0, 255, 0)  # 默认绿色

            x1, y1, x2, y2 = map(int, bbox[:4])

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制跟踪ID和类别
            label = f"ID:{track_id}"
            if 'class_name' in track:
                label = f"{track['class_name']} {label}"

            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(img, (x1, y1 - text_height - 5),
                        (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(img, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 更新轨迹历史
            if track_history is not None:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(center)
                # 限制历史长度
                if len(track_history[track_id]) > max_history:
                    track_history[track_id] = track_history[track_id][-max_history:]

        return img

    except Exception as e:
        print(f"绘制跟踪结果时出错: {e}")
        return image


def plot_detections(image: np.ndarray, detections: List[Dict],
                   save_path: str = None, show: bool = False) -> None:
    """
    绘制检测结果并保存或显示

    Args:
        image: 输入图像
        detections: 检测结果列表
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图像
    """
    try:
        if image is None:
            print("输入图像为空")
            return

        # 提取边界框和标签
        boxes = []
        labels = []
        scores = []

        for det in detections:
            if 'bbox' in det:
                boxes.append(det['bbox'])
            if 'class_name' in det:
                labels.append(det['class_name'])
            elif 'class_id' in det:
                labels.append(f"class_{det['class_id']}")
            if 'score' in det:
                scores.append(det['score'])

        # 绘制边界框
        result_img = draw_bboxes(image, np.array(boxes), labels,
                                np.array(scores) if scores else None)

        # 保存图像
        if save_path:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图像，支持中文路径
                cv2.imencode('.jpg', result_img)[1].tofile(save_path)
                print(f"检测结果已保存到: {save_path}")
            except Exception as e:
                print(f"保存图像时出错 {save_path}: {e}")

        # 显示图像
        if show:
            cv2.imshow('Detections', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"绘制检测结果时出错: {e}")


def plot_loss_curves(log_dir: str, save_path: str = None, show: bool = False) -> None:
    """
    绘制训练损失曲线

    Args:
        log_dir: 日志目录，包含results.csv文件
        save_path: 保存路径，如果为None则不保存
        show: 是否显示图像
    """
    try:
        results_file = os.path.join(log_dir, 'results.csv')

        if not os.path.exists(results_file):
            print(f"结果文件不存在: {results_file}")
            return

        # 读取结果
        df = pd.read_csv(results_file)

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # 绘制损失曲线
        loss_columns = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                       'val/box_loss', 'val/cls_loss', 'metrics/mAP50(B)']

        titles = ['训练边界框损失', '训练分类损失', '训练DFL损失',
                 '验证边界框损失', '验证分类损失', 'mAP50指标']

        for i, (col, title) in enumerate(zip(loss_columns, titles)):
            if col in df.columns:
                axes[i].plot(df[col], label=col, linewidth=2)
                axes[i].set_title(title, fontsize=12)
                axes[i].set_xlabel('Epoch', fontsize=10)
                axes[i].set_ylabel('Value', fontsize=10)
                axes[i].legend(fontsize=9)
                axes[i].grid(True, alpha=0.3)

        plt.suptitle('训练损失曲线', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"损失曲线已保存到: {save_path}")
            except Exception as e:
                print(f"保存损失曲线时出错 {save_path}: {e}")

        # 显示图像
        if show:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"绘制损失曲线时出错: {e}")


def plot_metrics(metrics: Dict[str, List[float]], save_path: str = None,
                show: bool = False) -> None:
    """
    绘制评估指标曲线

    Args:
        metrics: 指标字典，key为指标名称，value为指标值列表
        save_path: 保存路径
        show: 是否显示
    """
    try:
        if not metrics:
            print("指标数据为空")
            return

        # 创建图形
        fig, axes = plt.subplots(1, min(3, len(metrics)), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]

        for i, (metric_name, values) in enumerate(list(metrics.items())[:3]):
            ax = axes[i] if len(metrics) > 1 else axes
            ax.plot(values, label=metric_name, linewidth=2, marker='o', markersize=4)
            ax.set_title(metric_name, fontsize=12)
            ax.set_xlabel('Epoch/Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('评估指标曲线', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"指标曲线已保存到: {save_path}")
            except Exception as e:
                print(f"保存指标曲线时出错 {save_path}: {e}")

        # 显示图像
        if show:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"绘制指标曲线时出错: {e}")


def calculate_map(predictions: List[Dict], ground_truths: List[Dict],
                 iou_threshold: float = 0.5) -> float:
    """
    计算mAP（平均精度）
    简化版本，实际应用建议使用pycocotools

    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        iou_threshold: IOU阈值

    Returns:
        mAP值
    """
    try:
        if not predictions or not ground_truths:
            return 0.0

        # 按类别分组
        pred_by_class = {}
        gt_by_class = {}

        for pred in predictions:
            class_id = pred.get('class_id', 0)
            if class_id not in pred_by_class:
                pred_by_class[class_id] = []
            pred_by_class[class_id].append(pred)

        for gt in ground_truths:
            class_id = gt.get('class_id', 0)
            if class_id not in gt_by_class:
                gt_by_class[class_id] = []
            gt_by_class[class_id].append(gt)

        # 计算每个类别的AP
        ap_scores = []

        for class_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
            class_preds = pred_by_class.get(class_id, [])
            class_gts = gt_by_class.get(class_id, [])

            if not class_gts:
                # 没有真实标注，AP为0
                ap_scores.append(0.0)
                continue

            # 按置信度排序预测
            class_preds.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            # 匹配预测和真实标注
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_matched = [False] * len(class_gts)

            for i, pred in enumerate(class_preds):
                pred_bbox = pred.get('bbox', [0, 0, 0, 0])
                best_iou = 0.0
                best_gt_idx = -1

                # 找到最佳匹配的真实标注
                for j, gt in enumerate(class_gts):
                    if gt_matched[j]:
                        continue

                    gt_bbox = gt.get('bbox', [0, 0, 0, 0])
                    iou = bbox_iou(np.array([pred_bbox]), np.array([gt_bbox]))[0, 0]

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                # 判断是否为真正例
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1

            # 计算精度和召回率
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(class_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

            # 计算AP（使用11点插值法简化版）
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                mask = recalls >= t
                if np.any(mask):
                    ap += np.max(precisions[mask])

            ap /= 11
            ap_scores.append(ap)

        # 计算mAP
        map_score = np.mean(ap_scores) if ap_scores else 0.0

        return map_score

    except Exception as e:
        print(f"计算mAP时出错: {e}")
        return 0.0


def save_results_to_csv(results: List[Dict], save_path: str) -> None:
    """
    保存结果到CSV文件

    Args:
        results: 结果列表
        save_path: 保存路径
    """
    try:
        if not results:
            print("结果数据为空")
            return

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存CSV
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到CSV: {save_path}")

    except Exception as e:
        print(f"保存结果到CSV时出错 {save_path}: {e}")


def load_results_from_csv(csv_path: str) -> List[Dict]:
    """
    从CSV文件加载结果

    Args:
        csv_path: CSV文件路径

    Returns:
        结果列表
    """
    try:
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在: {csv_path}")
            return []

        # 读取CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 转换为字典列表
        results = df.to_dict('records')

        print(f"从CSV加载了 {len(results)} 条结果: {csv_path}")
        return results

    except Exception as e:
        print(f"从CSV加载结果时出错 {csv_path}: {e}")
        return []


if __name__ == '__main__':
    # 测试工具函数
    print("=" * 50)
    print("工具函数库测试")
    print("=" * 50)

    # 测试随机种子设置
    seed_everything(42)

    # 测试时间同步
    t1 = time_sync()
    time.sleep(0.1)
    t2 = time_sync()
    print(f"时间同步测试: {t2 - t1:.3f}秒")

    # 测试文件检查
    test_file = __file__
    print(f"文件检查测试: {test_file} -> {check_file(test_file)}")

    # 测试边界框转换
    test_box = [100, 100, 50, 30]  # x_center, y_center, width, height
    xyxy_box = xywh2xyxy(test_box)
    xywh_box = xyxy2xywh(xyxy_box)
    print(f"边界框转换测试: {test_box} -> {xyxy_box} -> {xywh_box}")

    # 测试IOU计算
    box1 = np.array([[0, 0, 10, 10]])
    box2 = np.array([[5, 5, 15, 15]])
    iou = bbox_iou(box1, box2)
    print(f"IOU计算测试: {iou[0, 0]:.3f}")

    # 测试NMS
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    print(f"NMS测试: 保留索引 {keep}")

    print("=" * 50)
    print("工具函数库测试完成!")
    print("=" * 50)