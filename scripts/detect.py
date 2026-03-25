#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 检测脚本
支持图像、视频、摄像头和图像文件夹检测
包含详细的结果统计和可视化
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import torch
except ImportError as e:
    print(f"导入依赖库失败: {e}")
    print("请安装所需依赖: pip install ultralytics opencv-python torch")
    sys.exit(1)

# 导入工具函数
try:
    from scripts.utils import (
        load_image, load_video, preprocess_frame,
        draw_bboxes, plot_detections, save_results_to_csv,
        check_file, time_sync
    )
except ImportError:
    print("警告: 无法导入工具函数，使用简化版本")
    # 定义简化版本的工具函数
    def load_image(image_path, mode='RGB'):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法加载图像: {image_path}")
                return None
            if mode.upper() == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None

    def load_video(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, {}
            return cap, {}
        except Exception as e:
            print(f"加载视频时出错: {e}")
            return None, {}

    def preprocess_frame(frame, target_size=(640, 640), normalize=True):
        if frame is None:
            return None
        frame = cv2.resize(frame, target_size)
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def draw_bboxes(image, boxes, labels=None, scores=None, colors=None, line_width=2, font_size=12):
        if image is None:
            return None
        img = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors[i] if colors else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        return img

    def plot_detections(image, detections, save_path=None, show=False):
        if save_path:
            cv2.imwrite(save_path, image)
        if show:
            cv2.imshow('Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_results_to_csv(results, save_path):
        import pandas as pd
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(results).to_csv(save_path, index=False)

    def check_file(file_path):
        return os.path.exists(file_path)

    def time_sync():
        return time.time()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 检测脚本 - 交通路口车辆检测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 模型参数
    parser.add_argument('--model', type=str, default='runs/train/weights/best.pt',
                       help='训练好的模型路径 (相对路径或绝对路径)')

    # 输入源参数
    parser.add_argument('--source', type=str, required=True,
                       help='输入源，可以是: 图像文件、视频文件、摄像头ID(如0)、图像文件夹路径')

    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--max-det', type=int, default=300,
                       help='每张图像最大检测数')

    # 设备参数
    parser.add_argument('--device', type=str, default='0',
                       help='推理设备，如: 0 (GPU0), cpu, 0,1 (多GPU)')

    # 保存参数
    parser.add_argument('--save-txt', action='store_true',
                       help='保存检测结果为YOLO格式txt文件')
    parser.add_argument('--save-img', action='store_true',
                       help='保存带检测框的图像')
    parser.add_argument('--save-vid', action='store_true',
                       help='保存带检测框的视频（仅对视频输入有效）')
    parser.add_argument('--project', type=str, default='runs',
                       help='结果保存目录 (相对路径或绝对路径)')
    parser.add_argument('--name', type=str, default='detect',
                       help='实验名称')
    parser.add_argument('--exist-ok', action='store_true',
                       help='是否覆盖同名实验')

    # 显示参数
    parser.add_argument('--show', action='store_true',
                       help='实时显示检测结果')
    parser.add_argument('--show-labels', action='store_true',
                       help='显示检测标签')
    parser.add_argument('--show-conf', action='store_true',
                       help='显示置信度')
    parser.add_argument('--line-width', type=int, default=3,
                       help='边界框线宽')

    # 其他参数
    parser.add_argument('--classes', type=int, nargs='+',
                       help='过滤特定类别，如: 0 2 3')
    parser.add_argument('--augment', action='store_true',
                       help='使用测试时增强')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--save-crop', action='store_true',
                       help='保存裁剪的检测目标')
    parser.add_argument('--hide-labels', action='store_true',
                       help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true',
                       help='隐藏置信度')

    return parser.parse_args()


def check_source_type(source: str) -> str:
    """
    检查输入源类型

    Args:
        source: 输入源路径或ID

    Returns:
        源类型: 'image', 'video', 'camera', 'folder', 'unknown'
    """
    # 检查是否为摄像头ID
    if source.isdigit():
        return 'camera'

    # 检查是否为文件或文件夹
    source_path = Path(source)

    # 首先检查路径是否存在
    if not source_path.exists():
        # 尝试使用绝对路径检查
        abs_path = os.path.abspath(source)
        source_path = Path(abs_path)
        if not source_path.exists():
            print(f"警告: 输入源不存在: {source}")
            return 'unknown'

    if source_path.is_file():
        # 检查文件扩展名
        ext = source_path.suffix.lower()
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

        if ext in image_exts:
            return 'image'
        elif ext in video_exts:
            return 'video'
        else:
            print(f"警告: 不支持的文件格式: {ext}")
            return 'unknown'
    elif source_path.is_dir():
        return 'folder'
    else:
        return 'unknown'


def load_model(model_path: str) -> Optional[YOLO]:
    """
    加载YOLOv8模型

    Args:
        model_path: 模型文件路径

    Returns:
        YOLO模型对象，失败返回None
    """
    try:
        # 检查模型文件是否存在
        if not check_file(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            print("请先运行训练脚本或提供正确的模型路径")
            return None

        print(f"加载模型: {model_path}")
        model = YOLO(model_path)

        # 检查模型类别
        if hasattr(model, 'names'):
            print(f"模型类别: {model.names}")

        return model

    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


def process_detection_results(results) -> List[Dict]:
    """
    处理检测结果

    Args:
        results: YOLOv8检测结果

    Returns:
        处理后的检测结果列表
    """
    processed_results = []

    for result in results:
        if result.boxes is None:
            continue

        # 提取检测信息
        boxes = result.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        xywh = boxes.xywh.cpu().numpy() if boxes.xywh is not None else []

        # 构建结果字典
        frame_results = []
        for i in range(len(cls_ids)):
            detection = {
                'frame_index': getattr(result, 'frame_index', 0),
                'class_id': int(cls_ids[i]),
                'class_name': result.names[int(cls_ids[i])] if hasattr(result, 'names') else f'class_{int(cls_ids[i])}',
                'confidence': float(confs[i]),
                'bbox_xyxy': [float(x) for x in xyxy[i]],
                'bbox_xywh': [float(x) for x in xywh[i]],
                'source_path': result.path if hasattr(result, 'path') else 'unknown',
                'image_size': result.orig_shape if hasattr(result, 'orig_shape') else (0, 0),
            }
            frame_results.append(detection)

        processed_results.extend(frame_results)

    return processed_results


def print_detection_stats(detections: List[Dict], processing_time: float):
    """
    打印检测统计信息

    Args:
        detections: 检测结果列表
        processing_time: 处理总时间
    """
    print("\n" + "=" * 60)
    print("📊 检测统计信息")
    print("=" * 60)

    if not detections:
        print("未检测到任何目标")
        return

    # 按类别统计
    class_stats = {}
    total_detections = len(detections)

    for det in detections:
        class_name = det['class_name']
        if class_name not in class_stats:
            class_stats[class_name] = {
                'count': 0,
                'total_confidence': 0.0,
                'min_confidence': 1.0,
                'max_confidence': 0.0,
            }

        stats = class_stats[class_name]
        stats['count'] += 1
        stats['total_confidence'] += det['confidence']
        stats['min_confidence'] = min(stats['min_confidence'], det['confidence'])
        stats['max_confidence'] = max(stats['max_confidence'], det['confidence'])

    # 打印统计信息
    print(f"总检测数: {total_detections}")
    print(f"处理时间: {processing_time:.2f} 秒")
    if processing_time > 0:
        print(f"平均FPS: {total_detections / processing_time:.2f}")

    print("\n📈 类别分布:")
    for class_name, stats in class_stats.items():
        avg_conf = stats['total_confidence'] / stats['count']
        percentage = (stats['count'] / total_detections) * 100
        print(f"  {class_name}:")
        print(f"    数量: {stats['count']} ({percentage:.1f}%)")
        print(f"    平均置信度: {avg_conf:.3f}")
        print(f"    最小置信度: {stats['min_confidence']:.3f}")
        print(f"    最大置信度: {stats['max_confidence']:.3f}")

    # 置信度分布
    conf_thresholds = [0.25, 0.5, 0.75, 0.9]
    print("\n🎯 置信度分布:")
    for threshold in conf_thresholds:
        count = sum(1 for det in detections if det['confidence'] >= threshold)
        percentage = (count / total_detections) * 100
        print(f"  ≥{threshold}: {count} ({percentage:.1f}%)")

    print("=" * 60)


def save_detection_results(detections: List[Dict], save_dir: Path, args):
    """
    保存检测结果

    Args:
        detections: 检测结果列表
        save_dir: 保存目录
        args: 命令行参数
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存为JSON
        json_file = save_dir / 'detections.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        print(f"检测结果已保存为JSON: {json_file}")

        # 保存为CSV
        csv_file = save_dir / 'detections.csv'
        save_results_to_csv(detections, str(csv_file))

        # 保存为YOLO格式txt（如果启用）
        if args.save_txt:
            txt_dir = save_dir / 'labels'
            txt_dir.mkdir(exist_ok=True)

            # 按源文件分组
            detections_by_source = {}
            for det in detections:
                source = det.get('source_path', 'unknown')
                if source not in detections_by_source:
                    detections_by_source[source] = []
                detections_by_source[source].append(det)

            for source, source_dets in detections_by_source.items():
                if source == 'unknown':
                    continue

                # 生成txt文件名
                source_path = Path(source)
                txt_file = txt_dir / f"{source_path.stem}.txt"

                with open(txt_file, 'w', encoding='utf-8') as f:
                    for det in source_dets:
                        # YOLO格式: class_id x_center y_center width height confidence
                        bbox = det['bbox_xywh']
                        img_w, img_h = det['image_size']
                        if img_w > 0 and img_h > 0:
                            x_center = bbox[0] / img_w
                            y_center = bbox[1] / img_h
                            width = bbox[2] / img_w
                            height = bbox[3] / img_h
                        else:
                            x_center = y_center = width = height = 0

                        line = (f"{det['class_id']} {x_center:.6f} {y_center:.6f} "
                               f"{width:.6f} {height:.6f} {det['confidence']:.6f}\n")
                        f.write(line)

            print(f"YOLO格式标签已保存到: {txt_dir}")

        # 保存统计报告
        report_file = save_dir / 'detection_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLOv8 检测报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"输入源: {args.source}\n")
            f.write(f"置信度阈值: {args.conf}\n")
            f.write(f"IOU阈值: {args.iou}\n\n")

            f.write(f"总检测数: {len(detections)}\n")

            # 类别统计
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            f.write("\n类别分布:\n")
            for class_name, count in class_counts.items():
                percentage = (count / len(detections)) * 100
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")

        print(f"检测报告已保存到: {report_file}")

    except Exception as e:
        print(f"保存检测结果时出错: {e}")


def detect_image(model: YOLO, image_path: str, args) -> List[Dict]:
    """
    检测单张图像

    Args:
        model: YOLO模型
        image_path: 图像路径
        args: 命令行参数

    Returns:
        检测结果列表
    """
    try:
        print(f"检测图像: {image_path}")

        # 执行检测
        results = model.predict(
            source=image_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            classes=args.classes,
            augment=args.augment,
            verbose=args.verbose,
        )

        # 处理结果
        detections = process_detection_results(results)

        # 保存带检测框的图像
        if args.save_img and results:
            save_dir = Path(args.project) / args.name / 'images'
            save_dir.mkdir(parents=True, exist_ok=True)

            image_name = Path(image_path).stem
            save_path = save_dir / f"{image_name}_detected.jpg"

            # 获取带检测框的图像
            plotted_img = results[0].plot(
                line_width=args.line_width,
                font_size=None,
                labels=not args.hide_labels,
                conf=not args.hide_conf,
            )

            # 保存图像
            cv2.imwrite(str(save_path), plotted_img)
            print(f"检测结果图像已保存: {save_path}")

        return detections

    except Exception as e:
        print(f"检测图像时出错 {image_path}: {e}")
        return []


def detect_video(model: YOLO, video_path: str, args) -> List[Dict]:
    """
    检测视频

    Args:
        model: YOLO模型
        video_path: 视频路径
        args: 命令行参数

    Returns:
        检测结果列表
    """
    try:
        print(f"检测视频: {video_path}")

        # 执行视频检测
        results = model.predict(
            source=video_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            classes=args.classes,
            augment=args.augment,
            verbose=args.verbose,
            save=args.save_vid,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            show=args.show,
        )

        # 处理结果
        detections = process_detection_results(results)

        if args.save_vid:
            print(f"检测视频已保存到: {Path(args.project) / args.name}")

        return detections

    except Exception as e:
        print(f"检测视频时出错 {video_path}: {e}")
        return []


def detect_camera(model: YOLO, camera_id: int, args) -> List[Dict]:
    """
    检测摄像头

    Args:
        model: YOLO模型
        camera_id: 摄像头ID
        args: 命令行参数

    Returns:
        检测结果列表
    """
    try:
        print(f"检测摄像头: {camera_id}")

        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头: {camera_id}")
            return []

        # 准备视频写入器（如果启用保存）
        writer = None
        if args.save_vid:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)
            video_path = save_dir / f"camera_{camera_id}_detected.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        detections = []
        frame_count = 0

        print("开始摄像头检测，按 'q' 键退出...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            frame_count += 1

            # 执行检测
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                classes=args.classes,
                augment=args.augment,
                verbose=False,
            )

            # 处理当前帧结果
            frame_detections = process_detection_results(results)
            for det in frame_detections:
                det['frame_index'] = frame_count
            detections.extend(frame_detections)

            # 显示结果
            if args.show and results:
                plotted_frame = results[0].plot(
                    line_width=args.line_width,
                    labels=not args.hide_labels,
                    conf=not args.hide_conf,
                )
                cv2.imshow(f'Camera {camera_id}', plotted_frame)

                # 保存帧
                if writer is not None:
                    writer.write(plotted_frame)

            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户中断检测")
                break

        # 清理
        cap.release()
        if writer is not None:
            writer.release()
            print(f"摄像头检测视频已保存: {video_path}")
        cv2.destroyAllWindows()

        print(f"摄像头检测完成，处理了 {frame_count} 帧")
        return detections

    except Exception as e:
        print(f"检测摄像头时出错: {e}")
        return []


def detect_folder(model: YOLO, folder_path: str, args) -> List[Dict]:
    """
    检测文件夹中的所有图像

    Args:
        model: YOLO模型
        folder_path: 文件夹路径
        args: 命令行参数

    Returns:
        检测结果列表
    """
    try:
        print(f"检测文件夹: {folder_path}")

        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"文件夹不存在或不是目录: {folder_path}")
            return []

        # 支持的图像格式
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for ext in image_exts:
            image_files.extend(list(folder_path.glob(f'*{ext}')))
            image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))

        if not image_files:
            print(f"文件夹中没有找到支持的图像文件: {folder_path}")
            return []

        print(f"找到 {len(image_files)} 个图像文件")

        detections = []
        processed_count = 0

        for image_file in image_files:
            image_detections = detect_image(model, str(image_file), args)
            detections.extend(image_detections)
            processed_count += 1

            # 显示进度
            if processed_count % 10 == 0:
                print(f"已处理 {processed_count}/{len(image_files)} 个图像")

        print(f"文件夹检测完成，处理了 {processed_count} 个图像")
        return detections

    except Exception as e:
        print(f"检测文件夹时出错: {e}")
        return []


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 将所有路径转换为基于当前工作目录的绝对路径
    current_dir = os.getcwd()

    # 处理模型路径
    if args.model and not os.path.isabs(args.model):
        args.model = os.path.abspath(os.path.join(current_dir, args.model))

    # 处理输入源路径（如果不是摄像头ID）
    if args.source and not args.source.isdigit():
        if not os.path.isabs(args.source):
            args.source = os.path.abspath(os.path.join(current_dir, args.source))

    # 处理项目目录路径
    if args.project and not os.path.isabs(args.project):
        args.project = os.path.abspath(os.path.join(current_dir, args.project))

    print("=" * 60)
    print("🚀 YOLOv8 检测脚本启动")
    print("=" * 60)

    # 检查输入源类型
    source_type = check_source_type(args.source)
    print(f"输入源类型: {source_type}")
    print(f"输入源: {args.source}")

    if source_type == 'unknown':
        print("错误: 无法识别的输入源类型")
        sys.exit(1)

    # 加载模型
    model = load_model(args.model)
    if model is None:
        sys.exit(1)

    # 开始检测
    start_time = time_sync()
    detections = []

    try:
        if source_type == 'image':
            detections = detect_image(model, args.source, args)
        elif source_type == 'video':
            detections = detect_video(model, args.source, args)
        elif source_type == 'camera':
            detections = detect_camera(model, int(args.source), args)
        elif source_type == 'folder':
            detections = detect_folder(model, args.source, args)

        processing_time = time_sync() - start_time

        # 打印统计信息
        print_detection_stats(detections, processing_time)

        # 保存结果
        if detections:
            save_dir = Path(args.project) / args.name
            save_detection_results(detections, save_dir, args)

        print("\n✅ 检测完成!")

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测被用户中断")
        processing_time = time_sync() - start_time
        print_detection_stats(detections, processing_time)

    except Exception as e:
        print(f"\n❌ 检测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()