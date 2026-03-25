#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 多目标跟踪脚本
支持ByteTrack、BoT-SORT等跟踪算法
包含MOT格式输出和跟踪性能统计
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
        load_video, draw_tracks, save_results_to_csv,
        check_file, time_sync
    )
except ImportError:
    print("警告: 无法导入工具函数，使用简化版本")
    # 定义简化版本的工具函数
    def load_video(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, {}
            return cap, {}
        except Exception as e:
            print(f"加载视频时出错: {e}")
            return None, {}

    def draw_tracks(image, tracks, track_history=None, max_history=30):
        if image is None:
            return None
        img = image.copy()
        for track in tracks:
            track_id = track.get('track_id', 0)
            bbox = track.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

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
        description='YOLOv8 多目标跟踪脚本 - 交通路口车辆跟踪',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 模型参数
    parser.add_argument('--model', type=str, default='runs/train/weights/best.pt',
                       help='训练好的模型路径 (相对路径或绝对路径)')

    # 输入源参数
    parser.add_argument('--source', type=str, required=True,
                       help='输入源，可以是: 视频文件、摄像头ID(如0)')

    # 跟踪器参数
    parser.add_argument('--tracker', type=str, default='configs/track.yaml',
                       help='跟踪器配置文件路径 (相对路径或绝对路径)，如: bytetrack.yaml')

    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--max-det', type=int, default=1000,
                       help='每帧最大检测数')

    # 设备参数
    parser.add_argument('--device', type=str, default='0',
                       help='推理设备，如: 0 (GPU0), cpu, 0,1 (多GPU)')

    # 保存参数
    parser.add_argument('--save-txt', action='store_true',
                       help='保存跟踪结果为MOT格式txt文件')
    parser.add_argument('--save-vid', action='store_true',
                       help='保存跟踪视频')
    parser.add_argument('--project', type=str, default='runs',
                       help='结果保存目录 (相对路径或绝对路径)')
    parser.add_argument('--name', type=str, default='track',
                       help='实验名称')
    parser.add_argument('--exist-ok', action='store_true',
                       help='是否覆盖同名实验')

    # 显示参数
    parser.add_argument('--show', action='store_true',
                       help='实时显示跟踪结果')
    parser.add_argument('--show-history', action='store_true',
                       help='显示轨迹历史')
    parser.add_argument('--history-length', type=int, default=30,
                       help='轨迹历史长度')
    parser.add_argument('--line-width', type=int, default=2,
                       help='边界框线宽')

    # 其他参数
    parser.add_argument('--classes', type=int, nargs='+',
                       help='过滤特定类别，如: 0 2 3')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--fps', type=int, default=30,
                       help='输出视频帧率（仅对摄像头有效）')
    parser.add_argument('--duration', type=int, default=0,
                       help='跟踪持续时间(秒)，0表示完整跟踪')

    return parser.parse_args()


def check_source_type(source: str) -> str:
    """
    检查输入源类型

    Args:
        source: 输入源路径或ID

    Returns:
        源类型: 'video', 'camera', 'unknown'
    """
    # 检查是否为摄像头ID
    if source.isdigit():
        return 'camera'

    # 检查是否为视频文件
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
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if ext in video_exts:
            return 'video'

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


def check_tracker_config(tracker_path: str) -> bool:
    """
    检查跟踪器配置文件

    Args:
        tracker_path: 跟踪器配置文件路径

    Returns:
        是否有效
    """
    try:
        # 检查文件是否存在
        if not check_file(tracker_path):
            print(f"警告: 跟踪器配置文件不存在: {tracker_path}")
            print("将使用默认跟踪器配置")
            return False

        print(f"使用跟踪器配置: {tracker_path}")
        return True

    except Exception as e:
        print(f"检查跟踪器配置时出错: {e}")
        return False


def process_tracking_results(results) -> List[Dict]:
    """
    处理跟踪结果

    Args:
        results: YOLOv8跟踪结果

    Returns:
        处理后的跟踪结果列表
    """
    processed_results = []

    for frame_idx, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue

        # 提取跟踪信息
        boxes = result.boxes
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        xywh = boxes.xywh.cpu().numpy() if boxes.xywh is not None else []

        # 构建结果字典
        for i in range(len(track_ids)):
            track_result = {
                'frame': frame_idx,
                'track_id': int(track_ids[i]),
                'class_id': int(cls_ids[i]),
                'class_name': result.names[int(cls_ids[i])] if hasattr(result, 'names') else f'class_{int(cls_ids[i])}',
                'confidence': float(confs[i]),
                'bbox_xyxy': [float(x) for x in xyxy[i]],
                'bbox_xywh': [float(x) for x in xywh[i]],
                'source_path': result.path if hasattr(result, 'path') else 'unknown',
                'image_size': result.orig_shape if hasattr(result, 'orig_shape') else (0, 0),
            }
            processed_results.append(track_result)

    return processed_results


def save_mot_format(tracks: List[Dict], save_path: str):
    """
    保存为MOT格式文件

    Args:
        tracks: 跟踪结果列表
        save_path: 保存路径
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            for track in tracks:
                # MOT格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
                frame = track['frame']
                track_id = track['track_id']
                bbox = track['bbox_xywh']
                conf = track['confidence']

                # 计算边界框左上角坐标
                bb_left = bbox[0] - bbox[2] / 2
                bb_top = bbox[1] - bbox[3] / 2
                bb_width = bbox[2]
                bb_height = bbox[3]

                line = (f"{frame},{track_id},{bb_left:.2f},{bb_top:.2f},"
                       f"{bb_width:.2f},{bb_height:.2f},{conf:.6f},-1,-1,-1\n")
                f.write(line)

        print(f"MOT格式结果已保存到: {save_path}")

    except Exception as e:
        print(f"保存MOT格式结果时出错: {e}")


def print_tracking_stats(tracks: List[Dict], processing_time: float, total_frames: int):
    """
    打印跟踪统计信息

    Args:
        tracks: 跟踪结果列表
        processing_time: 处理总时间
        total_frames: 总帧数
    """
    print("\n" + "=" * 60)
    print("📊 跟踪统计信息")
    print("=" * 60)

    if not tracks:
        print("未跟踪到任何目标")
        return

    # 基本统计
    total_tracks = len(tracks)
    unique_track_ids = len(set(t['track_id'] for t in tracks))

    print(f"总帧数: {total_frames}")
    print(f"总跟踪数: {total_tracks}")
    print(f"唯一跟踪ID数: {unique_track_ids}")
    print(f"处理时间: {processing_time:.2f} 秒")

    if processing_time > 0 and total_frames > 0:
        fps = total_frames / processing_time
        print(f"平均FPS: {fps:.2f}")

    # 按类别统计
    class_stats = {}
    for track in tracks:
        class_name = track['class_name']
        if class_name not in class_stats:
            class_stats[class_name] = {
                'count': 0,
                'track_ids': set(),
                'total_confidence': 0.0,
            }

        stats = class_stats[class_name]
        stats['count'] += 1
        stats['track_ids'].add(track['track_id'])
        stats['total_confidence'] += track['confidence']

    print("\n📈 类别分布:")
    for class_name, stats in class_stats.items():
        avg_conf = stats['total_confidence'] / stats['count']
        unique_tracks = len(stats['track_ids'])
        print(f"  {class_name}:")
        print(f"    总检测数: {stats['count']}")
        print(f"    唯一跟踪数: {unique_tracks}")
        print(f"    平均置信度: {avg_conf:.3f}")

    # 跟踪长度统计
    track_lengths = {}
    for track in tracks:
        track_id = track['track_id']
        if track_id not in track_lengths:
            track_lengths[track_id] = []
        track_lengths[track_id].append(track['frame'])

    if track_lengths:
        lengths = [len(frames) for frames in track_lengths.values()]
        avg_length = np.mean(lengths)
        max_length = np.max(lengths)
        min_length = np.min(lengths)

        print("\n🎯 跟踪长度统计:")
        print(f"    平均跟踪长度: {avg_length:.1f} 帧")
        print(f"    最长跟踪: {max_length} 帧")
        print(f"    最短跟踪: {min_length} 帧")

        # 跟踪长度分布
        length_bins = [1, 5, 10, 20, 50, 100]
        print(f"    跟踪长度分布:")
        for i in range(len(length_bins)):
            min_len = 1 if i == 0 else length_bins[i-1] + 1
            max_len = length_bins[i]
            count = sum(1 for length in lengths if min_len <= length <= max_len)
            percentage = (count / len(lengths)) * 100
            print(f"      {min_len}-{max_len}帧: {count} ({percentage:.1f}%)")

    print("=" * 60)


def save_tracking_results(tracks: List[Dict], save_dir: Path, args, total_frames: int):
    """
    保存跟踪结果

    Args:
        tracks: 跟踪结果列表
        save_dir: 保存目录
        args: 命令行参数
        total_frames: 总帧数
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存为JSON
        json_file = save_dir / 'tracks.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(tracks, f, indent=2, ensure_ascii=False)
        print(f"跟踪结果已保存为JSON: {json_file}")

        # 保存为CSV
        csv_file = save_dir / 'tracks.csv'
        save_results_to_csv(tracks, str(csv_file))

        # 保存为MOT格式（如果启用）
        if args.save_txt:
            mot_file = save_dir / 'tracks.txt'
            save_mot_format(tracks, str(mot_file))

        # 保存统计报告
        report_file = save_dir / 'tracking_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLOv8 跟踪报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"跟踪时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"跟踪器: {args.tracker}\n")
            f.write(f"输入源: {args.source}\n")
            f.write(f"置信度阈值: {args.conf}\n")
            f.write(f"IOU阈值: {args.iou}\n\n")

            f.write(f"总帧数: {total_frames}\n")
            f.write(f"总跟踪数: {len(tracks)}\n")
            f.write(f"唯一跟踪ID数: {len(set(t['track_id'] for t in tracks))}\n\n")

            # 类别统计
            class_counts = {}
            for track in tracks:
                class_name = track['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            f.write("类别分布:\n")
            for class_name, count in class_counts.items():
                percentage = (count / len(tracks)) * 100
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")

        print(f"跟踪报告已保存到: {report_file}")

    except Exception as e:
        print(f"保存跟踪结果时出错: {e}")


def track_video(model: YOLO, video_path: str, args) -> Tuple[List[Dict], int]:
    """
    跟踪视频

    Args:
        model: YOLO模型
        video_path: 视频路径
        args: 命令行参数

    Returns:
        (跟踪结果列表, 总帧数)
    """
    try:
        print(f"跟踪视频: {video_path}")

        # 检查跟踪器配置
        tracker_config = args.tracker if check_tracker_config(args.tracker) else None

        # 执行视频跟踪
        results = model.track(
            source=video_path,
            tracker=tracker_config,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            classes=args.classes,
            verbose=args.verbose,
            save=args.save_vid,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            show=args.show,
        )

        # 处理结果
        tracks = process_tracking_results(results)

        # 获取总帧数
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if args.save_vid:
            print(f"跟踪视频已保存到: {Path(args.project) / args.name}")

        return tracks, total_frames

    except Exception as e:
        print(f"跟踪视频时出错 {video_path}: {e}")
        return [], 0


def track_camera(model: YOLO, camera_id: int, args) -> Tuple[List[Dict], int]:
    """
    跟踪摄像头

    Args:
        model: YOLO模型
        camera_id: 摄像头ID
        args: 命令行参数

    Returns:
        (跟踪结果列表, 总帧数)
    """
    try:
        print(f"跟踪摄像头: {camera_id}")

        # 检查跟踪器配置
        tracker_config = args.tracker if check_tracker_config(args.tracker) else None

        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头: {camera_id}")
            return [], 0

        # 获取摄像头参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"摄像头参数: {width}x{height}, FPS: {fps:.2f}")

        # 准备视频写入器（如果启用保存）
        writer = None
        if args.save_vid:
            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)
            video_path = save_dir / f"camera_{camera_id}_tracked.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        tracks = []
        frame_count = 0
        track_history = {}

        print("开始摄像头跟踪，按 'q' 键退出...")

        start_time = time.time()
        max_duration = args.duration if args.duration > 0 else float('inf')

        while True:
            # 检查持续时间
            current_time = time.time()
            if current_time - start_time > max_duration:
                print(f"达到指定持续时间 {args.duration} 秒，停止跟踪")
                break

            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            frame_count += 1

            # 执行跟踪
            results = model.track(
                source=frame,
                tracker=tracker_config,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                classes=args.classes,
                verbose=False,
            )

            # 处理当前帧结果
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                frame_tracks = process_tracking_results(results)
                for track in frame_tracks:
                    track['frame'] = frame_count
                tracks.extend(frame_tracks)

                # 更新轨迹历史
                for track in frame_tracks:
                    track_id = track['track_id']
                    bbox = track['bbox_xywh']
                    center = (bbox[0], bbox[1])

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(center)

                    # 限制历史长度
                    if len(track_history[track_id]) > args.history_length:
                        track_history[track_id] = track_history[track_id][-args.history_length:]

            # 显示结果
            if args.show and results:
                # 准备跟踪数据用于绘制
                frame_track_data = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes.id)):
                        track_data = {
                            'track_id': int(boxes.id[i]),
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                            'class_id': int(boxes.cls[i]) if boxes.cls is not None else 0,
                            'score': float(boxes.conf[i]) if boxes.conf is not None else 0.0,
                        }
                        frame_track_data.append(track_data)

                # 绘制跟踪结果
                if args.show_history:
                    plotted_frame = draw_tracks(frame, frame_track_data, track_history, args.history_length)
                else:
                    plotted_frame = draw_tracks(frame, frame_track_data)

                # 显示FPS
                fps_text = f"FPS: {1/(time.time() - current_time):.1f}" if frame_count > 1 else "FPS: --"
                cv2.putText(plotted_frame, fps_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(f'Camera {camera_id} Tracking', plotted_frame)

                # 保存帧
                if writer is not None:
                    writer.write(plotted_frame)

            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户中断跟踪")
                break

        # 清理
        cap.release()
        if writer is not None:
            writer.release()
            print(f"摄像头跟踪视频已保存: {video_path}")
        cv2.destroyAllWindows()

        print(f"摄像头跟踪完成，处理了 {frame_count} 帧")
        return tracks, frame_count

    except Exception as e:
        print(f"跟踪摄像头时出错: {e}")
        return [], 0


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

    # 处理跟踪器配置文件路径
    if args.tracker and not os.path.isabs(args.tracker):
        args.tracker = os.path.abspath(os.path.join(current_dir, args.tracker))

    # 处理项目目录路径
    if args.project and not os.path.isabs(args.project):
        args.project = os.path.abspath(os.path.join(current_dir, args.project))

    print("=" * 60)
    print("🚀 YOLOv8 多目标跟踪脚本启动")
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

    # 开始跟踪
    start_time = time_sync()
    tracks = []
    total_frames = 0

    try:
        if source_type == 'video':
            tracks, total_frames = track_video(model, args.source, args)
        elif source_type == 'camera':
            tracks, total_frames = track_camera(model, int(args.source), args)

        processing_time = time_sync() - start_time

        # 打印统计信息
        print_tracking_stats(tracks, processing_time, total_frames)

        # 保存结果
        if tracks:
            save_dir = Path(args.project) / args.name
            save_tracking_results(tracks, save_dir, args, total_frames)

        print("\n✅ 跟踪完成!")

    except KeyboardInterrupt:
        print("\n\n⚠️ 跟踪被用户中断")
        processing_time = time_sync() - start_time
        print_tracking_stats(tracks, processing_time, total_frames)

    except Exception as e:
        print(f"\n❌ 跟踪过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()