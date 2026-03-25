import os
import sys
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# 1. 注册自定义模块（必须，否则无法加载实验B、C、D的模型）
from ultralytics.nn.modules import CBAM, BiFPN
import ultralytics.nn.tasks as tasks
tasks.CBAM = CBAM
tasks.BiFPN = BiFPN

def extract():
    # 配置文件和实验路径
    data_cfg = "configs/UA-DETRAC.yaml"
    base_dir = Path("runs/train")
    exps = ['expA_baseline', 'expB_cbam', 'expC_bifpn', 'expD_combined']
    
    class_names = ['car', 'bus', 'van', 'truck']
    all_results = []

    print("开始提取各类别详细指标...")

    for exp in exps:
        model_path = base_dir / exp / "weights" / "best.pt"
        if not model_path.exists():
            print(f"❌ 未找到模型: {model_path}")
            continue
        
        print(f"\n正在评估: {exp} ...")
        # 加载最佳模型
        model = YOLO(model_path)
        
        # 执行验证 (使用验证集)
        # plots=False 可以加快速度
        results = model.val(data=data_cfg, plots=False, verbose=False)
        
        # 提取每个类别的 AP50
        # results.box.maps 返回的是每个类别的 mAP50-95，
        # 我们通常论文用 AP50 (results.box.ap50)
        maps = results.box.ap50 
        
        res_dict = {
            "Experiment": exp,
            "mAP50_Total": round(results.box.map50, 4),
            "car_AP": round(maps[0], 4),
            "bus_AP": round(maps[1], 4),
            "van_AP": round(maps[2], 4),
            "truck_AP": round(maps[3], 4)
        }
        all_results.append(res_dict)
        print(f"✅ {exp} 提取完成: Car:{res_dict['car_AP']}, Bus:{res_dict['bus_AP']}, Truck:{res_dict['truck_AP']}")

    # 保存为新的详细汇总表
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("detailed_class_ap.csv", index=False)
        print("\n" + "="*50)
        print("🎉 详细类别指标已保存至: detailed_class_ap.csv")
        print(df.to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    extract()