import pandas as pd
import os
from pathlib import Path

# 1. 配置路径
base_dir = Path("runs/train")
exp_names = ['expA_baseline', 'expB_cbam', 'expC_bifpn', 'expD_combined']

summary_data = []

print("开始搜索各实验的最佳指标 (基于 mAP50)...")

for exp in exp_names:
    csv_path = base_dir / exp / "results.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # 清洗列名，去掉空格
            df.columns = [c.strip() for c in df.columns]
            
            if not df.empty:
                # 【核心修改】：寻找 mAP50(B) 最大值所在的索引
                # 如果有多个相同的最大值，idxmax() 会返回第一个
                best_idx = df['metrics/mAP50(B)'].idxmax()
                best_row = df.loc[best_idx]
                
                # 记录一下是第几个 epoch 达到的最佳
                best_epoch = best_row.get('epoch', best_idx)
                
                row = {
                    'Experiment': exp,
                    'mAP50': round(best_row['metrics/mAP50(B)'], 4),
                    'mAP50-95': round(best_row['metrics/mAP50-95(B)'], 4),
                    'car_AP': '-',  # 训练日志通常不分库记录，如有需要需从 val 日志提取
                    'bus_AP': '-',
                    'van_AP': '-',
                    'truck_AP': '-',
                    'Notes': f'Best @ Epoch {best_epoch}'
                }
                summary_data.append(row)
                print(f"✅ 已提取 {exp} 的最佳结果 (Epoch {best_epoch})")
        except Exception as e:
            print(f"✗ 读取失败 {exp}: {e}")
    else:
        print(f"❌ 未找到文件: {csv_path}")

# 2. 保存为 plot_results.py 需要的 results.csv
if summary_data:
    result_df = pd.DataFrame(summary_data)
    result_df.to_csv("results.csv", index=False, encoding='utf-8')
    print("\n" + "="*50)
    print("🎯 最佳指标汇总表 results.csv 已生成！")
    print(result_df[['Experiment', 'mAP50', 'Notes']])
    print("="*50)
else:
    print("\n错误：没有提取到任何数据，请检查路径。")