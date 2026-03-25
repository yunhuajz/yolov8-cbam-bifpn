import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 配置
# ============================================================
CSV_PATH = "detailed_class_ap.csv"
SAVE_DIR = "picture"  # 目标保存目录

# 设置绘图风格和中文字体支持
try:
    plt.style.use('seaborn-v0_8-muted') 
except:
    plt.style.use('ggplot') # 备用风格

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_thesis_results():
    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到数据文件 {CSV_PATH}，请确保脚本在项目根目录运行。")
        return

    # 1. 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[*] 已创建目录: {SAVE_DIR}")

    # 读取数据
    df = pd.read_csv(CSV_PATH)
    
    # 实验名称映射，让图表更专业
    name_map = {
        'expA_baseline': 'Baseline\n(YOLOv8n)',
        'expB_cbam': 'YOLOv8n\n+CBAM',
        'expC_bifpn': 'YOLOv8n\n+BiFPN',
        'expD_combined': 'YOLOv8n\n+Combined'
    }
    df['DisplayName'] = df['Experiment'].map(name_map)

    # --- 图 1: 总体 mAP50 对比图 ---
    print("[*] 正在绘制总体对比图...")
    plt.figure(figsize=(10, 6))
    colors = ['#bdc3c7', '#3498db', '#e67e22', '#9b59b6'] # 灰、蓝、橙、紫
    bars = plt.bar(df['DisplayName'], df['mAP50_Total'], color=colors, width=0.55)
    
    plt.title('不同改进方案的检测精度(mAP50)对比', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('mAP50 分数', fontsize=12)
    plt.ylim(0.55, 0.68) # 聚焦差异区间，让提升更直观
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标注
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    overall_fig_path = os.path.join(SAVE_DIR, 'mAP_Overall_Comparison.png')
    plt.savefig(overall_fig_path, dpi=300)
    print(f"✅ 已保存: {overall_fig_path}")

    # --- 图 2: 各类别指标细节对比图 ---
    print("[*] 正在绘制类别细节图...")
    classes = ['car_AP', 'bus_AP', 'van_AP', 'truck_AP']
    class_labels = ['轿车(Car)', '公交(Bus)', '面包车(Van)', '货车(Truck)']
    
    x = np.arange(len(df['DisplayName']))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 循环绘制四个类别的柱子
    for i, cls in enumerate(classes):
        ax.bar(x + i*width - width*1.5, df[cls], width, label=class_labels[i])

    ax.set_title('各改进模型在不同交通类别上的检测精度(AP50)对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['DisplayName'])
    ax.set_ylabel('AP 分数', fontsize=12)
    ax.set_ylim(0, 1.0) # AP 的标准区间
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    class_fig_path = os.path.join(SAVE_DIR, 'Class_AP_Comparison.png')
    plt.savefig(class_fig_path, dpi=300)
    print(f"✅ 已保存: {class_fig_path}")

if __name__ == "__main__":
    plot_thesis_results()
    print("\n[🎉] 所有可视化成果已存放至 picture 文件夹！")