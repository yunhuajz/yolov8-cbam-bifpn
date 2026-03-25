import csv
import os

val_dirs = [
    "valA_b80", "valA_b802", "valB_b80", "valB_b802",
    "valC_b80", "valC_b802", "valD_b80", "valD_b802"
]

print("实验\t版本\tmAP50\tmAP50-95\tPrecision\tRecall")
for d in val_dirs:
    results_file = os.path.join("runs", "val", d, "results.csv")
    if not os.path.exists(results_file):
        print(f"{d}\t--\t文件不存在")
        continue
    with open(results_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        # 找到所需列的索引
        idx_map50 = header.index("metrics/mAP50(B)") if "metrics/mAP50(B)" in header else None
        idx_map50_95 = header.index("metrics/mAP50-95(B)") if "metrics/mAP50-95(B)" in header else None
        idx_precision = header.index("metrics/precision(B)") if "metrics/precision(B)" in header else None
        idx_recall = header.index("metrics/recall(B)") if "metrics/recall(B)" in header else None
        # 读取最后一行
        last_row = None
        for row in reader:
            if row:
                last_row = row
        if last_row:
            map50 = last_row[idx_map50] if idx_map50 is not None else 'N/A'
            map50_95 = last_row[idx_map50_95] if idx_map50_95 is not None else 'N/A'
            precision = last_row[idx_precision] if idx_precision is not None else 'N/A'
            recall = last_row[idx_recall] if idx_recall is not None else 'N/A'
            print(f"{d[:5]}\t{d[5:]}\t{map50}\t{map50_95}\t{precision}\t{recall}")