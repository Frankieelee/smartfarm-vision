"""
YOLO 完整验证脚本 - 获取所有评估指标.

功能：
1. 标准 YOLO 验证（mAP@0.5, mAP@0.5-0.95）
2. 多 IoU 阈值自定义 mAP（mAP@0.3, mAP@0.4, mAP@0.6, mAP@0.75）
3. 每类别详细 AP
4. 完整对比表格和结果保存

使用方法：
    python validation_complete.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ultralytics import YOLO

# 🔑 关键：直接导入 YOLO 内部的 AP 计算函数
from ultralytics.utils.metrics import ap_per_class, box_iou

# ============================================================================
# 配置区域
# ============================================================================

# 模型配置
MODEL_PATH = "/tmp/dataset/yolo11s-0.8-1.pt"

# 数据集配置
DATA_YAML = "/tmp/dataset/yolo11/v13/data.yaml"
DATASETS = {
    "val": DATA_YAML,  # 验证集
    "train": DATA_YAML,  # 训练集（检查过拟合）
}

# 多 IoU 阈值配置
IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.75]

# 验证参数
VAL_ARGS = {
    "imgsz": 800,  # 图像尺寸
    "batch": 8,  # 批次大小
    "conf": 0.25,  # 置信度阈值
    "iou": 0.5,  # NMS IoU 阈值
    "max_det": 300,  # 每张图最大检测数
    "device": 0,  # GPU 设备
    "workers": 6,  # 数据加载线程数
    "save_json": False,  # 不保存 JSON（避免冗余）
    "save_hybrid": False,  # 不保存混合标签
    "verbose": False,  # 安静模式
    "plots": False,  # 不生成图表（加快速度）
}

# 输出配置
OUTPUT_DIR = Path("validation_complete_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# 核心函数
# ============================================================================


def get_predictions_and_labels(model, data_yaml, split="val", conf_threshold=0.25, nms_iou=0.5):
    """获取预测结果和真实标签（用于自定义 mAP 计算）."""
    with open(data_yaml) as f:
        yaml.safe_load(f)

    data_path = Path(data_yaml).parent

    # 确定数据集路径
    if split == "val":
        image_dir = data_path / "valid" / "images"
        label_dir = data_path / "valid" / "labels"
    elif split == "train":
        image_dir = data_path / "train" / "images"
        label_dir = data_path / "train" / "labels"
    else:
        image_dir = data_path / split / "images"
        label_dir = data_path / split / "labels"

    # 获取所有图像
    sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    # 使用 YOLO 进行预测
    results = model.predict(source=str(image_dir), conf=conf_threshold, iou=nms_iou, save=False, verbose=False)

    # 收集预测和标签
    all_stats = []

    for result in results:
        img_id = Path(result.path).stem
        label_file = label_dir / f"{img_id}.txt"

        # 预测框
        pred_boxes = []
        pred_confs = []
        pred_classes = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            img_h, img_w = result.orig_shape

            # 归一化坐标
            for box, conf, cls in zip(boxes, confs, classes):
                norm_box = np.array([box[0] / img_w, box[1] / img_h, box[2] / img_w, box[3] / img_h])
                pred_boxes.append(norm_box)
                pred_confs.append(conf)
                pred_classes.append(int(cls))

        # 真实标签
        target_boxes = []
        target_classes = []

        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 转换为 xyxy 格式
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    target_boxes.append(np.array([x1, y1, x2, y2]))
                    target_classes.append(class_id)

        all_stats.append(
            {
                "pred_boxes": np.array(pred_boxes) if pred_boxes else np.empty((0, 4)),
                "pred_confs": np.array(pred_confs) if pred_confs else np.empty(0),
                "pred_classes": np.array(pred_classes) if pred_classes else np.empty(0, dtype=int),
                "target_boxes": np.array(target_boxes) if target_boxes else np.empty((0, 4)),
                "target_classes": np.array(target_classes) if target_classes else np.empty(0, dtype=int),
            }
        )

    return all_stats


def compute_map_at_iou_threshold(all_stats, iou_threshold, class_names):
    """计算指定 IoU 阈值下的 mAP（使用 YOLO 内部函数）."""
    all_pred_boxes = []
    all_pred_confs = []
    all_pred_classes = []
    all_target_classes = []
    all_tp = []

    for stats in all_stats:
        pred_boxes = stats["pred_boxes"]
        pred_confs = stats["pred_confs"]
        pred_classes = stats["pred_classes"]
        target_boxes = stats["target_boxes"]
        target_classes = stats["target_classes"]

        if len(pred_boxes) == 0:
            all_target_classes.extend(target_classes)
            continue

        if len(target_boxes) == 0:
            all_pred_boxes.extend(pred_boxes)
            all_pred_confs.extend(pred_confs)
            all_pred_classes.extend(pred_classes)
            all_tp.extend([False] * len(pred_boxes))
            continue

        # 计算 IoU
        pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        target_boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
        iou_matrix = box_iou(pred_boxes_tensor, target_boxes_tensor).numpy()

        # 标记 TP/FP
        tp_flags = np.zeros(len(pred_boxes), dtype=bool)
        matched_targets = set()

        sorted_indices = np.argsort(-pred_confs)

        for pred_idx in sorted_indices:
            pred_class = pred_classes[pred_idx]

            best_iou = 0
            best_target_idx = -1

            for target_idx in range(len(target_boxes)):
                if target_classes[target_idx] != pred_class:
                    continue
                if target_idx in matched_targets:
                    continue

                iou = iou_matrix[pred_idx, target_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold and best_target_idx != -1:
                tp_flags[pred_idx] = True
                matched_targets.add(best_target_idx)

        all_pred_boxes.extend(pred_boxes)
        all_pred_confs.extend(pred_confs)
        all_pred_classes.extend(pred_classes)
        all_tp.extend(tp_flags)
        all_target_classes.extend(target_classes)

    # 转换为 numpy 数组
    all_pred_confs = np.array(all_pred_confs)
    all_pred_classes = np.array(all_pred_classes)
    all_target_classes = np.array(all_target_classes)
    all_tp = np.array(all_tp).reshape(-1, 1)

    if len(all_pred_confs) == 0:
        return {
            "mAP": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "per_class_ap": {},
        }

    # 使用 YOLO 内部函数计算 AP
    _tp, _fp, p, r, _f1, ap, unique_classes, *_ = ap_per_class(
        tp=all_tp,
        conf=all_pred_confs,
        pred_cls=all_pred_classes,
        target_cls=all_target_classes,
        plot=False,
    )

    mAP = ap[:, 0].mean()

    per_class_ap = {}
    for i, cls_id in enumerate(unique_classes):
        class_name = class_names.get(cls_id, f"class_{cls_id}")
        per_class_ap[class_name] = float(ap[i, 0])

    return {
        "mAP": float(mAP),
        "Precision": float(p.mean()),
        "Recall": float(r.mean()),
        "per_class_ap": per_class_ap,
    }


def validate_complete(model, dataset_name, data_yaml, split="val"):
    """完整验证：同时获取标准 YOLO 指标和多 IoU 阈值的自定义 mAP."""
    print(f"\n{'=' * 70}")
    print(f"📊 完整验证: {dataset_name} ({split})")
    print(f"{'=' * 70}")

    # ===== 步骤 1: 标准 YOLO 验证 =====
    print("\n🔸 步骤 1/2: 标准 YOLO 验证...")

    output_subdir = OUTPUT_DIR / f"{dataset_name}_{split}_{TIMESTAMP}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    results = model.val(
        data=data_yaml,
        split=split,
        project=str(output_subdir.parent),
        name=output_subdir.name,
        exist_ok=True,
        **VAL_ARGS,
    )

    # 提取标准指标
    standard_metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "Precision": float(results.box.mp),
        "Recall": float(results.box.mr),
    }

    # 每类别 AP
    per_class_standard = {}
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.names[int(class_idx)]
            per_class_standard[class_name] = {
                "AP50": float(results.box.ap50[i]),
                "AP": float(results.box.ap[i]),
            }

    print("✅ 标准验证完成")
    print(f"   mAP@0.5:     {standard_metrics['mAP50']:.4f}")
    print(f"   mAP@0.5-0.95: {standard_metrics['mAP50-95']:.4f}")

    # ===== 步骤 2: 多 IoU 阈值自定义 mAP =====
    print("\n🔸 步骤 2/2: 多 IoU 阈值自定义 mAP...")
    print("   正在生成预测和加载标签...")

    all_stats = get_predictions_and_labels(
        model, data_yaml, split=split, conf_threshold=VAL_ARGS["conf"], nms_iou=VAL_ARGS["iou"]
    )

    print(f"   正在计算 mAP@{IOU_THRESHOLDS}...")

    # 加载类别名称
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    class_names = {i: name for i, name in enumerate(data_config["names"])}

    # 计算每个 IoU 阈值的 mAP
    custom_metrics = {}
    for iou_threshold in IOU_THRESHOLDS:
        metrics = compute_map_at_iou_threshold(all_stats, iou_threshold, class_names)
        custom_metrics[iou_threshold] = metrics

    print("✅ 自定义 mAP 计算完成")

    # ===== 合并结果 =====
    complete_result = {
        "dataset": dataset_name,
        "split": split,
        "model": str(MODEL_PATH),
        "timestamp": TIMESTAMP,
        "standard_metrics": standard_metrics,
        "per_class_standard": per_class_standard,
        "custom_iou_metrics": custom_metrics,
    }

    # 打印完整结果
    print(f"\n{'─' * 70}")
    print("📈 完整验证结果总结")
    print(f"{'─' * 70}")

    print("\n🔹 标准 YOLO 指标:")
    print(f"   mAP@0.5:      {standard_metrics['mAP50']:.4f} ({standard_metrics['mAP50'] * 100:.2f}%)")
    print(f"   mAP@0.5-0.95: {standard_metrics['mAP50-95']:.4f} ({standard_metrics['mAP50-95'] * 100:.2f}%)")
    print(f"   Precision:    {standard_metrics['Precision']:.4f} ({standard_metrics['Precision'] * 100:.2f}%)")
    print(f"   Recall:       {standard_metrics['Recall']:.4f} ({standard_metrics['Recall'] * 100:.2f}%)")

    print("\n🔹 多 IoU 阈值 mAP:")
    for iou_threshold in IOU_THRESHOLDS:
        metrics = custom_metrics[iou_threshold]
        print(f"   mAP@{iou_threshold}:      {metrics['mAP']:.4f} ({metrics['mAP'] * 100:.2f}%)")

    if per_class_standard:
        print("\n🔹 每类别 AP@0.5:")
        for class_name, class_metrics in per_class_standard.items():
            print(f"   {class_name:12s}: {class_metrics['AP50']:.4f}")

    return complete_result


def main():
    """主函数."""
    print(f"{'=' * 70}")
    print("🚀 YOLO 完整验证脚本")
    print(f"{'=' * 70}")
    print(f"📦 模型: {MODEL_PATH}")
    print(f"📂 输出目录: {OUTPUT_DIR}/")
    print(f"📏 IoU 阈值: {IOU_THRESHOLDS}")
    print(f"{'=' * 70}\n")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("🔧 加载模型...")
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功")
    print(f"   类别: {model.names}\n")

    # 在所有数据集上验证
    all_results = []

    for dataset_name, data_yaml in DATASETS.items():
        try:
            # 确定分割
            if dataset_name in ["train", "val", "test"]:
                split = dataset_name
                actual_name = Path(data_yaml).parent.name
            else:
                split = "val"
                actual_name = dataset_name

            result = validate_complete(model, actual_name, data_yaml, split=split)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            import traceback

            traceback.print_exc()

    # ===== 保存完整结果 =====
    print(f"\n{'=' * 70}")
    print("💾 保存结果...")
    print(f"{'=' * 70}")

    # JSON 格式（完整数据）
    json_file = OUTPUT_DIR / f"complete_results_{TIMESTAMP}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"📊 JSON 保存到: {json_file}")

    # CSV 格式（汇总表格）
    if all_results:
        csv_data = []
        for result in all_results:
            row = {
                "数据集": result["dataset"],
                "分割": result["split"],
                "mAP@0.5": f"{result['standard_metrics']['mAP50']:.4f}",
                "mAP@0.5-0.95": f"{result['standard_metrics']['mAP50-95']:.4f}",
            }

            # 添加多 IoU 阈值的 mAP
            for iou_threshold in IOU_THRESHOLDS:
                if iou_threshold in result["custom_iou_metrics"]:
                    row[f"mAP@{iou_threshold}"] = f"{result['custom_iou_metrics'][iou_threshold]['mAP']:.4f}"

            row["Precision"] = f"{result['standard_metrics']['Precision']:.4f}"
            row["Recall"] = f"{result['standard_metrics']['Recall']:.4f}"

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_file = OUTPUT_DIR / f"complete_summary_{TIMESTAMP}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        print("\n📋 汇总表格:")
        print(df.to_string(index=False))
        print(f"\n💾 CSV 保存到: {csv_file}")

    # 打印 mAP 对比
    print(f"\n{'=' * 70}")
    print("📊 mAP 对比分析")
    print(f"{'=' * 70}")

    for result in all_results:
        print(f"\n📌 {result['dataset']} ({result['split']}):")

        standard_map50 = result["standard_metrics"]["mAP50"]
        custom_metrics = result["custom_iou_metrics"]

        print(f"   YOLO 标准 mAP@0.5: {standard_map50:.4f} ({standard_map50 * 100:.2f}%)")

        if 0.5 in custom_metrics:
            custom_map50 = custom_metrics[0.5]["mAP"]
            diff = abs(standard_map50 - custom_map50)
            print(f"   自定义 mAP@0.5:    {custom_map50:.4f} ({custom_map50 * 100:.2f}%)")
            print(f"   差异:              {diff:.4f} ({diff / standard_map50 * 100:.2f}%) ✓")

        print("\n   多 IoU 阈值 mAP 变化:")
        base_map = custom_metrics.get(0.5, {}).get("mAP", 0)
        for iou_threshold in IOU_THRESHOLDS:
            if iou_threshold in custom_metrics:
                current_map = custom_metrics[iou_threshold]["mAP"]
                if iou_threshold != 0.5 and base_map > 0:
                    diff = current_map - base_map
                    pct = diff / base_map * 100
                    symbol = "📈" if diff >= 0 else "📉"
                    print(f"      {symbol} mAP@{iou_threshold}: {current_map:.4f} ({pct:+.2f}%)")
                else:
                    print(f"      • mAP@{iou_threshold}: {current_map:.4f} (基准)")

    print(f"\n{'=' * 70}")
    print("🎉 所有验证完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
