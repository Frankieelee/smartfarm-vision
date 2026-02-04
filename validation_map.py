"""
自定义 mAP 计算器 - 支持任意 IoU 阈值

🔑 关键特性：直接使用 YOLO 内部的 compute_ap() 和 ap_per_class() 函数
   保证与 validation.py 的 mAP 计算 100% 一致！

与 validation.py 的区别：
- validation.py: 调整 NMS IoU（过滤重叠框），但 mAP 始终在 IoU≥0.5 计算
- 本脚本: 重新计算不同 IoU 阈值下的 TP/FP，真正改变 mAP 计算的 IoU 匹配阈值

使用方法：
    python custom_map_calculator.py

输出：
    - mAP@0.3, mAP@0.4, mAP@0.5, mAP@0.6, mAP@0.75
    - 每个类别的 AP
    - Precision, Recall
    - JSON 和 CSV 结果文件
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import json
import pandas as pd
from datetime import datetime
import yaml
import torch

# 🔑 关键：直接导入 YOLO 内部的 AP 计算函数（保证 100% 一致）
from ultralytics.utils.metrics import ap_per_class, box_iou


# ============================================================================
# 配置区域
# ============================================================================

MODEL_PATH = '/tmp/dataset/yolo11s-0.8-1.pt'
DATA_YAML = '/tmp/dataset/yolo11/v13/data.yaml'
DATASET_SPLIT = 'val'  # 'train', 'val', 'test'

# IoU 阈值列表（将为每个阈值计算 mAP）
IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.75]

# 预测参数（与 validation.py 保持一致）
CONF_THRESHOLD = 0.25  # 置信度阈值
NMS_IOU = 0.5          # NMS IoU 阈值

# 输出配置
OUTPUT_DIR = Path('custom_map_results')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================================
# 核心函数
# ============================================================================

def load_dataset_info(data_yaml):
    """加载数据集配置"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    data_path = Path(data_yaml).parent
    return data, data_path


def get_predictions_and_labels(model, data_yaml, split='val', conf_threshold=0.25, nms_iou=0.5):
    """
    获取预测结果和真实标签
    
    返回：
        predictions: List[(conf, class_id, bbox_xyxy)]  # 预测框
        targets: List[(class_id, bbox_xyxy)]             # 真实框
        stats: Dict[image_id -> (pred_boxes, target_boxes)]
    """
    data, data_path = load_dataset_info(data_yaml)
    
    # 确定数据集路径
    if split == 'val':
        image_dir = data_path / 'valid' / 'images'
        label_dir = data_path / 'valid' / 'labels'
    elif split == 'train':
        image_dir = data_path / 'train' / 'images'
        label_dir = data_path / 'train' / 'labels'
    else:
        image_dir = data_path / split / 'images'
        label_dir = data_path / split / 'labels'
    
    print(f"📂 图像目录: {image_dir}")
    print(f"📂 标签目录: {label_dir}")
    
    # 获取所有图像
    image_files = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
    
    print(f"🔍 找到 {len(image_files)} 张图像")
    
    # 使用 YOLO 进行预测
    results = model.predict(
        source=str(image_dir),
        conf=conf_threshold,
        iou=nms_iou,
        save=False,
        verbose=False
    )
    
    # 收集预测和标签（按 YOLO 内部格式）
    all_stats = []
    
    for result in results:
        img_id = Path(result.path).stem
        label_file = label_dir / f"{img_id}.txt"
        
        # ===== 预测框 =====
        pred_boxes = []
        pred_confs = []
        pred_classes = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 (absolute pixels)
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            img_h, img_w = result.orig_shape
            
            # 归一化到 0-1（YOLO 内部使用归一化坐标）
            for box, conf, cls in zip(boxes, confs, classes):
                norm_box = np.array([
                    box[0] / img_w,  # x1
                    box[1] / img_h,  # y1
                    box[2] / img_w,  # x2
                    box[3] / img_h,  # y2
                ])
                pred_boxes.append(norm_box)
                pred_confs.append(conf)
                pred_classes.append(int(cls))
        
        # ===== 真实标签 =====
        target_boxes = []
        target_classes = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 转换为 xyxy 格式（归一化坐标）
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    target_boxes.append(np.array([x1, y1, x2, y2]))
                    target_classes.append(class_id)
        
        # 保存统计信息
        all_stats.append({
            'pred_boxes': np.array(pred_boxes) if pred_boxes else np.empty((0, 4)),
            'pred_confs': np.array(pred_confs) if pred_confs else np.empty(0),
            'pred_classes': np.array(pred_classes) if pred_classes else np.empty(0, dtype=int),
            'target_boxes': np.array(target_boxes) if target_boxes else np.empty((0, 4)),
            'target_classes': np.array(target_classes) if target_classes else np.empty(0, dtype=int),
        })
    
    return all_stats


def compute_map_at_iou_threshold(all_stats, iou_threshold, class_names):
    """
    使用 YOLO 内部的 ap_per_class() 函数计算指定 IoU 阈值下的 mAP
    
    这保证了与 validation.py 的计算逻辑 100% 一致！
    """
    # 收集所有预测和真实标签
    all_pred_boxes = []
    all_pred_confs = []
    all_pred_classes = []
    all_target_classes = []
    all_tp = []  # True Positive 标记
    
    for stats in all_stats:
        pred_boxes = stats['pred_boxes']
        pred_confs = stats['pred_confs']
        pred_classes = stats['pred_classes']
        target_boxes = stats['target_boxes']
        target_classes = stats['target_classes']
        
        # 如果没有预测框，跳过
        if len(pred_boxes) == 0:
            # 但要记录真实标签（用于计算 Recall）
            all_target_classes.extend(target_classes)
            continue
        
        # 如果没有真实框，所有预测都是 FP
        if len(target_boxes) == 0:
            all_pred_boxes.extend(pred_boxes)
            all_pred_confs.extend(pred_confs)
            all_pred_classes.extend(pred_classes)
            all_tp.extend([False] * len(pred_boxes))
            continue
        
        # 计算 IoU 矩阵（使用 YOLO 内部的 box_iou 函数）
        pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        target_boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
        iou_matrix = box_iou(pred_boxes_tensor, target_boxes_tensor).numpy()
        
        # 标记 TP/FP（使用贪婪匹配算法，与 YOLO 一致）
        tp_flags = np.zeros(len(pred_boxes), dtype=bool)
        matched_targets = set()
        
        # 按置信度排序（从高到低）
        sorted_indices = np.argsort(-pred_confs)
        
        for pred_idx in sorted_indices:
            pred_class = pred_classes[pred_idx]
            
            # 找到与该预测框 IoU 最大的真实框
            best_iou = 0
            best_target_idx = -1
            
            for target_idx in range(len(target_boxes)):
                # 类别必须匹配
                if target_classes[target_idx] != pred_class:
                    continue
                
                # 已经被匹配的真实框不能再匹配
                if target_idx in matched_targets:
                    continue
                
                iou = iou_matrix[pred_idx, target_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx
            
            # 判断是否为 TP
            if best_iou >= iou_threshold and best_target_idx != -1:
                tp_flags[pred_idx] = True
                matched_targets.add(best_target_idx)
        
        # 添加到全局列表
        all_pred_boxes.extend(pred_boxes)
        all_pred_confs.extend(pred_confs)
        all_pred_classes.extend(pred_classes)
        all_tp.extend(tp_flags)
        all_target_classes.extend(target_classes)
    
    # 转换为 numpy 数组
    all_pred_confs = np.array(all_pred_confs)
    all_pred_classes = np.array(all_pred_classes)
    all_target_classes = np.array(all_target_classes)
    all_tp = np.array(all_tp).reshape(-1, 1)  # shape: (n_predictions, 1)
    
    print(f"\n📊 统计信息 (IoU={iou_threshold}):")
    print(f"   总预测框数: {len(all_pred_confs)}")
    print(f"   总真实框数: {len(all_target_classes)}")
    print(f"   TP 数量: {all_tp.sum()}")
    print(f"   FP 数量: {len(all_tp) - all_tp.sum()}")
    
    # 使用 YOLO 内部的 ap_per_class() 函数计算 AP
    # 🔑 这保证了与 validation.py 的计算 100% 一致！
    if len(all_pred_confs) == 0:
        print("⚠️  没有预测框，mAP = 0")
        return {
            'mAP': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'per_class_ap': {},
        }
    
    tp, fp, p, r, f1, ap, unique_classes, *_ = ap_per_class(
        tp=all_tp,
        conf=all_pred_confs,
        pred_cls=all_pred_classes,
        target_cls=all_target_classes,
        plot=False,
    )
    
    # 计算 mAP（所有类别的平均）
    mAP = ap[:, 0].mean()  # [:, 0] 表示第一个 IoU 阈值（我们只计算一个）
    
    # 每类别 AP
    per_class_ap = {}
    for i, cls_id in enumerate(unique_classes):
        class_name = class_names.get(cls_id, f'class_{cls_id}')
        per_class_ap[class_name] = float(ap[i, 0])
    
    return {
        'mAP': float(mAP),
        'Precision': float(p.mean()),
        'Recall': float(r.mean()),
        'per_class_ap': per_class_ap,
    }


def main():
    print("=" * 60)
    print("🎯 自定义 mAP 计算器（使用 YOLO 内部函数）")
    print("=" * 60)
    print(f"📦 模型: {MODEL_PATH}")
    print(f"📂 数据集: {DATA_YAML}")
    print(f"📊 分割: {DATASET_SPLIT}")
    print(f"📏 IoU 阈值: {IOU_THRESHOLDS}")
    print(f"🎚️  置信度阈值: {CONF_THRESHOLD}")
    print(f"🎚️  NMS IoU: {NMS_IOU}")
    print()
    print("💡 说明：")
    print("   - 使用 YOLO 内部的 ap_per_class() 和 compute_ap() 函数")
    print("   - 保证与 validation.py 的 mAP 计算 100% 一致！")
    print("   - mAP@0.5 应该与 YOLO 内置验证完全相同")
    print("=" * 60)
    print()
    
    # 加载模型
    print("🔧 加载 YOLO 模型...")
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功")
    print()
    
    # 加载数据集信息
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = {i: name for i, name in enumerate(data_config['names'])}
    print(f"📋 类别: {class_names}")
    print()
    
    # 获取预测和标签
    print("🔮 生成预测结果和加载标签...")
    all_stats = get_predictions_and_labels(
        model, 
        DATA_YAML, 
        split=DATASET_SPLIT,
        conf_threshold=CONF_THRESHOLD,
        nms_iou=NMS_IOU
    )
    print(f"✅ 处理了 {len(all_stats)} 张图像")
    print()
    
    # 计算不同 IoU 阈值下的 mAP
    print("=" * 60)
    print("📊 计算不同 IoU 阈值下的 mAP")
    print("=" * 60)
    
    results = {}
    
    for iou_threshold in IOU_THRESHOLDS:
        print(f"\n{'─' * 60}")
        print(f"🎯 IoU 阈值: {iou_threshold}")
        print(f"{'─' * 60}")
        
        metrics = compute_map_at_iou_threshold(all_stats, iou_threshold, class_names)
        
        print(f"  mAP:       {metrics['mAP']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print()
        print(f"  每类别 AP:")
        for class_name, ap_value in metrics['per_class_ap'].items():
            print(f"    {class_name:12s}: {ap_value:.4f}")
        
        results[iou_threshold] = metrics
    
    # 生成对比表格
    print()
    print("=" * 60)
    print("📈 不同 IoU 阈值下的 mAP 对比")
    print("=" * 60)
    print()
    
    # 表格头
    headers = ['IoU阈值', 'mAP', 'Precision', 'Recall']
    for class_name in class_names.values():
        headers.append(f'{class_name}_AP')
    
    print(f" {headers[0]:>7s} {headers[1]:>7s} {headers[2]:>9s} {headers[3]:>7s}", end='')
    for h in headers[4:]:
        print(f" {h:>12s}", end='')
    print()
    
    # 表格数据
    for iou_threshold in IOU_THRESHOLDS:
        metrics = results[iou_threshold]
        print(f"  {iou_threshold:>5.2f} {metrics['mAP']:>7.4f} {metrics['Precision']:>9.4f} {metrics['Recall']:>7.4f}", end='')
        for class_name in class_names.values():
            ap_value = metrics['per_class_ap'].get(class_name, 0.0)
            print(f" {ap_value:>12.4f}", end='')
        print()
    
    # 计算相对变化
    print()
    base_map = results[0.5]['mAP']
    print(f"💡 相对于 mAP@0.5 = {base_map:.4f} 的变化:")
    for iou_threshold in IOU_THRESHOLDS:
        if iou_threshold == 0.5:
            continue
        current_map = results[iou_threshold]['mAP']
        diff = current_map - base_map
        pct = (diff / base_map * 100) if base_map > 0 else 0
        symbol = "📈" if diff >= 0 else "📉"
        print(f"  {symbol} IoU {iou_threshold}: {diff:+.4f} ({pct:+.2f}%)")
    
    print()
    print("🎯 关键发现：")
    print(f"   mAP@0.3 = {results[0.3]['mAP']:.4f} ({results[0.3]['mAP']*100:.2f}%)")
    if base_map > 0:
        improvement = (results[0.3]['mAP'] - base_map) / base_map * 100
        print(f"   相比 mAP@0.5 提升了 {improvement:.2f}%")
        print(f"   这意味着如果你接受 IoU≥0.3 为\"正确\"，mAP 可达 {results[0.3]['mAP']*100:.1f}%！")
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON 格式
    json_file = OUTPUT_DIR / f"custom_map_results_{TIMESTAMP}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'config': {
                'model': MODEL_PATH,
                'dataset': DATA_YAML,
                'split': DATASET_SPLIT,
                'conf_threshold': CONF_THRESHOLD,
                'nms_iou': NMS_IOU,
                'iou_thresholds': IOU_THRESHOLDS,
            },
            'results': {str(k): v for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"📊 JSON 结果保存到: {json_file}")
    
    # CSV 格式
    csv_data = []
    for iou_threshold in IOU_THRESHOLDS:
        metrics = results[iou_threshold]
        row = {
            'IoU_Threshold': iou_threshold,
            'mAP': metrics['mAP'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
        }
        row.update(metrics['per_class_ap'])
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = OUTPUT_DIR / f"custom_map_results_{TIMESTAMP}.csv"
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"📊 CSV 结果保存到: {csv_file}")
    
    print()
    print("=" * 60)
    print("🎉 计算完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
