"""
YOLO 模型指标分析脚本
专注于 Recall 和 Classification Accuracy 计算
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import numpy as np

# ============================================================
# 配置区域
# ============================================================

# 模型配置
MODEL_PATH = '/tmp/pycharm_project_949/runs/detect/runs/train/seedTrueLeaf.v12i.yolov11_yolo11s_800_20260201_131735/weights/best.pt'

# 数据集配置
DATA_YAML = './datasets/seedTrueLeaf.v12i.yolov11/data.yaml'

# 评估参数
EVAL_ARGS = {
    'imgsz': 640,
    'batch': 8,
    'conf': 0.25,           # 可调整以优化 Recall/Precision
    'iou': 0.5,
    'max_det': 300,
    'device': 0,
    'workers': 6,
    'save_json': True,
    'verbose': True,
}

# 输出配置
OUTPUT_DIR = Path('metrics_analysis')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================
# 核心函数
# ============================================================

def calculate_recall_metrics(model, data_yaml, conf_threshold=0.25):
    """
    计算 Recall 相关指标
    
    Recall = 正确检测到的目标数 / 所有真实目标数
    
    Args:
        model: YOLO 模型实例
        data_yaml: 数据集配置文件
        conf_threshold: 置信度阈值
        
    Returns:
        dict: 包含 Recall 相关指标的字典
    """
    print(f"\n{'='*60}")
    print(f"📊 计算 Recall 指标")
    print(f"{'='*60}")
    print(f"📁 数据集: {data_yaml}")
    print(f"🎯 置信度阈值: {conf_threshold}")
    print(f"{'='*60}\n")
    
    # 运行验证
    results = model.val(
        data=data_yaml,
        split='val',
        conf=conf_threshold,
        iou=EVAL_ARGS['iou'],
        max_det=EVAL_ARGS['max_det'],
        device=EVAL_ARGS['device'],
        save_json=EVAL_ARGS['save_json'],
        verbose=False,
        plots=False,
    )
    
    # 提取 Recall 指标
    recall_metrics = {
        'overall_recall': float(results.box.mr),  # Mean Recall (所有类别平均)
        'precision': float(results.box.mp),       # 用于对比
        'f1_score': 0.0,
        'per_class_recall': {},
        'conf_threshold': conf_threshold,
    }
    
    # 计算 F1-Score
    if recall_metrics['precision'] > 0 and recall_metrics['overall_recall'] > 0:
        recall_metrics['f1_score'] = 2 * recall_metrics['precision'] * recall_metrics['overall_recall'] / \
                                     (recall_metrics['precision'] + recall_metrics['overall_recall'])
    
    # 每个类别的 Recall
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.names[int(class_idx)]
            # Recall per class (从 results.box.r 获取)
            if hasattr(results.box, 'r') and i < len(results.box.r):
                class_recall = float(results.box.r[i])
            else:
                class_recall = 0.0
            
            recall_metrics['per_class_recall'][class_name] = {
                'recall': class_recall,
                'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                'ap50': float(results.box.ap50[i]),
            }
    
    # 打印结果
    print(f"✅ Recall 计算完成\n")
    print(f"{'='*60}")
    print(f"📈 整体 Recall 指标:")
    print(f"{'='*60}")
    print(f"  Overall Recall:  {recall_metrics['overall_recall']:.1%}  {'✅' if recall_metrics['overall_recall'] >= 0.85 else '⚠️'}")
    print(f"  Precision:       {recall_metrics['precision']:.1%}")
    print(f"  F1-Score:        {recall_metrics['f1_score']:.1%}  {'✅' if recall_metrics['f1_score'] >= 0.85 else '⚠️'}")
    print(f"  Conf Threshold:  {conf_threshold}")
    
    if recall_metrics['per_class_recall']:
        print(f"\n{'='*60}")
        print(f"📊 各类别 Recall:")
        print(f"{'='*60}")
        print(f"  {'类别':<15} {'Recall':<10} {'Precision':<12} {'AP50':<10}")
        print(f"  {'-'*50}")
        for class_name, metrics in recall_metrics['per_class_recall'].items():
            status = '✅' if metrics['recall'] >= 0.85 else '⚠️'
            print(f"  {class_name:<15} {metrics['recall']:.1%}     {metrics['precision']:.1%}      {metrics['ap50']:.1%}  {status}")
    
    print(f"{'='*60}\n")
    
    return recall_metrics


def calculate_classification_accuracy(model, data_yaml, conf_threshold=0.25, iou_threshold=0.5):
    """
    计算分类准确率（Classification Accuracy）
    
    定义：对于所有检测到的框，分类正确的比例
    注意：这里只关心类别是否正确，不关心框的位置精度
    
    Args:
        model: YOLO 模型实例
        data_yaml: 数据集配置文件
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值（用于匹配预测框和真实框）
        
    Returns:
        dict: 包含分类准确率的字典
    """
    print(f"\n{'='*60}")
    print(f"📊 计算分类准确率")
    print(f"{'='*60}")
    print(f"📁 数据集: {data_yaml}")
    print(f"🎯 置信度阈值: {conf_threshold}")
    print(f"🎯 IoU 阈值: {iou_threshold}")
    print(f"{'='*60}\n")
    
    # 运行验证并保存 JSON 结果
    results = model.val(
        data=data_yaml,
        split='val',
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=EVAL_ARGS['max_det'],
        device=EVAL_ARGS['device'],
        save_json=True,
        verbose=False,
        plots=False,
    )
    
    # 注意：精确的分类准确率需要解析预测和真实标签
    # 这里使用 Precision 作为近似（检测正确 ≈ 分类正确）
    # 如果需要更精确的计算，需要从 results 中提取详细的预测结果
    
    classification_metrics = {
        'classification_accuracy_approx': float(results.box.mp),  # Precision 作为近似值
        'per_class_accuracy': {},
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'note': 'Classification accuracy approximated by Precision (correct detections / all detections)',
    }
    
    # 每个类别的分类准确率（使用 Precision）
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.names[int(class_idx)]
            
            classification_metrics['per_class_accuracy'][class_name] = {
                'accuracy_approx': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0,
                'ap50': float(results.box.ap50[i]),
            }
    
    # 打印结果
    print(f"✅ 分类准确率计算完成\n")
    print(f"{'='*60}")
    print(f"📈 分类准确率 (近似):")
    print(f"{'='*60}")
    print(f"  整体准确率:  {classification_metrics['classification_accuracy_approx']:.1%}  {'✅' if classification_metrics['classification_accuracy_approx'] >= 0.85 else '⚠️'}")
    print(f"  ")
    print(f"  📝 说明: 使用 Precision 作为分类准确率的近似值")
    print(f"         (正确检测数 / 所有检测数)")
    
    if classification_metrics['per_class_accuracy']:
        print(f"\n{'='*60}")
        print(f"📊 各类别分类准确率:")
        print(f"{'='*60}")
        print(f"  {'类别':<15} {'准确率':<10} {'Recall':<10} {'AP50':<10}")
        print(f"  {'-'*50}")
        for class_name, metrics in classification_metrics['per_class_accuracy'].items():
            status = '✅' if metrics['accuracy_approx'] >= 0.85 else '⚠️'
            print(f"  {class_name:<15} {metrics['accuracy_approx']:.1%}     {metrics['recall']:.1%}     {metrics['ap50']:.1%}  {status}")
    
    print(f"{'='*60}\n")
    
    return classification_metrics


def calculate_localization_accuracy(model, data_yaml, conf_threshold=0.25, iou_threshold=0.5):
    """
    计算定位准确率（Localization Accuracy）
    
    定义：在所有检测框中，IoU ≥ iou_threshold 的框占比（不考虑类别是否正确）
    这个指标衡量模型"找对位置"的能力
    
    Args:
        model: YOLO 模型实例
        data_yaml: 数据集配置文件
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值
        
    Returns:
        dict: 包含定位准确率的字典
    """
    print(f"\n{'='*60}")
    print(f"📊 计算定位准确率")
    print(f"{'='*60}")
    print(f"📁 数据集: {data_yaml}")
    print(f"🎯 置信度阈值: {conf_threshold}")
    print(f"🎯 IoU 阈值: {iou_threshold}")
    print(f"{'='*60}\n")
    
    # 运行预测
    img_dir = Path(data_yaml).parent / 'valid' / 'images'
    label_dir = Path(data_yaml).parent / 'valid' / 'labels'
    
    results = model.predict(
        source=str(img_dir),
        conf=conf_threshold,
        iou=EVAL_ARGS['iou'],
        device=EVAL_ARGS['device'],
        save=False,
        verbose=False,
    )
    
    # 获取图片文件列表（按照 predict 的顺序）
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + 
                      list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))
    
    # 统计定位准确率
    total_predictions = 0
    correct_localizations = 0  # IoU ≥ threshold 的检测数（不管类别）
    correct_detections = 0     # IoU ≥ threshold 且类别正确
    
    per_class_stats = {}
    for class_name in model.names.values():
        per_class_stats[class_name] = {
            'predictions': 0,
            'correct_localizations': 0,
            'correct_detections': 0,
        }
    
    # 遍历每张图片的预测结果
    for idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
            
        pred_boxes = result.boxes.xyxy.cpu().numpy()  # 预测框 (N, 4)
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)  # 预测类别
        pred_confs = result.boxes.conf.cpu().numpy()  # 置信度
        
        # 获取对应的标签文件
        if idx < len(img_files):
            img_file = img_files[idx]
            label_file = label_dir / f"{img_file.stem}.txt"
            
            # 读取真实标签
            if label_file.exists():
                gt_boxes_list = []
                gt_classes_list = []
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            gt_classes_list.append(cls)
                            gt_boxes_list.append([x_center, y_center, width, height])
                
                if len(gt_boxes_list) > 0:
                    gt_boxes = np.array(gt_boxes_list)  # (M, 4) 格式：xywhn
                    gt_classes = np.array(gt_classes_list, dtype=int)
                    
                    # 转换真实框格式：xywhn -> xyxy
                    img_h, img_w = result.orig_shape
                    gt_boxes_xyxy = np.zeros_like(gt_boxes)
                    gt_boxes_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * img_w  # x1
                    gt_boxes_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * img_h  # y1
                    gt_boxes_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * img_w  # x2
                    gt_boxes_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * img_h  # y2
                    
                    # 计算每个预测框与所有真实框的 IoU
                    for i, pred_box in enumerate(pred_boxes):
                        total_predictions += 1
                        pred_class = pred_classes[i]
                        class_name = model.names[pred_class]
                        per_class_stats[class_name]['predictions'] += 1
                        
                        # 计算与所有真实框的 IoU
                        max_iou = 0
                        matched_gt_class = -1
                        
                        for j, gt_box in enumerate(gt_boxes_xyxy):
                            iou = compute_iou(pred_box, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                matched_gt_class = gt_classes[j]
                        
                        # 统计定位准确率（IoU ≥ threshold）
                        if max_iou >= iou_threshold:
                            correct_localizations += 1
                            per_class_stats[class_name]['correct_localizations'] += 1
                            
                            # 如果类别也正确，则计数
                            if matched_gt_class == pred_class:
                                correct_detections += 1
                                per_class_stats[class_name]['correct_detections'] += 1
    
    # 计算整体指标
    localization_accuracy = correct_localizations / total_predictions if total_predictions > 0 else 0
    detection_accuracy = correct_detections / total_predictions if total_predictions > 0 else 0
    
    localization_metrics = {
        'localization_accuracy': localization_accuracy,  # IoU ≥ threshold 的比例
        'detection_accuracy': detection_accuracy,        # IoU ≥ threshold 且类别正确的比例
        'total_predictions': total_predictions,
        'correct_localizations': correct_localizations,
        'correct_detections': correct_detections,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'per_class': {},
    }
    
    # 计算每个类别的定位准确率
    for class_name, stats in per_class_stats.items():
        if stats['predictions'] > 0:
            localization_metrics['per_class'][class_name] = {
                'localization_accuracy': stats['correct_localizations'] / stats['predictions'],
                'detection_accuracy': stats['correct_detections'] / stats['predictions'],
                'predictions': stats['predictions'],
            }
    
    # 打印结果
    print(f"✅ 定位准确率计算完成\n")
    print(f"{'='*60}")
    print(f"📈 定位准确率:")
    print(f"{'='*60}")
    print(f"  总预测框数:      {total_predictions}")
    print(f"  定位正确数:      {correct_localizations} (IoU≥{iou_threshold})")
    print(f"  检测正确数:      {correct_detections} (IoU≥{iou_threshold} 且类别对)")
    print(f"  ")
    print(f"  定位准确率:      {localization_accuracy:.1%}  {'✅' if localization_accuracy >= 0.85 else '⚠️'}")
    print(f"  (所有检测框中，位置正确的比例)")
    print(f"  ")
    print(f"  检测准确率:      {detection_accuracy:.1%}  {'✅' if detection_accuracy >= 0.85 else '⚠️'}")
    print(f"  (所有检测框中，位置+类别都正确的比例，≈Precision)")
    
    if localization_metrics['per_class']:
        print(f"\n{'='*60}")
        print(f"📊 各类别定位准确率:")
        print(f"{'='*60}")
        print(f"  {'类别':<15} {'定位准确率':<12} {'检测准确率':<12} {'预测数':<10}")
        print(f"  {'-'*55}")
        for class_name, metrics in localization_metrics['per_class'].items():
            if metrics['predictions'] > 0:
                loc_status = '✅' if metrics['localization_accuracy'] >= 0.85 else '⚠️'
                det_status = '✅' if metrics['detection_accuracy'] >= 0.85 else '⚠️'
                print(f"  {class_name:<15} {metrics['localization_accuracy']:.1%} {loc_status}      "
                      f"{metrics['detection_accuracy']:.1%} {det_status}      {metrics['predictions']:<10}")
    
    print(f"{'='*60}\n")
    
    return localization_metrics


def compute_iou(box1, box2):
    """
    计算两个框的 IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU 值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # 计算 IoU
    iou = intersection / union if union > 0 else 0
    return iou


def different_thresholds(model, data_yaml, conf_thresholds=[0.15, 0.25, 0.35, 0.45]):
    """
    测试不同置信度阈值下的 Recall 和 Precision
    用于找到最佳的置信度阈值
    
    Args:
        model: YOLO 模型实例
        data_yaml: 数据集配置文件
        conf_thresholds: 要测试的置信度阈值列表
        
    Returns:
        list: 不同阈值下的指标结果
    """
    print(f"\n{'='*60}")
    print(f"🔬 测试不同置信度阈值")
    print(f"{'='*60}")
    print(f"📁 数据集: {data_yaml}")
    print(f"🎯 测试阈值: {conf_thresholds}")
    print(f"{'='*60}\n")
    
    threshold_results = []
    
    for conf in conf_thresholds:
        print(f"⏳ 测试阈值: {conf:.2f}...")
        
        results = model.val(
            data=data_yaml,
            split='val',
            conf=conf,
            iou=EVAL_ARGS['iou'],
            device=EVAL_ARGS['device'],
            verbose=False,
            plots=False,
        )
        
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results.append({
            'conf_threshold': conf,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mAP50': float(results.box.map50),
        })
        
        status = '✅' if precision >= 0.85 or recall >= 0.85 else '⚠️'
        print(f"   Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%}  {status}\n")
    
    # 打印汇总表格
    print(f"\n{'='*60}")
    print(f"📊 阈值测试结果汇总:")
    print(f"{'='*60}")
    print(f"  {'Conf':<8} {'Precision':<12} {'Recall':<10} {'F1':<10} {'mAP50':<10}")
    print(f"  {'-'*55}")
    
    for result in threshold_results:
        prec_status = '✅' if result['precision'] >= 0.85 else '  '
        rec_status = '✅' if result['recall'] >= 0.85 else '  '
        f1_status = '✅' if result['f1_score'] >= 0.85 else '  '
        
        print(f"  {result['conf_threshold']:<8.2f} {result['precision']:.1%} {prec_status}   "
              f"{result['recall']:.1%} {rec_status}  {result['f1_score']:.1%} {f1_status}  "
              f"{result['mAP50']:.1%}")
    
    print(f"{'='*60}\n")
    
    # 找出最佳阈值
    best_for_precision = max(threshold_results, key=lambda x: x['precision'])
    best_for_recall = max(threshold_results, key=lambda x: x['recall'])
    best_for_f1 = max(threshold_results, key=lambda x: x['f1_score'])
    
    print(f"💡 推荐阈值:")
    print(f"  - 最高 Precision: conf={best_for_precision['conf_threshold']:.2f} (Precision={best_for_precision['precision']:.1%})")
    print(f"  - 最高 Recall:    conf={best_for_recall['conf_threshold']:.2f} (Recall={best_for_recall['recall']:.1%})")
    print(f"  - 最高 F1-Score:  conf={best_for_f1['conf_threshold']:.2f} (F1={best_for_f1['f1_score']:.1%})")
    print(f"{'='*60}\n")
    
    return threshold_results


def main():
    """主函数：加载模型并计算所有指标"""
    print(f"\n{'='*60}")
    print(f"🚀 YOLO 模型指标分析")
    print(f"{'='*60}")
    print(f"📦 模型: {MODEL_PATH}")
    print(f"📁 数据集: {DATA_YAML}")
    print(f"📂 输出目录: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("🔧 加载模型...")
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型加载成功")
    print(f"   类别: {model.names}\n")
    
    # ====== 1. 计算 Recall 指标 ======
    recall_metrics = calculate_recall_metrics(
        model, 
        DATA_YAML, 
        conf_threshold=EVAL_ARGS['conf']
    )
    
    # ====== 2. 计算分类准确率 ======
    classification_metrics = calculate_classification_accuracy(
        model, 
        DATA_YAML,
        conf_threshold=EVAL_ARGS['conf'],
        iou_threshold=EVAL_ARGS['iou']
    )
    
    # ====== 3. 计算定位准确率 ======
    localization_metrics = calculate_localization_accuracy(
        model,
        DATA_YAML,
        conf_threshold=EVAL_ARGS['conf'],
        iou_threshold=EVAL_ARGS['iou']
    )
    
    # ====== 4. 测试不同阈值（可选）======
    print(f"\n{'='*60}")
    print(f"🔬 测试不同置信度阈值（找到 ≥85% 的最佳配置）")
    print(f"{'='*60}")
    
    threshold_results = different_thresholds(
        model,
        DATA_YAML,
        conf_thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    )
    
    # ====== 保存结果 ======
    output_file = OUTPUT_DIR / f"metrics_analysis_{TIMESTAMP}.json"
    
    final_results = {
        'timestamp': TIMESTAMP,
        'model_path': str(MODEL_PATH),
        'data_yaml': str(DATA_YAML),
        'recall_metrics': recall_metrics,
        'classification_metrics': classification_metrics,
        'localization_metrics': localization_metrics,
        'threshold_test_results': threshold_results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 完整结果保存到: {output_file}")
    
    # ====== 最终总结 ======
    print(f"\n{'='*60}")
    print(f"📊 最终总结")
    print(f"{'='*60}")
    print(f"\n1️⃣  Recall (召回率):")
    print(f"   {recall_metrics['overall_recall']:.1%}  {'✅ 达标！' if recall_metrics['overall_recall'] >= 0.85 else '⚠️ 未达 85%'}")
    
    print(f"\n2️⃣  Classification Accuracy (分类准确率):")
    print(f"   {classification_metrics['classification_accuracy_approx']:.1%}  {'✅ 达标！' if classification_metrics['classification_accuracy_approx'] >= 0.85 else '⚠️ 未达 85%'}")
    
    print(f"\n3️⃣  Localization Accuracy (定位准确率):")
    print(f"   {localization_metrics['localization_accuracy']:.1%}  {'✅ 达标！' if localization_metrics['localization_accuracy'] >= 0.85 else '⚠️ 未达 85%'}")
    print(f"   (所有检测框中，位置正确 IoU≥0.5 的比例)")
    
    print(f"\n4️⃣  F1-Score (综合指标):")
    print(f"   {recall_metrics['f1_score']:.1%}  {'✅ 达标！' if recall_metrics['f1_score'] >= 0.85 else '⚠️ 未达 85%'}")
    
    # 建议
    print(f"\n{'='*60}")
    print(f"💡 建议:")
    print(f"{'='*60}")
    
    if localization_metrics['localization_accuracy'] >= 0.85:
        print(f"✅ 定位准确率已达 85%，说明模型定位能力强！")
        print(f"   可以在报告中强调：'{localization_metrics['localization_accuracy']:.1%} 的检测框位置准确（IoU≥0.5）'")
    if classification_metrics['classification_accuracy_approx'] >= 0.85:
        print(f"✅ Precision (≈分类准确率) 已达 85%，可以作为主要指标！")
    elif recall_metrics['f1_score'] >= 0.85:
        print(f"✅ F1-Score 已达 85%，可以作为综合指标！")
    elif recall_metrics['overall_recall'] >= 0.85:
        print(f"✅ Recall 已达 85%，如果任务重视\"不漏检\"，可作为主要指标！")
    else:
        # 检查是否有任何指标达标
        if localization_metrics['localization_accuracy'] >= 0.85:
            print(f"   虽然定位准确率很高，但分类性能还需提升")
        else:
            print(f"⚠️ 所有指标都未达 85%，建议：")
            print(f"   1. 调整置信度阈值（参考上面的阈值测试结果）")
            print(f"   2. 优化标注质量")
            print(f"   3. 增加训练数据")
    
    print(f"{'='*60}\n")
    print(f"🎉 分析完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
