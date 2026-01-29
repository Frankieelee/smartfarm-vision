"""
YOLO 模型验证脚本 - 在多个数据集上评估
支持：训练集、验证集、测试集的完整评估
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# ============================================================
# 配置区域
# ============================================================

# 模型配置
MODEL_PATH = '/tmp/pycharm_project_990/runs/detect/runs/train/datasets_yolo11m_800_20260128_151115/weights/best.pt'

# 数据集配置（支持多个数据集）
DATASETS = {
    'val': './datasets/data.yaml',        # 验证集（默认）
    'train': './datasets/data.yaml',      # 训练集（检查过拟合）
    # 'test': './path/to/test/data.yaml', # 测试集（如果有）
}

# 验证参数
VAL_ARGS = {
    'imgsz': 640,           # 图像尺寸
    'batch': 8,             # 批次大小
    'conf': 0.25,           # 置信度阈值
    'iou': 0.5,             # NMS IoU 阈值
    'max_det': 300,         # 每张图最大检测数
    'device': 0,            # GPU 设备
    'workers': 6,           # 数据加载线程数
    'save_json': True,      # 保存 JSON 结果
    'save_hybrid': False,   # 保存混合标签
    'verbose': True,        # 详细输出
    'plots': True,          # 生成可视化图表
}

# 输出配置
OUTPUT_DIR = Path('validation_results')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================
# 验证代码
# ============================================================

def validate_on_dataset(model, dataset_name, data_yaml, split='val'):
    """
    在指定数据集上验证模型
    
    Args:
        model: YOLO 模型实例
        dataset_name: 数据集名称（用于保存结果）
        data_yaml: 数据集配置文件路径
        split: 验证的数据集分割 ('train', 'val', 'test')
    """
    print(f"\n{'='*60}")
    print(f"📊 验证数据集: {dataset_name} ({split})")
    print(f"{'='*60}")
    
    # 创建输出目录
    output_subdir = OUTPUT_DIR / f"{dataset_name}_{split}_{TIMESTAMP}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 运行验证
    results = model.val(
        data=data_yaml,
        split=split,              # 指定数据集分割
        project=str(output_subdir.parent),
        name=output_subdir.name,
        exist_ok=True,
        **VAL_ARGS
    )
    
    # 提取关键指标
    # 获取图片数量（从results对象或数据集路径中获取）
    num_images = 0
    
    # 方法1: 从 results.speed 字典中获取
    if hasattr(results, 'speed') and isinstance(results.speed, dict):
        if 'images' in results.speed:
            num_images = results.speed['images']
    
    # 方法2: 从 results 对象的其他属性尝试
    if num_images == 0:
        if hasattr(results, 'seen'):
            num_images = results.seen
        elif hasattr(results.box, 'nc'):
            # 尝试从数据集路径直接读取
            data_path = Path(data_yaml).parent
            if split == 'train':
                img_dir = data_path / 'train' / 'images'
            elif split == 'val':
                img_dir = data_path / 'valid' / 'images'
            else:
                img_dir = data_path / split / 'images'
            
            if img_dir.exists():
                num_images = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + 
                                list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))
    
    metrics = {
        'dataset': dataset_name,
        'split': split,
        'model': str(MODEL_PATH),
        'timestamp': TIMESTAMP,
        'images': num_images,
        'metrics': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        },
        'per_class': {}
    }
    
    # 每个类别的指标
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.names[int(class_idx)]
            metrics['per_class'][class_name] = {
                'AP50': float(results.box.ap50[i]),
                'AP': float(results.box.ap[i]),
            }
    
    # 打印结果
    print(f"\n📈 验证结果:")
    print(f"  图片数量: {metrics['images']}")
    print(f"  mAP50:    {metrics['metrics']['mAP50']:.3f}")
    print(f"  mAP50-95: {metrics['metrics']['mAP50-95']:.3f}")
    print(f"  Precision: {metrics['metrics']['precision']:.3f}")
    print(f"  Recall:    {metrics['metrics']['recall']:.3f}")
    
    if metrics['per_class']:
        print(f"\n  每类别 AP50:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"    {class_name:12s}: {class_metrics['AP50']:.3f}")
    
    print(f"\n💾 结果保存到: {output_subdir}/")
    
    return metrics


def validate_on_splits(model, data_yaml, dataset_name='dataset'):
    """
    在同一数据集的不同分割上验证（train, val, test）
    
    Args:
        model: YOLO 模型实例
        data_yaml: 数据集配置文件路径
        dataset_name: 数据集名称
    """
    all_metrics = []
    
    # 验证不同分割
    for split in ['val', 'train']:  # 通常有 train 和 val，test 视情况而定
        try:
            metrics = validate_on_dataset(model, dataset_name, data_yaml, split)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"⚠️  验证 {split} 分割失败: {e}")
            import traceback
            traceback.print_exc()
    
    return all_metrics


def main():
    """主函数：加载模型并在所有数据集上验证"""
    print(f"{'='*60}")
    print(f"🚀 YOLO 模型验证")
    print(f"{'='*60}")
    print(f"📦 模型: {MODEL_PATH}")
    print(f"📂 输出目录: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("🔧 加载模型...")
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型加载成功")
    print(f"   模型类型: {model.model.__class__.__name__}")
    print(f"   类别数量: {len(model.names)}")
    print(f"   类别: {model.names}\n")
    
    # 在所有数据集上验证
    all_results = []
    
    # 方式1: 在多个数据集/分割上验证
    for dataset_name, data_yaml in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"🔄 处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # 根据 dataset_name 判断是验证哪个分割
            if dataset_name in ['train', 'val', 'test']:
                split = dataset_name
                actual_name = Path(data_yaml).parent.name
            else:
                split = 'val'
                actual_name = dataset_name
            
            metrics = validate_on_dataset(model, actual_name, data_yaml, split=split)
            all_results.append(metrics)
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 方式2: 在同一数据集的不同分割上验证（可选）
    # 取消注释下面的代码来同时在 train 和 val 上验证
    # print(f"\n{'='*60}")
    # print(f"🔄 在所有数据分割上验证")
    # print(f"{'='*60}")
    # split_results = validate_on_splits(model, './datasets/data.yaml', 'seedTrueLeaf')
    # all_results.extend(split_results)
    
    # 保存汇总结果
    summary_file = OUTPUT_DIR / f"validation_summary_{TIMESTAMP}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 汇总结果保存到: {summary_file}")
    
    # 创建汇总表格
    if all_results:
        df_data = []
        for result in all_results:
            row = {
                '数据集': result['dataset'],
                '分割': result['split'],
                '图片数': result['images'],
                'mAP50': f"{result['metrics']['mAP50']:.3f}",
                'mAP50-95': f"{result['metrics']['mAP50-95']:.3f}",
                'Precision': f"{result['metrics']['precision']:.3f}",
                'Recall': f"{result['metrics']['recall']:.3f}",
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = OUTPUT_DIR / f"validation_summary_{TIMESTAMP}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"\n📋 汇总表格:")
        print(df.to_string(index=False))
        print(f"\n💾 CSV 保存到: {csv_file}")
    
    print(f"\n{'='*60}")
    print(f"🎉 所有验证完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
