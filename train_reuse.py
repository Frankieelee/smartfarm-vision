"""
YOLO11 继续训练脚本 - 基于已训练的 best.pt
"""

from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # ========== 配置 ==========
    # 之前训练的最佳权重
    pretrained_best = '/root/autodl-tmp/sf-vision/smartfarm-vision/runs/detect/runs/train/seedTrue4i_yolo11n_cbam_640_20260127_123449/weights/best.pt'
    
    data_path = '/root/autodl-tmp/seedTrue4i/data.yaml'
    
    # 从 pretrained_best 路径中提取原始实验名称
    pretrained_path = Path(pretrained_best)
    original_experiment_name = pretrained_path.parent.parent.name  # 获取 seedTrue4i_yolo11n_cbam_640_20260127_123449
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{original_experiment_name}_continue_{timestamp}"
    
    print(f"📁 原始实验: {original_experiment_name}")
    print(f"📁 新实验名称: {experiment_name}")
    print(f"📂 保存路径: runs/train/{experiment_name}/")
    print(f"🔄 继续训练自: {pretrained_best}")
    print("="*60 + "\n")
    
    # ========== 加载已训练的模型 ==========
    print("🔧 加载已训练的模型...")
    model = YOLO(pretrained_best)
    print("✅ 模型加载成功！")
    print("="*60 + "\n")

    # ========== 继续训练 ==========
    results = model.train(
        # ========== 基础配置 ==========
        data=data_path,
        epochs=1000,  # 额外训练 1000 个 epoch
        patience=150,

        # ========== Batch 与图像尺寸 ==========
        batch=8,
        imgsz=640,

        # ========== 设备配置 ==========
        device=0,
        workers=6,

        # ========== 性能优化 ==========
        cache='ram',
        amp=False,

        # ========== 优化器配置 ==========
        optimizer='AdamW',
        lr0=0.0005,  # 继续训练时使用更小的学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # ========== 学习率预热 ==========
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,

        # ========== 损失函数权重 ==========
        box=7.5,
        cls=0.3,
        dfl=2.0,
        nbs=64,

        # ========== 数据增强 ==========
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.2,
        multi_scale=0.5,

        degrees=10.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        perspective=0.0,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ========== 高级设置 ==========
        close_mosaic=10,
        rect=False,

        # ========== 验证与保存 ==========
        val=True,
        save=True,
        save_period=10,
        plots=True,

        # ========== 可重复性 ==========
        seed=42,
        deterministic=True,
        verbose=True,

        # ========== 项目管理 ==========
        project='runs/train',
        name=experiment_name,
        exist_ok=False,
    )

    # ========== 训练完成后的信息 ==========
    output_dir = Path('runs/train') / experiment_name
    print("\n" + "=" * 60)
    print("🎉 继续训练完成！")
    print("=" * 60)
    print(f"📁 原始实验: {original_experiment_name}")
    print(f"📁 新实验目录: {output_dir}/")
    print(f"🏆 最佳权重: {output_dir}/weights/best.pt")
    print(f"📊 最终权重: {output_dir}/weights/last.pt")
    print(f"📈 训练曲线: {output_dir}/results.png")
    print(f"📋 训练日志: {output_dir}/results.csv")
    print("-" * 60)
    print(f"📊 最终 mAP50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"📊 最终 mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print("=" * 60)
