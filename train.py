"""
YOLO11 训练脚本 - 针对密集小目标优化 + CBAM 注意力
数据集：seedTrueLeaf (58张训练图，4张验证图，200+个小目标/图)
优化目标：从 24% mAP50 提升到 60-75%
"""

from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # ========== 生成实验名称 ==========
    data_path = '/root/autodl-tmp/seedTrue4i/data.yaml'
    dataset_name = Path(data_path).parent.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_path = 'ultralytics/cfg/models/sf/yolo11n_cbam.yaml'
    model_name = os.path.basename(model_name_path).replace('.yaml', '')
    experiment_name = f"{dataset_name}_{model_name}_640_{timestamp}"
    
    print(f"📁 实验名称: {experiment_name}")
    print(f"📂 保存路径: runs/train/{experiment_name}/")
    print("="*60 + "\n")
    
    # ========== 模型初始化（带预训练权重迁移）==========
    print("🔧 初始化模型...")
    
    # 1. 创建新模型（带 CBAM）
    model = YOLO(model_name_path)
    
    # 2. 加载预训练权重（部分迁移）
    pretrained_path = 'yolo11n.pt'  # 官方预训练权重
    
    if os.path.exists(pretrained_path):
        print(f"📥 加载预训练权重: {pretrained_path}")
        
        # 加载预训练权重
        # pretrained = torch.load(pretrained_path, map_location='cpu')
        pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        pretrained_state = pretrained['model'].state_dict() if 'model' in pretrained else pretrained
        
        # 获取当前模型的 state_dict
        model_state = model.model.state_dict()
        
        # 过滤并加载兼容的权重
        compatible_state = {}
        incompatible_keys = []
        
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                incompatible_keys.append(k)
        
        # 加载兼容的权重
        model.model.load_state_dict(compatible_state, strict=False)
        
        print(f"✅ 成功加载 {len(compatible_state)}/{len(pretrained_state)} 个权重")
        print(f"⚠️  跳过 {len(incompatible_keys)} 个不兼容的权重（CBAM 层将随机初始化）")
        
        if len(incompatible_keys) <= 10:
            print(f"   跳过的层: {incompatible_keys}")
    else:
        print(f"⚠️  未找到预训练权重: {pretrained_path}")
        print("   将从头开始训练（随机初始化）")
    
    print("="*60 + "\n")

    # ========== 开始训练 ==========
    results = model.train(
        # ========== 基础配置 ==========
        data=data_path,
        epochs=150,
        patience=50,

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
        lr0=0.001,  # 使用预训练权重可以用稍高的学习率
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
    print("🎉 训练完成！")
    print("=" * 60)
    print(f"📁 输出目录: {output_dir}/")
    print(f"🏆 最佳权重: {output_dir}/weights/best.pt")
    print(f"📊 最终权重: {output_dir}/weights/last.pt")
    print(f"📈 训练曲线: {output_dir}/results.png")
    print(f"📋 训练日志: {output_dir}/results.csv")
    print("-" * 60)
    print(f"📊 最终 mAP50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"📊 最终 mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print("=" * 60)
