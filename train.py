"""
YOLO11 训练脚本 - 针对密集小目标优化
数据集：seedTrueLeaf (58张训练图，4张验证图，200+个小目标/图)
优化目标：从 24% mAP50 提升到 60-75%.
"""

import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    # ========== 生成实验名称 ==========
    # 数据集路径
    data_path = "./datasets/seedTrueLeaf.v4i.yolov11/data.yaml"

    # 提取数据集名称（去掉路径和 .yaml）
    dataset_name = Path(data_path).parent.name  # 例如：seedTrueLeaf.v4i.yolov11

    # 生成时间戳（格式：年月日_时分秒）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 组合实验名称：数据集名字_模型_分辨率_时间
    experiment_name = f"{dataset_name}_yolo11n_1280_{timestamp}"

    print(f"📁 实验名称: {experiment_name}")
    print(f"📂 保存路径: runs/train/{experiment_name}/")
    print("=" * 60 + "\n")

    # ========== 模型初始化 ==========
    # 直接加载预训练模型（修复之前的bug）
    model = YOLO("yolo11n.pt")

    # ========== 开始训练 ==========
    results = model.train(
        # ========== 基础配置 ==========
        data=data_path,
        epochs=2,  # 训练150轮
        patience=50,  # 50轮不提升才早停
        # ========== Batch 与图像尺寸 ==========
        batch=-1,  # 自动batch（使用60% GPU显存）
        imgsz=640,  # 高分辨率检测小目标
        # ========== 设备配置 ==========
        device=0,  # 使用GPU 0
        workers=8,  # 数据加载线程数
        # ========== 性能优化 ==========
        cache="ram",  # 缓存到内存（小数据集推荐）
        amp=False,  # 关闭AMP（避免bug）
        # ========== 优化器配置 ==========
        optimizer="AdamW",  # AdamW优化器（小数据集推荐）
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率 = lr0 * lrf
        momentum=0.937,  # SGD动量
        weight_decay=0.0005,  # L2正则化
        # ========== 学习率预热 ==========
        warmup_epochs=3.0,  # 前3轮预热
        warmup_momentum=0.8,  # 预热期动量
        warmup_bias_lr=0.1,  # 预热期bias学习率
        cos_lr=True,  # 余弦学习率衰减
        # ========== 损失函数权重（针对小目标）==========
        box=7.5,  # box loss权重
        cls=0.3,  # cls loss权重（降低，只有2类）
        dfl=2.0,  # DFL loss权重（提高，精确边界）
        nbs=64,  # 标称batch size
        # ========== 数据增强（针对小目标优化）==========
        mosaic=1.0,  # Mosaic拼接增强
        mixup=0.1,  # MixUp混合增强
        copy_paste=0.2,  # Copy-Paste增强（增加少数类）
        multi_scale=0.5,  # 多尺度训练（0.5x-1.5x范围）
        # 几何变换（保护小目标）
        degrees=10.0,  # 随机旋转±10度
        translate=0.05,  # 平移5%（降低，避免小目标移出）
        scale=0.3,  # 缩放±30%（降低，保护小目标）
        fliplr=0.5,  # 50%概率左右翻转
        flipud=0.5,  # 50%概率上下翻转
        perspective=0.0,  # 透视变换（俯拍设为0）
        # 颜色变换
        hsv_h=0.015,  # 色调抖动
        hsv_s=0.7,  # 饱和度抖动
        hsv_v=0.4,  # 亮度抖动
        # ========== 高级设置 ==========
        close_mosaic=10,  # 最后10轮关闭mosaic精细化训练
        rect=False,  # 不使用矩形训练
        # ========== 验证与保存 ==========
        val=True,  # 每轮验证
        save=True,  # 保存checkpoint
        save_period=10,  # 每10轮保存一次
        plots=True,  # 生成训练曲线图
        # ========== 可重复性 ==========
        seed=42,  # 固定随机种子
        deterministic=True,  # 确定性训练
        verbose=True,  # 显示详细信息
        # ========== 项目管理 ==========
        project="runs/train",
        name=experiment_name,  # 使用自定义名称：数据集_模型_分辨率_时间
        exist_ok=False,  # 不允许覆盖（因为每次都是新的时间戳）
    )

    # ========== 训练完成后的信息 ==========
    output_dir = Path("runs/train") / experiment_name
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
