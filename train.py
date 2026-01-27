"""
YOLO11 训练脚本 - 针对密集小目标优化
支持标准模型和CBAM注意力增强
"""

from ultralytics import YOLO
import os
import sys
from datetime import datetime
from pathlib import Path
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================
# 日志保存工具类
# ============================================================

class Logger:
    """同时输出到控制台和文件的日志工具"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 实时写入文件
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# ============================================================
# 🔧 配置区域 - 修改这里的参数
# ============================================================

# ========== 模型配置 ==========
# 支持三种训练模式：
# 1. 'scratch'    - 从头训练：使用yaml，随机初始化（适合：测试新架构）
# 2. 'pretrained' - 预训练模型：直接用.pt（适合：标准YOLO训练）
# 3. 'custom'     - 自定义+迁移：yaml+预训练权重（适合：改架构+迁移学习）

MODEL_CONFIG = {
    'type': 'custom',
    'path': 'ultralytics/cfg/models/sf/yolo11n_cbam.yaml',
    'pretrained': 'yolo11n.pt',        # 官方预训练权重
}

# ========== 快速切换示例 ==========
# 
# 📌 模式1️⃣: 从头训练（learn from scratch）
#    用途：测试新架构、不需要迁移学习时使用
#    MODEL_CONFIG = {
#        'type': 'scratch',
#        'path': 'ultralytics/cfg/models/sf/yolo26_p2_cbam.yaml',
#    }
# 
# 📌 模式2️⃣: 使用预训练模型
#    用途：使用官方权重或之前训练好的完整模型
#    MODEL_CONFIG = {
#        'type': 'pretrained',
#        'path': 'yolo26n.pt',              # 官方预训练
#        # 或 'path': 'runs/train/exp1/weights/best.pt'  # 自己的训练结果
#    }
# 
# 📌 模式3️⃣: 自定义架构 + 迁移学习
#    用途：修改架构（如添加CBAM）+ 加载预训练权重做迁移学习
#    MODEL_CONFIG = {
#        'type': 'custom',
#        'path': 'ultralytics/cfg/models/sf/yolo26_p2_cbam.yaml',
#        'pretrained': 'yolo26n.pt',        # 官方预训练权重
#        # 或 'pretrained': 'runs/train/exp1/weights/epoch800.pt'  # 之前的checkpoint
#    }

# ========== 数据配置 ==========
DATA_PATH = '/root/autodl-tmp/seedTure7i/data.yaml'

# ========== 训练参数 ==========
TRAIN_ARGS = {
    # ========== 基础训练配置 ==========
    # 'resume': "/path/to/last.pt",  # 恢复训练：从中断的训练继续（包含优化器状态和epoch）
    'epochs': 2000,                   # 训练轮数：完整遍历数据集的次数
    'patience': 200,                  # 早停耐心值：多少轮验证指标不提升就停止训练
    'batch': 8,                       # 批次大小：每次训练使用的图片数量（受GPU显存限制）
    'imgsz': 640,                     # 输入图像尺寸：训练时图片会被缩放到此大小
    
    # ========== 设备与性能 ==========
    'device': 0,                      # GPU设备：0表示第一块GPU，'cpu'表示使用CPU
    'workers': 6,                     # 数据加载线程数：并行加载数据（通常设为CPU核心数）
    'cache': 'ram',                   # 数据缓存：'ram'缓存到内存，'disk'缓存到硬盘，False不缓存
    'amp': False,                     # 混合精度训练：True可节省显存但可能影响精度
    
    # ========== 优化器配置 ==========
    'optimizer': 'AdamW',             # 优化器类型：AdamW、SGD、Adam等
    'lr0': 0.001,                     # 初始学习率：控制权重更新步长（迁移学习用0.001，从头训练用0.01）
    'lrf': 0.01,                      # 最终学习率比例：最终学习率 = lr0 * lrf
    'momentum': 0.937,                # 动量：SGD优化器的动量参数（AdamW不使用）
    'weight_decay': 0.0005,           # 权重衰减：L2正则化系数，防止过拟合
    
    # ========== 学习率预热 ==========
    'warmup_epochs': 3.0,             # 预热轮数：前N轮学习率从0逐渐增加到lr0（稳定训练初期）
    'warmup_momentum': 0.8,           # 预热动量：预热阶段的动量值
    'warmup_bias_lr': 0.1,            # 预热偏置学习率：预热阶段bias层的学习率
    'cos_lr': True,                   # 余弦学习率衰减：学习率按余弦曲线衰减（推荐开启）
    
    # ========== 损失函数权重 ==========
    'box': 7.5,                       # 边界框损失权重：检测框位置损失的权重（小目标可适当增大）
    'cls': 0.3,                       # 分类损失权重：类别预测损失的权重
    'dfl': 2.0,                       # DFL损失权重：Distribution Focal Loss权重（提高定位精度）
    'nbs': 64,                        # 标称批次大小：用于自动缩放损失权重（不需要修改）
    
    # ========== 数据增强 - Mosaic/Mixup系列 ==========
    'mosaic': 1.0,                    # Mosaic增强概率：将4张图拼接成1张（1.0=100%使用，推荐）
    'mixup': 0.1,                     # MixUp增强概率：混合两张图片（0.1=10%使用）
    'copy_paste': 0.2,                # Copy-Paste增强概率：复制粘贴目标到其他位置（0.2=20%）
    'multi_scale': 0.5,               # 多尺度训练范围：随机缩放图像尺寸±50%（增强尺度不变性）
    
    # ========== 数据增强 - 几何变换 ==========
    'degrees': 10.0,                  # 随机旋转角度：±10度（0表示不旋转）
    'translate': 0.05,                # 随机平移比例：图像宽高的±5%
    'scale': 0.3,                     # 随机缩放比例：±30%缩放
    'fliplr': 0.5,                    # 水平翻转概率：50%概率左右翻转
    'flipud': 0.5,                    # 垂直翻转概率：50%概率上下翻转（一般任务设为0）
    'perspective': 0.0,               # 透视变换概率：模拟相机视角变化（0表示不使用）
    
    # ========== 数据增强 - 颜色变换 ==========
    'hsv_h': 0.015,                   # 色调(Hue)增强范围：±0.015（色彩变化）
    'hsv_s': 0.7,                     # 饱和度(Saturation)增强范围：±0.7（颜色鲜艳度）
    'hsv_v': 0.4,                     # 明度(Value)增强范围：±0.4（亮度变化）
    
    # ========== 高级设置 ==========
    'close_mosaic': 10,               # 停止Mosaic轮数：最后N轮关闭Mosaic增强（让模型适应真实图像）
    'rect': False,                    # 矩形训练：保持图像原始宽高比（False=正方形padding）
    
    # ========== 验证与保存 ==========
    'val': True,                      # 是否验证：每轮训练后在验证集上评估
    'save': True,                     # 是否保存：保存训练权重
    'save_period': 10,                # 保存周期：每N轮保存一次权重（-1只保存last和best）
    'plots': True,                    # 是否绘图：生成训练曲线、混淆矩阵等可视化
    
    # ========== 可重复性 ==========
    'seed': 42,                       # 随机种子：固定随机数保证结果可复现
    'deterministic': True,            # 确定性训练：使用确定性算法（可能稍慢但结果可复现）
    'verbose': True,                  # 详细输出：打印详细的训练信息
}

# ============================================================
# 训练代码（不用修改）
# ============================================================

if __name__ == '__main__':
    # ========== 模型初始化 ==========
    print("="*60)
    print("🚀 YOLO训练配置")
    print("="*60)
    
    model_path = MODEL_CONFIG['path']
    model_type = MODEL_CONFIG['type']
    
    # ========== 模式1️⃣: 从头训练（随机初始化） ==========
    if model_type == 'scratch':
        print(f"📦 模型类型: 从头训练")
        print(f"📁 架构文件: {model_path}")
        print(f"🎲 权重初始化: 随机")
        
        model = YOLO(model_path)
        model_name = Path(model_path).stem
        
    # ========== 模式2️⃣: 使用预训练模型 ==========
    elif model_type == 'pretrained':
        print(f"📦 模型类型: 预训练模型")
        print(f"📁 模型文件: {model_path}")
        print(f"✅ 权重来源: 官方/已训练权重")
        
        model = YOLO(model_path)
        model_name = Path(model_path).stem
        
    # ========== 模式3️⃣: 自定义架构 + 迁移学习 ==========
    elif model_type == 'custom':
        print(f"📦 模型类型: 自定义架构 + 迁移学习")
        print(f"📁 架构文件: {model_path}")
        
        model = YOLO(model_path)
        model_name = Path(model_path).stem
        
        # 加载预训练权重（如果指定）
        if MODEL_CONFIG.get('pretrained'):
            pretrained_path = MODEL_CONFIG['pretrained']
            
            if os.path.exists(pretrained_path):
                print(f"📥 预训练权重: {pretrained_path}")
                
                try:
                    # 加载checkpoint
                    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                    
                    # 验证加载是否成功
                    if pretrained is None:
                        raise ValueError("权重文件为空或损坏")
                    
                    # 提取 state_dict（处理不同格式）
                    if isinstance(pretrained, dict):
                        if 'model' in pretrained:
                            # Ultralytics格式：{'model': model_object, 'optimizer': ...}
                            pretrained_state = pretrained['model'].state_dict()
                        elif 'state_dict' in pretrained:
                            # 标准格式：{'state_dict': {...}, ...}
                            pretrained_state = pretrained['state_dict']
                        else:
                            # 直接是 state_dict
                            pretrained_state = pretrained
                    elif hasattr(pretrained, 'state_dict'):
                        pretrained_state = pretrained.state_dict()
                    else:
                        raise ValueError("无法从checkpoint中提取state_dict")
                    
                    model_state = model.model.state_dict()
                    
                    # 过滤兼容的权重
                    compatible_state = {}
                    for k, v in pretrained_state.items():
                        if k in model_state and model_state[k].shape == v.shape:
                            compatible_state[k] = v
                    
                    incompatible_count = len(pretrained_state) - len(compatible_state)
                    
                    # 加载权重
                    model.model.load_state_dict(compatible_state, strict=False)
                    
                    print(f"✅ 成功加载 {len(compatible_state)}/{len(pretrained_state)} 个权重")
                    if incompatible_count > 0:
                        print(f"⚠️  跳过 {incompatible_count} 个不兼容权重（新模块将随机初始化）")
                    
                except Exception as e:
                    print(f"❌ 加载权重失败: {e}")
                    print("   将从头开始训练（随机初始化）")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ 文件不存在: {pretrained_path}")
                print("   将从头开始训练（随机初始化）")
        else:
            print(f"⚠️  未指定预训练权重，将随机初始化")
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请使用 'scratch', 'pretrained', 'custom'")
    
    # ========== 生成实验名称 ==========
    dataset_name = Path(DATA_PATH).parent.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{dataset_name}_{model_name}_{TRAIN_ARGS['imgsz']}_{timestamp}"
    
    print(f"📊 数据集: {dataset_name}")
    print(f"🎯 实验名称: {experiment_name}")
    print(f"📂 保存路径: runs/train/{experiment_name}/")
    
    # ========== 设置日志保存 ==========
    original_stdout = sys.stdout
    log_file = None
    
    log_dir = Path('runs/train') / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'training_log.txt'
    
    print(f"📝 日志文件: {log_file}")
    print("="*60 + "\n")
    
    # 重定向输出（同时打印到控制台和文件）
    sys.stdout = Logger(log_file)

    try:
        # ========== 开始训练 ==========
        results = model.train(
            data=DATA_PATH,
            project='runs/train',
            name=experiment_name,
            exist_ok=False,
            **TRAIN_ARGS  # 展开所有训练参数
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
        print(f"📝 控制台日志: {output_dir}/training_log.txt")
        print("-" * 60)
        print(f"📊 最终 mAP50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"📊 最终 mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
        print("=" * 60)
        
    finally:
        # 确保恢复标准输出（即使出错也会执行）
        sys.stdout.log_file.close()
        sys.stdout = original_stdout
        print(f"\n✅ 日志已保存到: {log_file}")
