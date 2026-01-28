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
    'path': '/root/autodl-tmp/sf-vision/smartfarm-vision/ultralytics/cfg/models/sf/yolo11s-cbam-p2-dectp2p3.yaml',
    'pretrained': 'yolo11s.pt',        # 官方预训练权重
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
DATA_PATH = '/root/autodl-tmp/seedTrue9i/data.yaml'

# ========== 训练参数 ==========
TRAIN_ARGS = {
    # ========== 基础配置 ==========
    'epochs': 2000,
    'patience': 300,              # 增加耐心值（小目标收敛慢）
    'batch': 8,                  # 增大batch（如果显存允许）
    'imgsz': 800,                 # 🔥 提高分辨率！关键改进
    
    # ========== 设备 ==========
    'device': 0,
    'workers': 6,
    'cache': 'ram',
    'amp': False,                 # 小目标建议关闭混合精度
    
    # ========== 优化器（小目标专用） ==========
    'optimizer': 'AdamW',
    'lr0': 0.0005,                # 🔥 降低学习率（更稳定）
    'lrf': 0.001,                 # 🔥 更小的最终学习率
    'momentum': 0.937,
    'weight_decay': 0.0001,       # 🔥 减小正则化（避免欠拟合）
    
    # ========== 学习率策略 ==========
    'warmup_epochs': 5.0,         # 🔥 增加预热（稳定训练）
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,       # 🔥 降低bias预热学习率
    'cos_lr': True,
    
    # ========== 损失权重（密集小目标专用） ==========
    'box': 10.0,                  # 🔥🔥 大幅增加box loss
    'cls': 0.2,                   # 🔥 降低cls loss（两类相似）
    'dfl': 3.0,                   # 🔥 增加DFL（提高定位精度）
    'nbs': 64,
    
    # ========== 数据增强（密集场景优化） ==========
    'mosaic': 0.5,                # 🔥 降低mosaic（密集场景mosaic会更密集）
    'mixup': 0.0,                 # 🔥 关闭mixup（密集场景不适用）
    'copy_paste': 0.5,            # 🔥 增加copy_paste（增强小目标）
    'multi_scale': 0.3,           # 🔥 减小多尺度范围
    
    # ========== 几何变换（俯视场景） ==========
    'degrees': 180.0,             # 🔥 俯视可以任意旋转
    'translate': 0.05,            # 🔥 减小平移（密集场景）
    'scale': 0.2,                 # 🔥 减小缩放（目标尺寸稳定）
    'fliplr': 0.5,
    'flipud': 0.5,                # 🔥 俯视可以上下翻转
    'perspective': 0.0,           # 俯视不需要透视
    
    # ========== 颜色增强（减弱） ==========
    'hsv_h': 0.01,                # 🔥 减小色调变化
    'hsv_s': 0.3,                 # 🔥 减小饱和度变化
    'hsv_v': 0.2,                 # 🔥 减小亮度变化
    
    # ========== 高级设置 ==========
    'close_mosaic': 50,           # 🔥 提前关闭mosaic
    'rect': False,
    
    # ========== 其他 ==========
    'val': True,
    'save': True,
    'save_period': 50,            # 🔥 增加保存频率
    'plots': True,
    'seed': 42,
    'deterministic': True,
    'verbose': True,
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
