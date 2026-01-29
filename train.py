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
from train_strategy import *

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
# 🔧 工具函数
# ============================================================

def freeze_layers(model, freeze_config):
    """
    冻结模型的指定层，只训练 detection head
    
    Args:
        model: YOLO 模型实例
        freeze_config: 冻结配置字典
    
    Returns:
        tuple: (冻结的参数数量, 可训练的参数数量)
    """
    if not freeze_config.get('freeze_backbone', False):
        print("⚠️  未启用 backbone 冻结，所有层都将训练")
        return 0, sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    freeze_layers_list = freeze_config.get('freeze_layers', 10)
    
    # 如果是数字，转换为列表
    if isinstance(freeze_layers_list, int):
        freeze_layers_list = list(range(freeze_layers_list))
    
    print(f"\n{'='*60}")
    print(f"❄️  冻结 Backbone 配置")
    print(f"{'='*60}")
    print(f"🔒 将冻结前 {len(freeze_layers_list)} 层")
    
    # 获取模型的所有层
    total_layers = len(list(model.model.model))
    print(f"📊 模型总层数: {total_layers}")
    print(f"🎯 冻结层: {freeze_layers_list}")
    print(f"🔥 可训练层: {list(range(max(freeze_layers_list) + 1, total_layers))}")
    
    # 冻结指定层
    frozen_params = 0
    trainable_params = 0
    
    for idx, (name, module) in enumerate(model.model.model.named_children()):
        if idx in freeze_layers_list:
            # 冻结该层的所有参数
            for param in module.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            print(f"   ❄️  层 {idx:2d} ({name:15s}): 已冻结 ({sum(p.numel() for p in module.parameters()):,} 参数)")
        else:
            # 保持该层可训练
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
            print(f"   🔥 层 {idx:2d} ({name:15s}): 可训练 ({sum(p.numel() for p in module.parameters()):,} 参数)")
    
    print(f"\n📊 参数统计:")
    print(f"   ❄️  冻结参数: {frozen_params:,} ({frozen_params / (frozen_params + trainable_params) * 100:.1f}%)")
    print(f"   🔥 可训练参数: {trainable_params:,} ({trainable_params / (frozen_params + trainable_params) * 100:.1f}%)")
    print(f"{'='*60}\n")
    
    return frozen_params, trainable_params


# ============================================================
# 训练代码（不用修改）
# ============================================================

if __name__ == '__main__':
    # ========== 显示配置信息 ==========
    print("="*60)
    print("🚀 YOLO训练配置")
    print("="*60)
    
    # 显示冻结状态
    if FREEZE_CONFIG.get('freeze_backbone', False):
        freeze_layers_count = FREEZE_CONFIG.get('freeze_layers', 10)
        if isinstance(freeze_layers_count, int):
            print(f"❄️  Backbone 冻结: 启用 (前 {freeze_layers_count} 层)")
        else:
            print(f"❄️  Backbone 冻结: 启用 (共 {len(freeze_layers_count)} 层)")
    else:
        print(f"🔥 Backbone 冻结: 禁用 (全量训练)")
    
    print(f"📊 数据增强方案: {SELECTED_AUGMENTATION.upper()}")
    
    # 重新获取训练参数（确保使用最新配置）
    TRAIN_ARGS = get_train_args()
    
    print("-"*60)
    
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
    
    # ========== 冻结 Backbone（如果启用）==========
    frozen_params, trainable_params = freeze_layers(model, FREEZE_CONFIG)
    
    # ========== 生成实验名称 ==========
    dataset_name = Path(DATA_PATH).parent.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 如果冻结了 backbone，在实验名称中标注
    freeze_suffix = ""
    if FREEZE_CONFIG.get('freeze_backbone', False):
        freeze_layers_count = FREEZE_CONFIG.get('freeze_layers', 10)
        if isinstance(freeze_layers_count, int):
            freeze_suffix = f"_freeze{freeze_layers_count}"
        else:
            freeze_suffix = f"_freeze{len(freeze_layers_count)}"
    
    experiment_name = f"{dataset_name}_{model_name}_{TRAIN_ARGS['imgsz']}{freeze_suffix}_{timestamp}"
    
    print(f"📊 数据集: {dataset_name}")
    print(f"🎯 实验名称: {experiment_name}")
    print(f"📂 保存路径: runs/train/{experiment_name}/")
    
    # 显示冻结状态
    if FREEZE_CONFIG.get('freeze_backbone', False):
        print(f"❄️  Backbone 状态: 已冻结 ({frozen_params:,} 参数)")
        print(f"🔥 可训练参数: {trainable_params:,} 参数")
        print(f"💡 建议: 使用较小的学习率 (当前: lr0={TRAIN_ARGS['lr0']})")
    else:
        print(f"🔥 训练模式: 全部参数可训练")
    
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
