"""训练策略配置文件 - 所有可调参数"""

# ============================================================
# 模型配置
# ============================================================
# 三种模式：
# 1. 'scratch'    - 从零训练官方架构（如 'yolo11s.yaml'）
# 2. 'pretrained' - 加载预训练权重（如 'yolo11s.pt'）
# 3. 'custom'     - 自定义架构 + 可选预训练（如 'yolo11s_cbam.yaml' + 'yolo11s.pt'）

# 示例 3: 训练自定义架构（如 yolo11s-cbam），从零开始
# MODEL_CONFIG = {
#     'type': 'pretrained',
#     'path': '/tmp/pycharm_project_949/runs/detect/runs/train/seedTrueLeaf.v12i.yolov11_yolo11s_800_20260201_131735/weights/best.pt',
# }

MODEL_CONFIG = {
    'type': 'scratch',
    'path': 'ultralytics/cfg/models/26/yolo26-p2.yaml',
}


# ============================================================
# 📋 配置模板示例
# ============================================================
# 
# 示例 1: 从零训练 yolo11s（不使用预训练）
# MODEL_CONFIG = {
#     'type': 'scratch',
#     'path': 'yolo11s.yaml',
# }
#
# 示例 2: 使用预训练权重训练 yolo11s（推荐）
# MODEL_CONFIG = {
#     'type': 'pretrained',
#     'path': 'yolo11s.pt',
# }
#
# 示例 3: 训练自定义架构（如 yolo11s-cbam），从零开始
# MODEL_CONFIG = {
#     'type': 'custom',
#     'path': 'ultralytics/cfg/models/sf/yolo11s_cbam.yaml',
# }
#
# 示例 4: 训练自定义架构，使用预训练权重（迁移学习）⭐
# MODEL_CONFIG = {
#     'type': 'custom',
#     'path': 'ultralytics/cfg/models/sf/yolo11s_cbam.yaml',
#     'pretrained': 'yolo11s.pt',  # 从 yolo11s 迁移权重
# }
# ============================================================

# ============================================================
# 数据配置
# ============================================================
DATA_PATH = './datasets/seedTrueLeaf.v13i.yolov11/data.yaml'

# ============================================================
# 冻结配置
# ============================================================
# YOLO11 架构：
# - 层 0-10:  Backbone（特征提取）
# - 层 11-23: Neck + Head（特征融合 + 检测）
#
# 100张图片 → 推荐冻结整个 Backbone（层 0-10）
FREEZE_CONFIG = {
    'freeze_backbone': False,       # True=冻结backbone, False=全量训练
    'freeze_layers': 11,           # 冻结前 11 层（层 0-10 = 完整 backbone）
}

# ============================================================
# 数据增强方案
# ============================================================
# 100张图片 + 冻结训练 → 推荐 aggressive（强数据增强）
SELECTED_AUGMENTATION = 'conservative'  # 'balanced' | 'aggressive' | 'conservative'

AUGMENTATION_PRESETS = {
    'balanced': {
        'mosaic': 0.8,
        'mixup': 0.1,
        'copy_paste': 0.7,
        'close_mosaic': 100,
        'multi_scale': 0.5,
        'degrees': 180.0,
        'translate': 0.1,
        'fliplr': 0.5,
        'flipud': 0.5,
        'perspective': 0.0001,
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.4,
        'rect': False,
    },
    
    'aggressive': {
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.9,
        'close_mosaic': 150,
        'multi_scale': 0.7,
        'degrees': 180.0,
        'translate': 0.15,
        'fliplr': 0.5,
        'flipud': 0.5,
        'perspective': 0.0002,
        'hsv_h': 0.02,
        'hsv_s': 0.7,
        'hsv_v': 0.6,
        'rect': False,
    },
    
    'conservative': {
        'mosaic': 0.5,
        'mixup': 0.0,
        'copy_paste': 0.5,
        'close_mosaic': 80,
        'multi_scale': 0.3,
        'degrees': 180.0,
        'translate': 0.05,
        'fliplr': 0.5,
        'flipud': 0.5,
        'perspective': 0.0,
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'rect': False,
    },
}

# ============================================================
# 训练参数 - 全量训练
# ============================================================
TRAIN_ARGS_FULL = {
    'epochs': 4000,
    'patience': 300,
    'batch': 4,
    'imgsz': 800,
    'device': 0,
    'workers': 6,
    'cache': 'ram',
    'amp': False,
    'optimizer': 'AdamW',
    'lr0': 0.0005,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0001,
    'warmup_epochs': 5.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,
    'cos_lr': True,
    'box': 15.0,
    'cls': 0.2,
    'dfl': 5.0,
    'nbs': 64,
    'val': True,
    'save': True,
    'save_period': 50,
    'plots': True,
    'seed': 42,
    'deterministic': True,
    'verbose': True,
}

# ============================================================
# 训练参数 - 冻结训练
# ============================================================
TRAIN_ARGS_FREEZE = {
    'epochs': 1000,
    'patience': 150,
    'batch': 8,
    'imgsz': 800,
    'device': 0,
    'workers': 6,
    'cache': 'ram',
    'amp': False,
    'optimizer': 'AdamW',
    'lr0': 0.0001,
    'lrf': 0.0001,
    'momentum': 0.937,
    'weight_decay': 0.00005,
    'warmup_epochs': 10.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.01,
    'cos_lr': True,
    'box': 10.0,
    'cls': 0.2,
    'dfl': 3.0,
    'nbs': 64,
    'val': True,
    'save': True,
    'save_period': 50,
    'plots': True,
    'seed': 42,
    'deterministic': True,
    'verbose': True,
}

# ============================================================
# 参数选择逻辑
# ============================================================
USE_FREEZE_PARAMS = None  # None=自动, True=强制freeze, False=强制full

def get_train_args():
    """根据冻结配置选择训练参数"""
    use_freeze = USE_FREEZE_PARAMS if USE_FREEZE_PARAMS is not None else FREEZE_CONFIG.get('freeze_backbone', False)
    args = TRAIN_ARGS_FREEZE.copy() if use_freeze else TRAIN_ARGS_FULL.copy()
    args.update(AUGMENTATION_PRESETS[SELECTED_AUGMENTATION])
    return args
