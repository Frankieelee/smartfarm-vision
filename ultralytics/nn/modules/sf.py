"""
ScalSeq (Scale Sequence) 特征融合模块 for YOLO11

实现了一种先进的多尺度特征融合策略，通过3D卷积沿尺度维度进行语义融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalSeq(nn.Module):
    """
    ScalSeq (Scale Sequence) 特征融合模块
    
    接收来自不同尺度的多个特征图（如 P3, P4, P5），通过 3D 卷积沿尺度维度
    进行特征融合，替代传统的 Concat 操作。
    
    处理流程：
    1. 通过 1×1 卷积统一各个输入特征图的通道维度
    2. 将所有特征图上采样到相同的空间分辨率
    3. 沿尺度维度堆叠，形成 3D 张量 (B, C, S, H, W)
    4. 使用 3D 卷积 (1,1,1) 进行跨尺度语义融合
    5. 批归一化 + LeakyReLU 激活
    6. 使用 3D 最大池化 (S,1,1) 压缩尺度维度
    7. 输出标准 2D 特征图 (B, C, H, W)
    
    Args:
        c1 (int or list): 输入通道数。如果是列表，表示每个输入特征图的通道数
        c2 (int): 输出通道数
        n (int, optional): 输入特征图数量（尺度数量）。默认为 3
        act (bool, optional): 是否使用激活函数。默认为 True
    
    Example:
        >>> scalseq = ScalSeq(c1=[256, 512, 1024], c2=256, n=3)
        >>> p3 = torch.randn(1, 256, 80, 80)
        >>> p4 = torch.randn(1, 512, 40, 40)
        >>> p5 = torch.randn(1, 1024, 20, 20)
        >>> out = scalseq([p3, p4, p5])
        >>> print(out.shape)  # torch.Size([1, 256, 80, 80])
    """
    
    def __init__(self, c1, c2, n=3, act=True):
        super().__init__()
        
        self.n = n
        self.c2 = c2
        
        # 处理输入通道数
        if isinstance(c1, (list, tuple)):
            self.c1_list = list(c1)
        else:
            self.c1_list = [c1] * n
        
        # 为每个输入尺度创建 1×1 卷积，统一通道维度
        self.channel_align = nn.ModuleList([
            nn.Conv2d(c_in, c2, kernel_size=1, stride=1, padding=0, bias=False)
            for c_in in self.c1_list
        ])
        
        # 3D 卷积：沿尺度维度进行跨尺度语义融合
        # kernel_size = (1, 1, 1) 表示 (尺度维度, 高度, 宽度)
        self.conv3d = nn.Conv3d(
            in_channels=c2,
            out_channels=c2,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=0,
            bias=False
        )
        
        # 批归一化（3D）
        self.bn = nn.BatchNorm3d(c2)
        
        # LeakyReLU 激活函数
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        
        # 3D 最大池化：压缩尺度维度
        # kernel_size = (n, 1, 1) 表示在尺度维度上池化，空间维度不变
        self.pool3d = nn.MaxPool3d(
            kernel_size=(n, 1, 1),
            stride=1,
            padding=0
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (list): 多个特征图的列表，例如 [P3, P4, P5]
        
        Returns:
            Tensor: 融合后的 2D 特征图
        """
        assert isinstance(x, (list, tuple)), f"ScalSeq 输入必须是列表，当前类型: {type(x)}"
        assert len(x) == self.n, f"ScalSeq 期望 {self.n} 个输入，但收到 {len(x)} 个"
        
        # 获取目标分辨率（通常是第一个特征图，即最高分辨率）
        target_h, target_w = x[0].shape[2], x[0].shape[3]
        
        # 通道对齐 + 空间上采样
        aligned_features = []
        for i, feat in enumerate(x):
            # 1×1 卷积统一通道数
            feat_aligned = self.channel_align[i](feat)
            
            # 如果分辨率不匹配，上采样到目标分辨率
            if feat_aligned.shape[2] != target_h or feat_aligned.shape[3] != target_w:
                feat_aligned = F.interpolate(
                    feat_aligned,
                    size=(target_h, target_w),
                    mode='nearest'
                )
            
            aligned_features.append(feat_aligned)
        
        # 沿尺度维度堆叠 -> (B, C, S, H, W)
        stacked = torch.stack(aligned_features, dim=2)
        
        # 3D 卷积进行跨尺度特征融合
        fused = self.conv3d(stacked)
        
        # 批归一化 + 激活
        fused = self.bn(fused)
        fused = self.act(fused)
        
        # 3D 最大池化压缩尺度维度 (B, C, S, H, W) -> (B, C, 1, H, W)
        pooled = self.pool3d(fused)
        
        # 移除尺度维度 -> (B, C, H, W)
        out = pooled.squeeze(2)
        
        return out
