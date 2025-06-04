import torch
import torch.nn as nn
from .GraphAttention import GraphAttentionLayer, EdgeConv, TransitionDown, TransitionUp

class DefectNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征提取层
        self.conv1 = EdgeConv(3, 64, nsample=16)
        self.conv2 = EdgeConv(64, 128, nsample=16)
        
        # 图注意力层
        self.attention1 = GraphAttentionLayer(128, 256, nsample=16)
        self.attention2 = GraphAttentionLayer(256, 512, nsample=16)
        
        # 下采样层
        self.down1 = TransitionDown(512, 512, stride=4, nsample=16)
        
        # 上采样层
        self.up1 = TransitionUp(512, 256)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, points):
        # 准备输入数据
        n = points.shape[1]  # 点的数量
        o = torch.tensor([n], device=points.device, dtype=torch.int32)  # batch信息
        p = points.squeeze(0)  # 移除batch维度
        n_xyz = p.clone()  # 法向量（这里简单使用坐标代替）
        
        # 特征提取
        x = self.conv1([p, n_xyz, p, o])
        x = self.conv2([p, n_xyz, x, o])
        
        # 图注意力处理
        x = self.attention1([p, n_xyz, x, o])
        x = self.attention2([p, n_xyz, x, o])
        
        # 下采样
        p, n_xyz, x, o = self.down1([p, n_xyz, x, o])
        
        # 上采样
        x = self.up1(x, o)
        
        # 分类
        x = self.classifier(x)
        
        return x