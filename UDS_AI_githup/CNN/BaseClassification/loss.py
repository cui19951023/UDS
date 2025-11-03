import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 如果是类别索引形式，转换为one-hot编码
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # 使用log_softmax提高数值稳定性
        log_p = F.log_softmax(inputs, dim=1)
        
        # 计算交叉熵
        ce_loss = -torch.sum(targets * log_p, dim=1)
        
        # 获取正确类别的概率
        p = torch.exp(log_p)
        p_t = torch.sum(p * targets, dim=1)
        
        # 计算调制因子
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 计算Focal Loss
        focal_loss = modulating_factor * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            alpha_weight = torch.sum(self.alpha * targets, dim=1)
            focal_loss = alpha_weight * focal_loss
        
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss