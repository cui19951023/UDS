import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultimodalClassifier(nn.Module):
    def __init__(self, num_numerical_features, num_classes, image_embedding_size=128,
                 numerical_embedding_size=128, fusion_size=512, dropout_prob=0.2):
        """
        多模态分类模型构造函数

        参数:
            num_numerical_features: 数值特征的数量
            num_classes: 分类类别数
            image_embedding_size: 图像嵌入维度
            numerical_embedding_size: 数值特征嵌入维度
            fusion_size: 融合层维度
            dropout_prob: dropout概率
        """
        super(MultimodalClassifier, self).__init__()

        # 图像特征提取器 (使用预训练的ResNet18)
        self.image_feature_extractor = models.resnet18()
        # 移除最后的全连接层
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])
        # 图像特征适配层
        self.image_adapter = nn.Sequential(
            nn.Linear(512, image_embedding_size),
            nn.BatchNorm1d(image_embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 数值特征处理
        self.numerical_branch = nn.Sequential(
            nn.Linear(num_numerical_features, numerical_embedding_size),
            nn.BatchNorm1d(numerical_embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(numerical_embedding_size, numerical_embedding_size),
            nn.BatchNorm1d(numerical_embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(image_embedding_size + numerical_embedding_size, fusion_size),
            nn.BatchNorm1d(fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 分类器
        self.classifier = nn.Linear(fusion_size, num_classes)

    def forward(self, image_input, numerical_input):
        """
        前向传播

        参数:
            image_input: 图像输入 (batch_size, 3, H, W)
            numerical_input: 数值输入 (batch_size, num_numerical_features)
        """
        # 处理图像输入
        image_features = self.image_feature_extractor(image_input)
        image_features = image_features.view(image_features.size(0), -1)  # 展平
        # image_embedding = self.image_adapter(image_features)

        # # 处理数值输入
        # numerical_embedding = self.numerical_branch(numerical_input)

        # # 融合两种模态
        # combined = torch.cat((image_embedding, numerical_embedding), dim=1)
        # fused = self.fusion(combined)

        # 分类
        logits = self.classifier(image_features)

        return logits


# 示例使用
if __name__ == "__main__":
    # 假设参数
    num_numerical = 10  # 数值特征数量
    num_classes = 5  # 分类类别数
    batch_size = 32

    # 创建模型
    model = MultimodalClassifier(num_numerical_features=num_numerical,
                                 num_classes=num_classes)

    # 模拟输入数据
    dummy_images = torch.randn(batch_size, 3, 224, 224)  # 假设图像大小为224x224
    dummy_numerical = torch.randn(batch_size, num_numerical)

    # 前向传播
    outputs = model(dummy_images, dummy_numerical)
    print(f"Output shape: {outputs.shape}")  # 应该为 (batch_size, num_classes)