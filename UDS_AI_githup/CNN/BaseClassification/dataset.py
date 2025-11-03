import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
from torchvision import transforms


class MultimodalDataset(Dataset):
    def __init__(self, image_dir, csv_path, numerical_columns, label_column, transform=None):
        """
        多模态数据集类

        参数:
            image_dir: 图像文件夹路径
            csv_path: 包含数值特征和标签的CSV文件路径
            numerical_columns: 数值特征列名的列表
            label_column: 标签列名
            transform: 图像变换
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.numerical_columns = numerical_columns
        self.label_column = label_column
        self.transform = transform if transform is not None else self.get_default_transform()
        self.class_num = self.data[label_column].max() + 1

    def get_default_transform(self):
        """默认的图像预处理变换"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0] + '.png')  # 假设CSV第一列是图像文件名(不带扩展名)
        image = Image.open(img_name).convert('RGB')

        # 应用变换
        image = self.transform(image)

        # 获取数值特征
        numerical_features = self.data[self.numerical_columns].iloc[idx].values
        numerical_features = numerical_features.astype(np.float32)

        # 获取标签
        label = np.zeros(self.class_num)
    
        label[self.data[self.label_column].iloc[idx]] = 1

        return {
            'image': image,
            'numerical': torch.from_numpy(numerical_features),
            'label': torch.tensor(label, dtype=torch.float16)
        }


# 示例使用数据集
if __name__ == "__main__":
    # 假设数据结构
    # CSV文件包含: image_id, feature1, feature2, ..., featureN, label
    # 图像存储在images文件夹中，命名为image_id.jpg

    dataset = MultimodalDataset(
        image_dir='images',
        csv_path='data.csv',
        numerical_columns=['feature1', 'feature2', 'feature3'],  # 示例特征列
        label_column='label'
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    sample = next(iter(dataloader))
    print(f"Image batch shape: {sample['image'].shape}")
    print(f"Numerical features shape: {sample['numerical'].shape}")
    print(f"Labels shape: {sample['label'].shape}")