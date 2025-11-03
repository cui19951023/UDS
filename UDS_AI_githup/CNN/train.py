import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 方法1：添加父目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from BaseClassification.model import MultimodalClassifier
from BaseClassification.dataset import MultimodalDataset
from BaseClassification.loss import FocalLoss


# 设置中文字体（Windows系统）


def train_model(task_name, model, train_loader, val_loader, criterion, optimizer,
                device, num_epochs=10, scheduler=None, early_stopping_patience=3):
    """
    训练多模态分类模型

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备(cpu/cuda)
        num_epochs: 训练轮数
        scheduler: 学习率调度器
        early_stopping_patience: 早停耐心值
    """
    model.to(device)
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_preds = []
        train_true = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            images = batch['image'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, numerical)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 记录统计信息
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(true_labels.cpu().numpy())

        # 计算训练指标
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_true, train_preds)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                images = batch['image'].to(device)
                numerical = batch['numerical'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, numerical)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(true_labels.cpu().numpy())

        # 计算验证指标
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_true, val_preds)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # 更新学习率
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # 早停检查
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            # 保存最佳模型
            os.makedirs(f'Weights/{task_name}/', exist_ok=True)
            torch.save(model.state_dict(), f'Weights/{task_name}/best_model.pth')
            print('Save Best Model!!!!')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

        # 打印进度
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Train LR: {scheduler.get_last_lr()[0]:.6f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Best Acc: {best_val_acc:.4f}')



    return history


def train(task_name, batch_size, num_epochs):
    feature_columns = ['BC', '初感觉', '初急迫', '强烈急迫', '最大灌注容量', 'BCI', '膀胱安全容量',
                       '最大尿流率对应逼尿肌压力', '最大尿流率', '残余尿', 'BOOI', '急迫性尿失禁情况', '咳嗽漏尿情况']
    task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
             'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
             'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
             'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
             'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                            "Unable to determine"],
             'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"]}
    # 创建模型
    model = MultimodalClassifier(num_numerical_features=len(feature_columns), num_classes=len(task_dict[task_name]))

    train_dataset = MultimodalDataset('../UDS_Data/uds_png', f'../UDS_Data/{task_name}/train.csv', feature_columns, task_name)
    val_dataset = MultimodalDataset('../UDS_Data/uds_png', f'../UDS_Data/{task_name}/val.csv', feature_columns, task_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    # 训练模型
    history = train_model(
        task_name= task_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=20
    )

# 示例使用训练函数
if __name__ == "__main__":
    train('Bladder_Sensation', 32, 100)
    train('Bladder_Compliance', 32, 100)
    train('Detrusor_Muscle_Stability', 32, 100)
    train('Detrusor_Contractility', 32, 100)
    train('Bladder_Outlet_Obstruction', 32, 100)
    train('Coordination_of_Detrusor_Sphincter', 32, 100)






