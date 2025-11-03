import os

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import sys
from pathlib import Path
import pandas as pd

# 方法1：添加父目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from BaseClassification.model import MultimodalClassifier
from BaseClassification.dataset import MultimodalDataset
from BaseClassification.metric import plt_confusion_matrix, plot_multiclass_roc, save_classification_report


task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
                 'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
                 'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
                 'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
                 'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                                "Unable to determine"],
                 'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"]}



def test_model(model, test_loader, device):
    """
    测试多模态分类模型

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 测试设备(cpu/cuda)
        class_names: 类别名称列表(用于可视化)
    """
    model.to(device)
    model.eval()

    test_trues, test_preds, test_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            image = batch['image'].to(device)
            numerical = batch['numerical'].to(device)
            label = batch['label'].to(device)

            outputs = model(image, numerical)

            _, pred = torch.max(outputs, 1)
            _, true_label = torch.max(label, 1)
            probs = torch.softmax(outputs, dim=1)

            test_preds.extend(pred.cpu().numpy())
            test_trues.extend(true_label.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    return {
        'predictions': test_preds,
        'labels': test_trues,
        'probabilities': np.vstack(test_probs)
    }


def test(task_name, test_csv,category,image_dir, batch_size=32):
    feature_columns = ['BC', '初感觉', '初急迫', '强烈急迫', '最大灌注容量', 'BCI', '膀胱安全容量',
                       '最大尿流率对应逼尿肌压力', '最大尿流率', '残余尿', 'BOOI', '急迫性尿失禁情况', '咳嗽漏尿情况']

    class_names = task_dict[task_name]
    # 创建模型并加载预训练权重
    model = MultimodalClassifier(num_numerical_features=len(feature_columns), num_classes=len(class_names))
    model.load_state_dict(torch.load(f'Weights/{task_name}/best_model.pth'))

    # 假设已经创建了测试数据集
    test_dataset = MultimodalDataset(
        image_dir=image_dir,
        csv_path=test_csv,
        numerical_columns=feature_columns,
        label_column=task_name
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_results = test_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    save_dir = f'results/CNN/{task_name}/'
    os.makedirs(save_dir, exist_ok=True)
    # 打印分类报告
    print('\nTest Results:')
    # print(classification_report(test_results['labels'], test_results['predictions'], target_names=class_names))
    plt_confusion_matrix(test_results['labels'], test_results['predictions'], class_names, save_dir + f'{category}_CM_CNN.pdf')
    plot_multiclass_roc(test_results['labels'], test_results['probabilities'], class_names,save_dir + f'{category}_ROC_CNN.pdf')
    save_classification_report(test_results['labels'], test_results['predictions'], save_dir + f'{category}_report_CNN.csv')

    df_test = pd.read_csv(test_csv)
    df_test = df_test.drop(['Bladder_Sensation', 'Bladder_Compliance',
                  'Detrusor_Muscle_Stability', 'Detrusor_Contractility',
                  'Bladder_Outlet_Obstruction', 'Coordination_of_Detrusor_Sphincter'],
                 axis=1
                 )
    for i, name in enumerate(class_names):
        df_test[name] = test_results['probabilities'][:, i]
    df_test.to_csv(save_dir + f'{category}_CNN.csv', index=False)


# 示例使用测试函数
if __name__ == "__main__":
    for category in ['train','val','test']:
        test('Bladder_Sensation', f'../UDS_Data//Bladder_Sensation/{category}.csv',category, '../UDS_Data//uds_png')
        test('Bladder_Compliance', f'../UDS_Data//Bladder_Compliance/{category}.csv',category, '../UDS_Data//uds_png')
        test('Detrusor_Muscle_Stability', f'../UDS_Data//Detrusor_Muscle_Stability/{category}.csv',category,  '../UDS_Data//uds_png')
        test('Detrusor_Contractility', f'../UDS_Data//Detrusor_Contractility/{category}.csv',category,  '../UDS_Data//uds_png')
        test('Bladder_Outlet_Obstruction', f'../UDS_Data//Bladder_Outlet_Obstruction/{category}.csv',category,  '../UDS_Data//uds_png')
        test('Coordination_of_Detrusor_Sphincter', f'../UDS_Data//Coordination_of_Detrusor_Sphincter/{category}.csv',category,  '../UDS_Data//uds_png')


