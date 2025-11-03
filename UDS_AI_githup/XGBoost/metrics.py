import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc, \
    precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
import seaborn as sns

color_list = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'darkorange']

plt.rcParams["font.family"] = "serif"  # 使用衬线字体
plt.rcParams["font.serif"] = ["Times New Roman"]  # 指定 Times New Roman
plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体（可选，与 Times 风格一致）


def calculate_metrics(y_true, y_predict, y_score, class_names, save_path=None):
    """
    Calculate all discrimination metrics
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    results_dict = {'class_name': class_names, 'AUC': []}
    for i, name in enumerate(class_names):
        roc_auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        results_dict['AUC'].append(roc_auc)
    if save_path:
        df = pd.DataFrame(results_dict)
        df.to_csv(save_path, index=False)


def plot_multiclass_roc(y_true, y_score, class_names, save_path=None):
    """
    绘制多分类ROC曲线并计算AUC值
    """
    n_classes = len(class_names)
    # 将标签二值化（多类别转多标签）
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # 绘制ROC曲线
    fig = plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        if not np.isnan(roc_auc):

            class_label = class_names[i] if class_names is not None else f"Class {i}"
            plt.plot(fpr, tpr, color=color_list[i], lw=2, label=f'{class_label} (AUC = {roc_auc:0.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plt_confusion_matrix(y_true, y_predict, class_names, save_path=None):
    """绘制混淆矩阵"""

    cm = confusion_matrix(y_true, y_predict)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_classification_report(y_true, y_predict, save_path=None):
    """
    生成并保存分类报告为CSV文件
    """
    # 生成分类报告字典
    report_dict = classification_report(y_true, y_predict, output_dict=True)

    # 转换字典为DataFrame
    df = pd.DataFrame(report_dict).transpose()

    # 处理多类分类的accuracy行
    if 'accuracy' in df.index:
        df.loc['accuracy', ['precision', 'recall', 'f1-score']] = np.nan

    # 保存为CSV
    if save_path:
        df.to_csv(save_path, float_format='%.4f')
        print(f"分类报告已保存为 {save_path}")


