import matplotlib.pyplot
import shap
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import xgboost

from metrics import save_classification_report, plot_multiclass_roc, plt_confusion_matrix

plt.rcParams["font.family"] = "serif"  # 使用衬线字体
plt.rcParams["font.serif"] = ["Times New Roman"]  # 指定 Times New Roman
plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体（可选，与 Times 风格一致）

seed = 42

'''
Load the data
'''

numeric_features = ['BC', '初感觉', '初急迫', '强烈急迫', '最大灌注容量', 'BCI', '膀胱安全容量',
                    '最大尿流率对应逼尿肌压力', '最大尿流率', '残余尿', 'BOOI']

categorical_features = ['性别', '急迫性尿失禁情况', '咳嗽漏尿情况']
features_list = numeric_features + categorical_features

features_list_en = ['BC', 'First sensation of bladder filling',
                    'First desire to void', 'Strong desire to void', 'MCC', 'BCI',
                    'Bladder safe capacity',
                    'PdetQmax', 'Qmax', 'PVR', 'BOOI','Sex', 'UUI', 'Cough-induced urinary leakage']


task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
             'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
             'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
             'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
             'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                            "Unable to determine"],
             'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"], }


def load_data(csv_path, label_name):
    df = pd.read_csv(csv_path)
    x = df[features_list]
    y = df[label_name]
    return x, y


def calculate_proba(model_path, input_):
    loaded_model = joblib.load(model_path)
    y_predict = loaded_model.predict(input_)
    y_score = loaded_model.predict_proba(input_)
    return y_predict, y_score, loaded_model


def test(test_csv, task_name, category):
    print(f'------------------{task_name}----------------------')
    x, y = load_data(test_csv, task_name)
    y_predict, y_score, model = calculate_proba(f'weights/{task_name}_xgb.pkl', x)
    label_names = task_dict[task_name]

    save_dir = f'results/ML/{task_name}/'
    os.makedirs(save_dir, exist_ok=True)

    plot_multiclass_roc(y, y_score, label_names, save_dir + f'{category}_ROC_ML.pdf')
    plt_confusion_matrix(y, y_predict, label_names, save_dir + f'{category}_CM_ML.pdf')
    save_classification_report(y, y_predict, save_dir + f'{category}_report_ML.csv')

    df_test = pd.read_csv(test_csv)
    for i, name in enumerate(label_names):
        df_test[name] = y_score[:, i]
    df_test.to_csv(save_dir + f'{category}_ML.csv', index=False)


def plot_shap(test_csv, task_name):
    print(f'-------------------{task_name}------------------------')
    x, y = load_data(test_csv, task_name)

    y_predict, y_score, model = calculate_proba(f'weights/{task_name}_xgb.pkl', x)
    # 初始化解释器
    explainer = shap.TreeExplainer(model)

    # 计算测试集的 SHAP 值（返回长度为4的列表，对应每个类别）
    shap_values = explainer.shap_values(x)
    # SHAP分析
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values,
        x,
        feature_names=features_list_en,
        class_names=task_dict[task_name],
        plot_type='bar',  # 显示平均绝对SHAP值
        show=False
    )

    ax = plt.gca()
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.title(task_name)
    plt.tight_layout()
    plt.savefig(f'results/shap/{task_name}_shap.pdf', bbox_inches='tight')  # 保存时包含完整图例
    plt.close()



if __name__ == '__main__':
    for category in ['train', 'val', 'test']:
        test(f'data/Bladder_Sensation/{category}.csv', 'Bladder_Sensation',category)
        test(f'data/Bladder_Compliance/{category}.csv', 'Bladder_Compliance',category)
        test(f'data/Detrusor_Muscle_Stability/{category}.csv', 'Detrusor_Muscle_Stability',category)
        test(f'data/Detrusor_Contractility/{category}.csv', 'Detrusor_Contractility',category)
        test(f'data/Bladder_Outlet_Obstruction/{category}.csv', 'Bladder_Outlet_Obstruction',category)
        test(f'data/Coordination_of_Detrusor_Sphincter/{category}.csv', 'Coordination_of_Detrusor_Sphincter',category)

    # for task_name in task_dict.keys():
    #     plot_shap(f'data/{task_name}/test.csv', task_name)

