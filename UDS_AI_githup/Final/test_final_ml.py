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

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows

seed = 42

'''
Load the data
'''

numeric_features = ['BC', '初感觉', '初急迫', '强烈急迫', '最大灌注容量', 'BCI', '膀胱安全容量',
                    '最大尿流率对应逼尿肌压力', '最大尿流率', '残余尿', 'BOOI']

categorical_features = ['性别', '急迫性尿失禁情况', '咳嗽漏尿情况']
features_list = numeric_features + categorical_features

task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
             'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
             'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
             'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
             'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                            "Unable to determine"],
             'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"], }



def load_data_ml(test_csv, features_list, label_name):
    df_test = pd.read_csv(test_csv)
    x_test, y_test = df_test[features_list], df_test[label_name]
    return x_test, y_test

def load_data_cnn(test_csv, features_list):
    df_test = pd.read_csv(test_csv)
    x_test = df_test[features_list]
    return x_test



def calculate_proba(model_path, input_):
    loaded_model = joblib.load(model_path)
    y_predict = loaded_model.predict(input_)
    y_score = loaded_model.predict_proba(input_)
    return y_predict, y_score, loaded_model


def test(test_csv, task_name,category):
    class_names = task_dict[task_name]
    print(f'------------------{task_name}----------------------')
    x_cnn= load_data_cnn(f'../results/CNN/{task_name}/{category}_CNN.csv', class_names)
    x_ml, y = load_data_ml(f'../results/ML/{task_name}/{category}_ML.csv', class_names, task_name)

    x = pd.concat([x_cnn, x_ml], axis=1).to_numpy()

    y_predict, y_score, model = calculate_proba(f'weights/{task_name}_lr.pkl', x)

    save_dir = f'../results/Final/{task_name}/'
    os.makedirs(save_dir, exist_ok=True)

    plot_multiclass_roc(y, y_score, class_names, save_dir+f'{category}_ROC_final.pdf')
    plt_confusion_matrix(y, y_predict, class_names, save_dir+f'{category}_CM_final.pdf')
    save_classification_report(y, y_predict,  save_dir+f'{category}_report_final.csv')

    df_test = pd.read_csv(test_csv)
    for i, name in enumerate(class_names):
        df_test[name] = y_score[:, i]
    df_test.to_csv(save_dir + f'{category}_test_final.csv', index=False)



if __name__ == '__main__':
    test('data/Bladder_Sensation/test.csv','Bladder_Sensation')
    test('data/Bladder_Compliance/test.csv', 'Bladder_Compliance')
    test('data/Detrusor_Muscle_Stability/test.csv', 'Detrusor_Muscle_Stability')
    test('data/Detrusor_Contractility/test.csv', 'Detrusor_Contractility')
    test('data/Bladder_Outlet_Obstruction/test.csv', 'Bladder_Outlet_Obstruction')
    test('data/Coordination_of_Detrusor_Sphincter/test.csv', 'Coordination_of_Detrusor_Sphincter')
