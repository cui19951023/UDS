import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# import shap

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from metrics import calculate_metrics, plot_multiclass_roc, plt_confusion_matrix, save_classification_report

seed = 33
np.random.seed(seed)
random.seed(seed)

task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
             'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
             'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
             'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
             'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                            "Unable to determine"],
             'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"]}


def load_data(label_name):
    df_train = pd.read_csv(f'data/{label_name}/train.csv')
    df_val = pd.read_csv(f'data/{label_name}/val.csv')
    df_test = pd.read_csv(f'data/{label_name}/test.csv')

    df_train = pd.concat([df_train, df_val])

    x_train, y_train = df_train[features_list], df_train[label_name]
    x_test, y_test = df_test[features_list], df_test[label_name]

    return x_train, y_train, x_test, y_test


def train(task_name, x_train, y_train, x_test, y_test, model, param_grid, model_path=None):
    # stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grid_search.fit(x_train, y_train, sample_weight=sample_weights)
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters for {model}:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    # 预测
    y_predict = best_model.predict(x_test)
    y_score = best_model.predict_proba(x_test)
    print("分类报告:\n", classification_report(y_test, y_predict))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_predict))

    if model_path:
        joblib.dump(best_model, model_path)


def train_xgb(task_name, x_train, y_train, x_test, y_test):
    print('XGB starting...')
    xgb_model = XGBClassifier(random_state=seed)
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0.1, 1],
        'eval_metric': ['mlogloss'],  # 多分类对数损失
        'objective': ['multi:softmax'],  # 多分类目标函数
        'num_class': [y_train.max() + 1]  # 类别数量
    }
    train(task_name, x_train, y_train, x_test, y_test, xgb_model, param_grid_xgb, f"weights/{task_name}_xgb.pkl")


def main(label_name):
    print(f'-------------{label_name}--------------------')
    # Load data
    x_train, y_train, x_test, y_test = load_data(label_name)
    # 计算每个类别的权重

    # x_train, y_train, x_test, y_test = load_data2(label_name)
    print("Refined x_train shape:", x_train.shape)
    print("Refined y_train shape:", y_train.shape)

    # Train models
    train_xgb(label_name, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    numeric_features = ['BC', '初感觉', '初急迫', '强烈急迫', '最大灌注容量', 'BCI', '膀胱安全容量',
                        '最大尿流率对应逼尿肌压力', '最大尿流率', '残余尿', 'BOOI']

    categorical_features = ['性别', '急迫性尿失禁情况', '咳嗽漏尿情况']
    features_list = numeric_features + categorical_features
    print(len(features_list))

    # main('Bladder_Sensation')
    # main('Bladder_Compliance')
    # main('Detrusor_Muscle_Stability')
    # main('Detrusor_Contractility')
    # main('Bladder_Outlet_Obstruction')
    # main('Coordination_of_Detrusor_Sphincter')
