import os
import joblib

import pandas as pd

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from metrics import save_classification_report, plot_multiclass_roc, plt_confusion_matrix

task_dict = {'Bladder_Sensation': ['Normal', 'Sensitive', 'Hypoesthesia', 'No Sensation'],
             'Bladder_Compliance': ['Normal', 'Hight', 'Low'],
             'Detrusor_Muscle_Stability': ["Filling end contraction", "Stable", "Uninhibited Contraction"],
             'Detrusor_Contractility': ["Normal", "Weakened", "No contraction"],
             'Bladder_Outlet_Obstruction': ["Obstruction", "Suspected obstruction", "No obstruction",
                                            "Unable to determine"],
             'Coordination_of_Detrusor_Sphincter': ["Normal", "Abnormal", "Undetectable"]}


def load_data_cnn(data_dir, features_list):
    df_train = pd.read_csv(data_dir + 'train_CNN.csv')
    df_val = pd.read_csv(data_dir + 'val_CNN.csv')
    df_test = pd.read_csv(data_dir + 'test_CNN.csv')

    df_train = pd.concat([df_train, df_val])

    x_train = df_train[features_list]
    x_test  = df_test[features_list]

    return x_train, x_test

def load_data_ml(data_dir, features_list,label_name):
    df_train = pd.read_csv(data_dir + 'train_ML.csv')
    df_val = pd.read_csv(data_dir + 'val_ML.csv')
    df_test = pd.read_csv(data_dir + 'test_ML.csv')

    df_train = pd.concat([df_train, df_val])

    x_train, y_train = df_train[features_list], df_train[label_name]
    x_test, y_test = df_test[features_list], df_test[label_name]

    return x_train, y_train, x_test, y_test


def final_train(task_name,model_path=None):

    class_names = task_dict[task_name]

    x_train_cnn, x_test_cnn = load_data_cnn(f'../results/CNN/{task_name}/', class_names)
    x_train_ml, y_train, x_test_ml, y_test = load_data_ml(f'../results/ML/{task_name}/', class_names, task_name)

    x_train = pd.concat([x_train_cnn, x_train_ml],axis=1).to_numpy()
    x_test = pd.concat([x_test_cnn, x_test_ml],axis=1).to_numpy()

    print("Refined x_train shape:", x_train.shape)
    print("Refined y_train shape:", y_train.shape)

    # xgb_model =
    meta_model = LogisticRegression(random_state=42,class_weight='balanced',penalty='l2', C=0.1)
    meta_model.fit(x_train, y_train)

    fused_pred = meta_model.predict(x_test)
    fused_score = meta_model.predict_proba(x_test)

    print("分类报告:\n", classification_report(y_test, fused_pred))
    print("混淆矩阵:\n", confusion_matrix(y_test, fused_pred))

    save_dir = f'../results/Final/{task_name}/'
    os.makedirs(save_dir, exist_ok=True)

    plot_multiclass_roc(y_test, fused_score, class_names, save_dir + 'ROC_final.pdf')
    plt_confusion_matrix(y_test, fused_pred, class_names, save_dir + 'CM_final.pdf')
    save_classification_report(y_test, fused_pred, save_dir + 'report_final.csv')

    if model_path:
        joblib.dump(meta_model, model_path)


if __name__ == '__main__':
    # final_train('Bladder_Sensation','weights/Bladder_Sensation_lr.pkl')
    # final_train('Bladder_Compliance','weights/Bladder_Compliance_lr.pkl')
    final_train('Detrusor_Muscle_Stability','weights/Detrusor_Muscle_Stability_lr.pkl')
    # final_train('Detrusor_Contractility','weights/Detrusor_Contractility_lr.pkl')
    # final_train('Bladder_Outlet_Obstruction','weights/Bladder_Outlet_Obstruction_lr.pkl')
    # final_train('Coordination_of_Detrusor_Sphincter','weights/Coordination_of_Detrusor_Sphincter_lr.pkl')