import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, matthews_corrcoef

"""获得数据集"""
Feature_data = pd.read_csv(r".\result\data process result/test_standardized.csv")
Feature = Feature_data.iloc[:, 3:]
Feature_id = Feature_data.iloc[:, :2]
print(Feature)
print(Feature_id)
Feature = np.array(Feature, dtype=np.float64)
Feature_label = Feature_data.iloc[:, 2]
Feature_label = np.array(Feature_label, dtype=np.int64)

print(Feature)
"""调用函数模型"""
clf = joblib.load(r"./ML/trained model/KNN.m")

y_pred = clf.predict(Feature)
pred_prob = clf.predict_proba(Feature)
print(Feature_label)
print(y_pred)
print(pred_prob)
# confusion matrix computation and display
print("------------------------------------------------------------------")
print("SVC Accuracy: {0:0.1f}%".format(accuracy_score(Feature_label, y_pred) * 100))
print("precision: {0:0.1f}%".format(precision_score(Feature_label, y_pred) * 100))
print("------------------------------------------------------------------")
print(y_pred)
print(pred_prob)
pd.DataFrame(y_pred).to_csv(r"./result/predict result\each model/KNN.csv", index=False)
pd.DataFrame(pred_prob).to_csv(r"./result/predict result\each model/prob/KNN.csv", index=False)