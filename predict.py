import subprocess
import pandas as pd

file_list = [
    './ML/ANN.py',
    './ML/CNN.py',
    './ML/DT.py',
    './ML/KNN.py',
    './ML/LGBM.py',
    './ML/LSTM.py',
    './ML/NB.py',
    './ML/RF.py',
    './ML/SVM.py',
    './ML/XGB.py'
]

# Run each prediction model file sequentially
for file in file_list:
    subprocess.run(['python', file])


# Summary of Prediction Results
pep = pd.read_csv(r"./data imput_sequence.csv", skiprows=[0])
ANN = pd.read_csv(r'./result/predict result\each model/ANN.csv', skiprows=[0])
CNN = pd.read_csv(r'./result/predict result\each model/CNN.csv', skiprows=[0])
LSTM = pd.read_csv(r'./result/predict result\each model/LSTM.csv', skiprows=[0])
DT = pd.read_csv(r'./result/predict result\each model/DT.csv', skiprows=[0])
KNN = pd.read_csv(r'./result/predict result\each model/KNN.csv', skiprows=[0])
LGBM = pd.read_csv(r'./result/predict result\each model/LGBM.csv', skiprows=[0])
NB = pd.read_csv(r'./result/predict result\each model/NB.csv', skiprows=[0])
RF = pd.read_csv(r'./result/predict result\each model/RF.csv', skiprows=[0])
SVM = pd.read_csv(r'./result/predict result\each model/svm.csv', skiprows=[0])
XGB = pd.read_csv(r'./result/predict result\each model/XGB.csv', skiprows=[0])
id = pep.iloc[:, :1]
seq = pep.iloc[:, 2:]
combined_result = pd.concat([id, seq, ANN, CNN, LSTM, DT, KNN, LGBM, NB, RF, SVM, XGB], axis=1, keys=['id', 'seq', 'ANN', 'CNN', 'LSTM', 'DT', 'KNN', 'LGBM', 'NB', 'RF', 'SVM', 'XGB'])
combined_result = combined_result.reset_index(drop=True)
print(combined_result)
combined_result.to_csv(r"./result/predict result/total_prediction.csv", index=False)


# Summary of Potential Results
pep = pd.read_csv(r"./data imput_sequence.csv", skiprows=[0])
ANN = pd.read_csv(r'./result/predict result\each model/prob/ANN.csv', skiprows=[0])
CNN = pd.read_csv(r'./result/predict result\each model/prob/CNN.csv', skiprows=[0])
LSTM = pd.read_csv(r'./result/predict result\each model/prob/LSTM.csv', skiprows=[0])
DT = pd.read_csv(r'./result/predict result\each model/prob/DT.csv', skiprows=[0])
KNN = pd.read_csv(r'./result/predict result\each model/prob/KNN.csv', skiprows=[0])
LGBM = pd.read_csv(r'./result/predict result\each model/prob/LGBM.csv', skiprows=[0])
NB = pd.read_csv(r'./result/predict result\each model/prob/NB.csv', skiprows=[0])
RF = pd.read_csv(r'./result/predict result\each model/prob/RF.csv', skiprows=[0])
SVM = pd.read_csv(r'./result/predict result\each model/prob/svm.csv', skiprows=[0])
XGB = pd.read_csv(r'./result/predict result\each model/prob/XGB.csv', skiprows=[0])
id = pep.iloc[:, :1]
seq = pep.iloc[:, 2:]
combined_result = pd.concat([id, seq, ANN.iloc[:, 1], CNN, LSTM, DT.iloc[:, 1], KNN.iloc[:, 1], LGBM.iloc[:, 1], NB.iloc[:, 1], RF.iloc[:, 1], SVM.iloc[:, 1], XGB.iloc[:, 1]], axis=1, keys=['id', 'seq', 'ANN', 'CNN', 'LSTM', 'DT', 'KNN', 'LGBM', 'NB', 'RF', 'SVM', 'XGB'])
combined_result = combined_result.reset_index(drop=True)
print(combined_result)
combined_result.to_csv(r"./result/predict result/total_prob.csv", index=False)
