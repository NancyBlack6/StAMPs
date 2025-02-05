import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef


protein_dict = {'X': 0,
                'G': 1,
                'A': 2,
                'V': 3,
                'L': 4,
                'I': 5,
                'P': 6,
                'F': 7,
                'Y': 8,
                'W': 9,
                'S': 10,
                'T': 11,
                'C': 12,
                'M': 13,
                'N': 14,
                'Q': 15,
                'D': 16,
                'E': 17,
                'K': 18,
                'R': 19,
                'H': 20}
seq_length = 200
max_length = 200
batch_size = 32
epochs = 500

def seq_to_number(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - j - 1] = protein_dict[line[len(line) - j -1]]
    return seq
"""准备数据"""
file = open(r"./data imput_sequence.csv")
read_file = file.readlines()
read_file = read_file[1:]

X = []
for i in range(len(read_file)):
    line = read_file[i]
    line = list(line.split(","))
    line = line[2]
    line = line[0:len(line) - 1]
    seq = seq_to_number(line, seq_length)
    X.append(seq)

Y = []

for i in range(len(read_file)):
    line = read_file[i]
    line = list(line.split(","))
    line = int(line[1])
    Y.append(line)
X = np.array(X, dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=2022)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

"""获得数据集"""
Feature_data = pd.read_csv(r".\result\data process result/test_standardized.csv")
Feature = Feature_data.iloc[:, 3:]
Feature_id = Feature_data.iloc[:, :2]
print(Feature)
print(Feature_id)
Feature = np.array(Feature, dtype=np.float64)
Feature_label = Feature_data.iloc[:, 2]
Feature_label = np.array(Feature_label, dtype=np.int64)
feature_x_train, feature_x_test, feature_y_train, feature_y_test = train_test_split(Feature, Feature_label, random_state=2022)
start_time = time.time()

"""调用函数模型"""
model = load_model(r'./ML/trained model/LSTM_best_classification_model.h5')
print(Feature)
print("------------------------------------------------------------------")
print(X)
pred_prob = model.predict([Feature, X])
y_pred = np.where(pred_prob >0.5, 1, 0)
print(y_pred)
print(pred_prob)
# confusion matrix computation and display
print("------------------------------------------------------------------")
print("SVC Accuracy: {0:0.1f}%".format(accuracy_score(Feature_label, y_pred) * 100))
print("precision: {0:0.1f}%".format(precision_score(Feature_label, y_pred) * 100))
print("------------------------------------------------------------------")
pd.DataFrame(y_pred).to_csv(r"./result/predict result\each model/LSTM.csv", index=False)
pd.DataFrame(pred_prob).to_csv(r"./result/predict result\each model/prob/LSTM.csv", index=False)
