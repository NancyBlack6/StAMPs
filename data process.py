import pandas as pd
from sklearn.preprocessing import StandardScaler

# load Excel
train_data = pd.read_csv('./data process/train_data.CSV')
imput_data = pd.read_csv('./data imput_feature.CSV')
print(imput_data)

# Screen
imput_data_selected = imput_data.reindex(columns=train_data.columns, fill_value=1)
print(imput_data_selected)
imput_data_selected.to_csv("./data process/screened.CSV", index=False)


# Standardized train dataset
datanum = train_data.iloc[:, 3:]
dataid = train_data.iloc[:, :3]
scaler = StandardScaler()
datanum_scaled = scaler.fit_transform(datanum)
datanum = pd.DataFrame(datanum_scaled, columns=datanum.columns)
print("Standardized training data:\n", datanum)


# Standardized imput data
imputnum = imput_data_selected.iloc[:, 3:]
imputid = imput_data_selected.iloc[:, :3]
imputnum_scaled = scaler.transform(imputnum)
imputnum = pd.DataFrame(imputnum_scaled, columns=imputnum.columns)
print("Standardized test data:\n", imputnum)
imput_result = pd.concat([imputid, imputnum], axis=1)


# Save result
imput_result.to_csv(r"./result\data process result/test_standardized.CSV", index=False)

