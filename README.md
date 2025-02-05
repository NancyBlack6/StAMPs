# StAMPs
## 1.Intruduction
Welcome to StAMP, the model for predicting stapled antimicrobial peptides. It is possible to read and cite relevant literature during use.
## 2.Procedure
### 2-1.Acquiring and entering data
Initially, the PDB structure of the stapled peptide must be obtained. This structure can be acquired through experimental determination or structure prediction. Subsequently, the 2D and 3D molecular descriptors of the stapled peptide must be calculated using the molecular operating environment (MOE) version 2019, and the specific molecular descriptors must be input into the data_imput_feature.csv file. Finally, the sequence information of the stapled peptide must be input into the data_imput_sequence.csv file.
### 2-2.Data processing
In order to standardize the input data, it is necessary to run "data process.py" file.
### 2-3.Activity Prediction
Download the “trained model.zip” file in the “release” module and extract it to the "ML" folder. Run the “predict.py” file to output and summarize the predictions. Prediction results can be viewed in the "result" folder.
## 3.Contect us
You can cntect us with the e-mail of "zhengh18@hotmail.com"  
Dr. Heng Zheng  
School of Life Science and Technology,  
China Pharmaceutical University
