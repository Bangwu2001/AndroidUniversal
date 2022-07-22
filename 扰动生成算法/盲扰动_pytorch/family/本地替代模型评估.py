import torch
import pickle
import numpy as np


cnnModel=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
train_data_path="dataset/markov_evaluate_1000_X"
train_label_path="dataset/markov_evaluate_1000_Y"
with open(train_data_path,"rb") as f:
    train_data=pickle.load(f)
with open(train_label_path,"rb") as f:
    train_label=pickle.load(f)
train_label=np.squeeze(train_label)
print(train_data.shape,train_label.shape)
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=np.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer

train_data=torch.from_numpy(train_data)
train_label=torch.from_numpy(train_label)
train_label=torch.unsqueeze(train_label,dim=1)
print(train_data.shape,train_label.shape)
#
#在测试数据干净样本上测试
print("干净样本评估")
res=cnnModel(train_data)
correct=torch.sum((res>=0.5)==train_label).detach().numpy()
print(correct/len(train_label))

#
#对抗样本评估

train_data_path="iter_dataset/markov_evaluate_1000_X_sample_trainLoop2000_1_5000_selfDefined_v3"
train_label_path="iter_dataset/markov_evaluate_1000_Y_sample_trainLoop2000_1_5000_selfDefined_v3"
with open(train_data_path,"rb") as f:
    train_data=pickle.load(f)
with open(train_label_path,"rb") as f:
    train_label=pickle.load(f)

train_data=torch.from_numpy(train_data)
train_label=torch.from_numpy(train_label)
train_label=np.squeeze(train_label)
# train_data=train_data.numpy()
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=torch.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer
train_label=torch.unsqueeze(train_label,dim=1)
print(train_data.shape,train_label.shape)
# train_data=torch.from_numpy(train_data)

train_data=train_data.float()
res=cnnModel(train_data)
correct=torch.sum((res>=0.5)==train_label).detach().numpy()
print("对抗样本")
print(correct/len(train_label))