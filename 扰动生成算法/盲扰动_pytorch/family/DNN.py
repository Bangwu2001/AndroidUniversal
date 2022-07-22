"""
黑盒分类器:DNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import pickle
import numpy as np

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1=nn.Linear(121,300)
        self.layer2=nn.Linear(300,300)
        self.layer3=nn.Linear(300,300)
        self.layer4=nn.Linear(300,300)
        self.layer5=nn.Linear(300,1)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=torch.sigmoid(self.layer5(x))
        return x
#
model=DNN()
#
# #数据
# #1.加载数据
# with open("dataset/targetF_data","rb") as f:
#     train_data=pickle.load(f)
# with open("dataset/targetF_label","rb") as f:
#     train_label=pickle.load(f)
# print(train_data.shape)
# for index in range(train_data.shape[0]):
#     for row in range(train_data.shape[2]):
#         sumer=np.sum(train_data[index,:,row,:])
#         if sumer!=0:
#             train_data[index,:,row,:]/=sumer
#
# train_label=np.expand_dims(train_label,axis=1)
# train_data=torch.from_numpy(train_data)
# train_data=train_data.reshape(train_data.shape[0],-1)
# train_label=torch.from_numpy(train_label)
# MyDataset=TensorDataset(train_data,train_label)
# MyDataLoader=DataLoader(MyDataset,batch_size=64,shuffle=True)
#
# #模型训练配置
# criterion=nn.BCELoss()
# opt=torch.optim.Adam(model.parameters(),lr=1e-3)
# epochs=30
#
# epoch_loss=[]
# epoch_acc=[]
#
# for epoch in range(epochs):
#     train_loss=[]
#     train_correct=0.
#     for i,(x,y) in enumerate(MyDataLoader):
#         x=x.float()
#         y=y.float()
#         pred=model(x)
#         correct=torch.sum((pred>=0.5)==y).detach().numpy()
#         train_correct+=correct
#         loss=criterion(pred,y)
#         train_loss.append(loss.detach().numpy())
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#     epoch_loss.append(np.array(train_loss).mean())
#     epoch_acc.append(train_correct/len(train_data))
#
#
#     print("Epoch：%d || train loss:%.3f || train acc:%.3f "%(epoch,epoch_loss[-1],epoch_acc[-1]))
#
# torch.save(model,"markov_classifier/dnn.pkl")
# import matplotlib.pyplot as plt
#
# plt.figure(1)
# plt.plot(epoch_loss)
# plt.title("train&test loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("loss")
#
# plt.figure(2)
# plt.plot(epoch_acc)
# plt.legend()
# plt.title("train&test acc")
# plt.xlabel("epoch")
# plt.ylabel("acc")
#
# plt.show()

dnnModel=torch.load("markov_classifier/dnn.pkl")
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
train_data=train_data.reshape(train_data.shape[0],-1)
print(train_data.shape,train_label.shape)
#
#在测试数据干净样本上测试
print("干净样本评估")
res=dnnModel(train_data)
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
# train_label=np.squeeze(train_label)
train_data=train_data.numpy()
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=np.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer
train_label=torch.unsqueeze(train_label,dim=1)
# print(train_label.shape)
train_data=train_data.reshape(train_data.shape[0],-1)
print(train_data.shape,train_label.shape)
train_data=torch.from_numpy(train_data)

#在测试数据干净样本上测试
train_data=train_data.float()
res=dnnModel(train_data)
correct=torch.sum((res>=0.5)==train_label).detach().numpy()
print("对抗样本")
print(correct/len(train_label))