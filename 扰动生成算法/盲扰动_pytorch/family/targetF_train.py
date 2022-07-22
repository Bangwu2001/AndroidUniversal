"""
通过训练CNN,作为本地替代模型
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn.functional as F
import pickle
from targetF_Model import FamilyTargetModel

#1.加载数据
with open("dataset/targetF_data","rb") as f:
    train_data=pickle.load(f)
with open("dataset/targetF_label","rb") as f:
    train_label=pickle.load(f)
print(train_data.shape)
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=np.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer

train_label=np.expand_dims(train_label,axis=1)
train_data=torch.from_numpy(train_data)
train_label=torch.from_numpy(train_label)
MyDataset=TensorDataset(train_data,train_label)
MyDataLoader=DataLoader(MyDataset,batch_size=64,shuffle=True)

#2.加载测试数据
with open("dataset/trainG_data","rb") as f:
    test_data=pickle.load(f)
with open("dataset/trainG_label","rb") as f:
    test_label=pickle.load(f)

for index in range(test_data.shape[0]):
    for row in range(test_data.shape[2]):
        sumer=np.sum(test_data[index,:,row,:])
        if sumer!=0:
            test_data[index,:,row,:]/=sumer


test_label=np.expand_dims(test_label,axis=1)
test_data=torch.from_numpy(test_data)
test_label=torch.from_numpy(test_label)




#网络实例化
model=FamilyTargetModel()

#模型训练配置
criterion=nn.BCELoss()
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
epochs=100

epoch_loss=[]
epoch_acc=[]
epoch_test_loss=[]
epoch_test_acc=[]
for epoch in range(epochs):
    train_loss=[]
    train_correct=0.
    for i,(x,y) in enumerate(MyDataLoader):
        x=x.float()
        y=y.float()
        pred=model(x)
        correct=torch.sum((pred>=0.5)==y).detach().numpy()
        train_correct+=correct
        loss=criterion(pred,y)
        train_loss.append(loss.detach().numpy())
        opt.zero_grad()
        loss.backward()
        opt.step()
    epoch_loss.append(np.array(train_loss).mean())
    epoch_acc.append(train_correct/len(train_data))

    with torch.no_grad():
        test_data=test_data.float()
        test_label=test_label.float()
        test_pred=model(test_data)
        test_loss=criterion(test_pred,test_label).detach().numpy()
        correct=torch.sum((test_pred>=0.5)==test_label).detach().numpy()
        acc=correct/len(test_data)
        epoch_test_loss.append(test_loss)
        epoch_test_acc.append(acc)
    print("Epoch：%d || train loss:%.3f || train acc:%.3f || test loss:%.3f || test acc:%.3f"%(epoch,epoch_loss[-1],epoch_acc[-1],test_loss,acc))

torch.save(model,"markov_classifier/Family_targetF_CNNModel.pkl")
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(epoch_loss)
plt.plot(epoch_test_loss)
plt.title("train&test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(2)
plt.plot(epoch_acc)
plt.plot(epoch_test_acc)
plt.legend()
plt.title("train&test acc")
plt.xlabel("epoch")
plt.ylabel("acc")

plt.show()

