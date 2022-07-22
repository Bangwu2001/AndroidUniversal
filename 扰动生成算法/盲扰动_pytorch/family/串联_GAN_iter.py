"""
两种扰动方式串联
    ①将扰动作为参数，用优化的方式生成通用对抗扰动
    ②将通用对抗扰动叠加到训练数据上，训练一对一的GAN扰动生成器

    对于训练数据而言,在叠加扰动之前准确率:0.8791,叠加扰动之后准确率:0.1407
"""

import torch
import numpy as np
import pickle
from GANModel import FamilyGeneratorModel
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

def StandardScale(data):
    for index in range(data.shape[0]):
        for row in range(data.shape[2]):
            sumer = torch.sum(data[index, :, row, :])
            if sumer != 0:
                data[index, :, row, :] /= sumer
    return data


#1.加载预先生成的通用对抗扰动
with open("iter_dataset/markov_universal_1000_trainLoop2000_1_1000","rb") as f:
    universal_pertubation=pickle.load(f)
universal_pertubation=universal_pertubation.int().float().numpy()
print(universal_pertubation[0,0])

#2.加载本地替代分类模型
targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")


#3.加载GAN的训练数据
with open("dataset/trainG_data","rb") as f:
    train_data=pickle.load(f)
with open("dataset/trainG_label","rb") as f:
    train_label=pickle.load(f)

train_data=train_data+universal_pertubation
train_data=torch.from_numpy(train_data)
train_label=torch.from_numpy(train_label)

MyDataset=TensorDataset(train_data,train_label)
MyDataLoader=DataLoader(MyDataset,batch_size=100,shuffle=True)

#构建生成器
generatorModel=FamilyGeneratorModel()
opt=torch.optim.Adam(generatorModel.parameters(),lr=1e-3)

#超参数配置
#两个损失函数的比例
alpha=1
beta=1
mask=torch.cat([torch.zeros(size=(1,9,11)),torch.ones(size=(1,2,11))],dim=1)  #掩膜,只有最后两行允许出现扰动量

epochs=10

for epoch in range(1,epochs+1):
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_both_loss = []
    epoch_correct = 0.
    for i,(batch_x,batch_y) in enumerate(MyDataLoader):
        batch_x=batch_x.float()
        batch_y=batch_y.float()
        # batch_adv_y=torch.ones_like(batch_y)  #对抗标签,1为bengin
        #扰动量
        pertubation=generatorModel(batch_x)
        # print(batch_x.shape,pertubation.shape,mask.shape)
        batch_adv_x=batch_x+pertubation*mask
        # print(type(batch_adv_x))
        batch_adv_x=StandardScale(batch_adv_x)  #归一化
        batch_adv_pred=targetF(batch_adv_x)
        correct=torch.sum(batch_adv_pred<0.5).detach().numpy()
        epoch_correct+=correct
        #扰动量损失
        loss1=torch.sum(pertubation*mask)/torch.sum(batch_x)

        #对抗样本损失,采用HingLoss,以0.5作为阈值
        loss2=torch.sum(torch.clamp(0.5-batch_adv_pred,min=0))/len(batch_x)

        #总的loss
        loss=alpha*loss1+beta*loss2
        epoch_loss1.append(loss1.detach().numpy())
        epoch_loss2.append(loss2.detach().numpy())
        epoch_both_loss.append(loss.detach().numpy())
        #参数更新
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Epoch:%d || all_loss:%.3f || overHeadLoss:%.3f || advLoss:%.3f || acc:%.3f" % (
    epoch, np.mean(epoch_both_loss), np.mean(epoch_loss1), np.mean(epoch_loss2), epoch_correct / len(train_label)))


torch.save(generatorModel,"generator/markov_generator_1_1_epoch_10_GANAnditer.pkl")




