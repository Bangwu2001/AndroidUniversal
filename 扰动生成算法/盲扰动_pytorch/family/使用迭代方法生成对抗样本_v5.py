"""
使用迭代的方法生成对抗样本:
    主体思路,将通用扰动量作为参数去更新
    本地和黑盒分类器均是用Markov转移概率进行训练
    试验一:训练样本选用1000个Malware
           损失函数:1.扰动量约束,2.对抗扰动损失

"""
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader,TensorDataset


#数据归一化处理,将调用次数矩阵转换为markov转移概率矩阵
def StandardScale(data):
    for index in range(data.shape[0]):
        for row in range(data.shape[2]):
            sumer = torch.sum(data[index, :, row, :])
            if sumer != 0:
                data[index, :, row, :] /= sumer
    return data


#1.加载训练数据
with open("dataset/trainG_data","rb") as f:
    train_data=pickle.load(f)
with open("dataset/trainG_label","rb") as f:
    train_label=pickle.load(f)

train_label=np.expand_dims(train_label,axis=1)
train_data=torch.from_numpy(train_data)
train_label=torch.from_numpy(train_label)
train_data=train_data[:1000]
train_label=train_label[:1000]
test_data=train_data[-1000:]
print(train_data.shape,train_label.shape)
#构建训练数据
MyDataset=TensorDataset(train_data,train_label)
MyDataLoader=DataLoader(MyDataset,batch_size=50,shuffle=True)

#扰动量,pertubation
v=torch.normal(mean=0,std=1.,size=(1,1,11,11),requires_grad=True)
mask=torch.cat((torch.zeros(1,1,9,11),torch.ones(1,1,2,11)),dim=2)
mask[:,:,-2:,-5:-2]=0
# print(mask)
# input()
noise=np.zeros(shape=(11,11))
noise[9,9]=1
noise[9,4]=2
noise[10,4]=1
noise[9,10]=1
# print(noise)
# input()
noise=torch.from_numpy(noise).float()

#优化器
opt=torch.optim.Adam([v],lr=1e-2)
#两个损失函数的比例
alpha=1
belta=1000
epochs=2000
#加载本地替代模型
targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
for parm in targetF.parameters():
    parm.requires_grad=False

for epoch in range(epochs):
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_both_loss = []
    epoch_correct = 0.
    for i,(x,y) in enumerate(MyDataLoader):
        x=x.float()
        y=y.float()
        # print(x.shape,y.shape)
        pertubation=torch.relu(v*mask)+noise
        adv_x=x+pertubation
        adv_pred=targetF(StandardScale(adv_x))
        correct = torch.sum((adv_pred < 0.5)).detach().numpy()
        epoch_correct += correct
        loss1=torch.sum(pertubation)/len(x)
        # print(loss1)
        loss2=torch.sum(torch.clamp(0.5-adv_pred,min=0.))/len(x)

        loss=alpha*loss1+belta*loss2
        epoch_loss1.append(loss1.detach().numpy())
        epoch_loss2.append(loss2.detach().numpy())
        epoch_both_loss.append(loss.detach().numpy())
        opt.zero_grad()
        loss.backward()
        # print(v.grad.data)
        opt.step()

    # print(v[0,0,:,:])
        # print(loss.detach().numpy())
    #测试:
    with torch.no_grad():
        test_data=test_data.float()
        pertubation = torch.relu(v).int() * mask + noise
        test_adv_data=test_data+pertubation
        test_adv_pred=targetF.forward(StandardScale(test_adv_data))
        correct=torch.sum((test_adv_pred < 0.5)).detach().numpy()
        test_acc=correct/len(test_data)
    #增加的扰动量和扰动比例
    pertubation = torch.relu(v).int() * mask + noise
    aver_overhead=torch.sum(pertubation).detach().numpy()
    overhead_ratio=torch.sum(pertubation)/(torch.sum(train_data)/len(train_data)).detach().numpy()
    print("Epoch:%d || 扰动量:%.3f || 扰动比例:%.3f || 测试准确率:%.3f"%(epoch,aver_overhead,overhead_ratio,test_acc))
    print("Epoch:%d || all_loss:%.3f || overHeadLoss:%.3f || advLoss:%.3f || acc:%.3f" % (
        epoch, np.mean(epoch_both_loss), np.mean(epoch_loss1), np.mean(epoch_loss2), epoch_correct / len(train_label)))
    # input()
pertubation=torch.relu(v).int()*mask + noise
print(pertubation[0,0,:,:])

with open("iter_dataset/markov_universal_1000_trainLoop2000_1_1000_selfDefined_v6","wb") as f:
    pickle.dump(pertubation,f)
