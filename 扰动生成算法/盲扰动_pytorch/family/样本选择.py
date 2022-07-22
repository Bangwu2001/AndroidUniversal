import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

def StandScale(ori_data):
    tar_data=np.zeros_like(ori_data)
    for index in range(ori_data.shape[0]):
        for row in range(ori_data.shape[2]):
            sumer = np.sum(ori_data[index, :, row, :])
            if sumer != 0:
                tar_data[index, :, row, :] =ori_data[index,:,row,:]/ sumer

    return tar_data

targetFModel=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")

def featureExactor(model,x):
    x=torch.from_numpy(x).float()
    x = F.relu(model.layer1(x))
    x = F.relu(model.layer2(x))
    x = F.relu(model.layer3(x))
    x = F.relu(model.layer4(x))
    x = F.relu(model.layer5(x))
    x = F.relu(model.layer6(x))
    return x

with open("dataset/trainG_data","rb") as f:
    trainG_malware=pickle.load(f)
with open("dataset/trainG_label","rb") as f:
    trainG_malware_label=pickle.load(f)

with open("dataset/benginG_data","rb") as f:
    trainG_bengin=pickle.load(f)
with open("dataset/benginG_label","rb") as f:
    trainG_bengin_label=pickle.load(f)

print(trainG_malware.shape,trainG_malware_label.shape)
print(trainG_bengin.shape,trainG_bengin_label.shape)
print(trainG_malware_label)
print(trainG_bengin_label)




trainG_malware_feature=featureExactor(targetFModel,StandScale(trainG_malware))
print(trainG_malware_feature.shape)

trainG_bengin_feature=featureExactor(targetFModel,StandScale(trainG_bengin))
print(trainG_bengin_feature.shape)

all_feature=torch.cat((trainG_malware_feature,trainG_bengin_feature),dim=0)
print(all_feature.shape)
all_feature=all_feature.detach().numpy()

all_feature_norm=np.sqrt(np.sum(np.square(all_feature),axis=1,keepdims=True))+1e-10
# input()
embs=all_feature/all_feature_norm
# embs = all_feature / np.linalg.norm(all_feature, axis=1, keepdims=True)  # 对向量归一化
all_sims = np.dot(embs, embs.T)  # 计算所有的特征值归一化的相似性
print(all_sims.shape)
malwareWithBengin=np.argsort(all_sims[:10000,10000:],axis=1)[:,::-1][:,0]
print(malwareWithBengin)


#构建训练G的数据
trainG_data=[]
for i in range(10000):
    sample=np.concatenate((trainG_malware[i,:,:,:],trainG_bengin[malwareWithBengin[i],:,:,:]),axis=0)
    trainG_data.append(sample)

trainG_data=np.array(trainG_data)

with open("dataset/double_trainG","wb") as f:
    pickle.dump(trainG_data,f)
print(trainG_data.shape)


