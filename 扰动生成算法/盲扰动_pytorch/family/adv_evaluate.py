"""
对抗样本测试
"""
import torch
import torch.nn as nn
import numpy as np
import pickle

def StandardScale(data):
    for index in range(data.shape[0]):
        for row in range(data.shape[2]):
            sumer = torch.sum(data[index, :, row, :])
            if sumer != 0:
                data[index, :, row, :] /= sumer
    return data
mask=torch.cat([torch.zeros(size=(1,9,11)),torch.ones(size=(1,2,11))],dim=1)
#1.加载生成器模型
generatorModel=torch.load("generator/markov_generator_1_1000_epoch_10_hingLoss_smallData.pkl")
#2.加载本地替代模型分类器
targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")

#3.加载测试数据
with open("dataset/markov_evaluate_1000_X","rb") as f:
    evaluate_x=pickle.load(f)
with open("dataset/markov_evaluate_1000_Y","rb") as f:
    evaluate_y=pickle.load(f)

evaluate_y=np.expand_dims(evaluate_y,axis=1)
evaluate_x=torch.from_numpy(evaluate_x)
evaluate_y=torch.from_numpy(evaluate_y)
evaluate_x=evaluate_x.float()
evaluate_y=evaluate_y.float()

#干净样本测试
markov_evaluate_x=evaluate_x.clone()
markov_evaluate_x=StandardScale(markov_evaluate_x)
pred=targetF.forward(markov_evaluate_x)
# print(pred.shape)
# print(markov_evaluate_x.shape)
correct=torch.sum((pred>=0.5)==evaluate_y).detach().numpy()
print("干净样本准确率:",correct/len(markov_evaluate_x))


#生成对抗样本
pertubation=generatorModel(evaluate_x)
evaluate_adv_x=evaluate_x+pertubation*mask
evaluate_adv_x=evaluate_adv_x.int().float()

#对抗样本存储
with open("dataset/markov_adv_evaluate_1000_X_1_5000_hingLoss_loop10_smallData","wb") as f:
    pickle.dump(evaluate_adv_x,f)

with open("dataset/markov_adv_evaluate_1000_Y_1_5000_hingLoss_loop10_smallData","wb") as f:
    pickle.dump(evaluate_y,f)



markov_evaluate_adv_x=evaluate_adv_x.clone()
markov_evaluate_adv_x=StandardScale(markov_evaluate_adv_x)
pred=targetF(markov_evaluate_adv_x)
correct=torch.sum((pred>=0.5)==evaluate_y).detach().numpy()
print("对抗样本准确率:",correct/len(markov_evaluate_adv_x))

# 平均每个样本添加的扰动量
aver_overhead=torch.sum(evaluate_adv_x-evaluate_x)/len(evaluate_x)
print("平均每个样本的扰动量:",aver_overhead.detach().numpy())

#扰动量占原始比例
overhead_ratio=torch.sum(evaluate_adv_x-evaluate_x)/torch.sum(evaluate_x)
print("总的扰动量比例:",overhead_ratio.detach().numpy())

print("原始样本:")
print(evaluate_x[0,0,:,:].detach().numpy())
print("对抗样本:")
print(evaluate_adv_x[0,0,:,:].detach().numpy())
print("扰动量:")
print((evaluate_adv_x[0,0,:,:].detach().numpy()-evaluate_x[0,0,:,:].detach().numpy()))


