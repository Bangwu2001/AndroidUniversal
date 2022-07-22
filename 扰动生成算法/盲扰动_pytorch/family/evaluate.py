"""
验证对抗样本的有效性
"""
import torch
import numpy as np
import pickle

#加载数据
with open("dataset/markov_evaluate_100_X","rb") as f:
    test_data=pickle.load(f)
with open("dataset/markov_evaluate_100_Y","rb") as f:
    test_label=pickle.load(f)



for index in range(test_data.shape[0]):
    for row in range(test_data.shape[2]):
        sumer=np.sum(test_data[index,:,row,:])
        if sumer!=0:
            test_data[index,:,row,:]/=sumer
test_label=np.expand_dims(test_label,axis=1)
test_data=torch.from_numpy(test_data)
test_label=torch.from_numpy(test_label)
model=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
test_data=test_data.float()
test_label=test_label.float()
pred=model(test_data)

correct=torch.sum((pred>=0.5)==test_label).numpy()
acc=correct/len(test_label)
print("makov概率测试集准确率")
print(acc)