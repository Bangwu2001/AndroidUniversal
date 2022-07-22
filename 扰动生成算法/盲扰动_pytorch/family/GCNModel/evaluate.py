import torch
from GCN_Model import *
import pickle
from mydataset import *
from torch.utils.data import DataLoader
#1.加载模型
model=torch.load("GCN_MAMA_FAMILY_20000.pt")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#2.加载干净样本
#加载数据
evaluate_data_path="../dataset/markov_evaluate_1000_GCN"
with open(evaluate_data_path, "rb") as f:
    evaluate_data=pickle.load(f)
test_loader_evaluate=DataLoader(Mydataset(evaluate_data),batch_size=len(evaluate_data),shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
test_x,test_y=next(iter(test_loader_evaluate))
test_x=test_x.to(device)
test_pred=torch.sigmoid(model.forward(test_x))
correct=torch.sum(test_pred<0.5).detach().cpu().numpy()
print("干净样本准确率:",correct/len(test_y))


#3.对抗样本评估
evaluate_data_path="../dataset/markov_adv_evaluate_GCN_new"
with open(evaluate_data_path, "rb") as f:
    evaluate_data=pickle.load(f)
test_loader_evaluate=DataLoader(Mydataset(evaluate_data),batch_size=len(evaluate_data),shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
test_x,test_y=next(iter(test_loader_evaluate))
test_x=test_x.to(device)
test_pred=torch.sigmoid(model.forward(test_x))
correct=torch.sum(test_pred<0.5).detach().cpu().numpy()
print("对抗样本准确率:",correct/len(test_y))
