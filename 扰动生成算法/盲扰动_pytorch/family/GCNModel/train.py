"""模型训练"""
import torch
import torch.nn as nn
# from utils import *
from GCN_Model import *
from mydataset import *
from torch.utils.data import DataLoader
import torch.optim as optimizer
import pickle
import os
import matplotlib.pyplot as plt
import random
import time
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Config = {
    'learning_rate' : 0.01,
    'batch_size' : 200,
    'accumulation_steps' : 8,
    'epoches' : 100,
    'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'save_folder' : './model',
    'input_dim' : 11,
    'output_dim1' : 256,
    'output_dim2' : 512,
    'output_dim3' : 256,
    'output_dim4' : 128,
    'load_model' :'./GCN_model_family.pkl'
}


#加载数据
target_data_path="../dataset/trainGCN_Family"
with open(target_data_path, "rb") as f:
    all_data=pickle.load(f)

#划分数据集,训练集+验证集+测试集:9:0:1
random.shuffle(all_data)
train_data=all_data[:int(len(all_data)*0.9)]
test_data=all_data[int(len(all_data)*0.9):]
print(len(all_data))
print(len(train_data),len(test_data))

#
train_loader=DataLoader(Mydataset(train_data),batch_size=Config['batch_size'], shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
test_loader=DataLoader(Mydataset(test_data),batch_size=Config['batch_size'], shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
test_loader_evaluate=DataLoader(Mydataset(test_data),batch_size=len(test_data),shuffle=True, pin_memory=True, collate_fn=collate, drop_last=False)
#加载模型
device = Config['device']
model = GCN_Model(Config['input_dim'], Config['output_dim1'], Config['output_dim2'], Config['output_dim3'], Config['output_dim4'])
model.to(device)

#模型配置
criterion=nn.BCEWithLogitsLoss() #损失
opt=optimizer.Adam(model.parameters(),lr=1e-2)
def train(epochs):
    epoch_loss=[]
    epoch_acc=[]
    epoch_test_loss=[]
    epoch_test_acc=[]
    for epoch in range(epochs):
        # model.train()
        train_loss=[]
        train_val=[]
        train_correct=0.
        for i,(x,y) in enumerate(train_loader):
            y=y.float()
            x,y=x.to(device),y.to(device)
            pred=model(x)
            prob=torch.sigmoid(pred)
            # print((prob >= 0.5)[:10])
            # print(y[:10])
            # print()
            # print(((prob >= 0.5) == y)[:10])
            # print(torch.sum(((prob >= 0.5) == y)[:10]))
            # input()
            correct=torch.sum((prob>=0.5)==y).detach().cpu().numpy()
            train_correct=train_correct+correct
            loss=criterion(pred,y)
            train_loss.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
        epoch_loss.append(np.array(train_loss).mean())
        epoch_acc.append(train_correct/len(train_data))


        #测试
        # model.eval()
        with torch.no_grad():
            test_correct=0.
            test_loss=[]
            for i,(test_x,test_y) in enumerate(test_loader):
                test_y=test_y.float()
                test_x,test_y=test_x.to(device),test_y.to(device)
                pred=model(test_x)
                prob=torch.sigmoid(pred)
                loss=criterion(pred,test_y)
                test_loss.append(loss.detach().cpu().numpy())

                correct=torch.sum((prob>=0.5)==test_y).detach().cpu().numpy()
                test_correct=test_correct+correct
                # print("Label",test_y[:10].reshape(-1))
                # print("pred:",(pred>0.5)[:10].reshape(-1))
            epoch_test_acc.append(test_correct/len(test_data))
            epoch_test_loss.append(np.array(test_loss).mean())
        print("Epoch:%d || train loss:%f  || train acc:%f || test loss:%f || test acc:%f" % (epoch, epoch_loss[-1], epoch_acc[-1],epoch_test_loss[-1], epoch_test_acc[-1]))
        if epoch_test_acc[-1]>0.83:
            break
    torch.save(model,"GCN_MAMA_FAMILY_20000.pt")

    #计算指标
    test_x,test_y=next(iter(test_loader_evaluate))
    print("Test Sample Num:",len(test_y))
    test_y=test_y.float()
    test_x=test_x.to(device)
    test_y=test_y.to(device)
    test_pred=torch.sigmoid(model(test_x))>=0.5  #预测结果eg:[True,False,True,...]
    TP = torch.sum(torch.multiply(test_y, test_pred)).detach().cpu().numpy()
    FP = torch.sum(torch.logical_and((test_pred == 1), (test_y == 0))).detach().cpu().numpy()
    FN = torch.sum(torch.logical_and((test_y == 1), (test_pred == 0))).detach().cpu().numpy()
    TN = torch.sum(torch.logical_and((test_y == 0), (test_pred == 0))).detach().cpu().numpy()

    #
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)

    #
    precision= TP/(TP+FP)
    recall= TP/(TP+FN)
    F1_SCORE=2*precision*recall/(precision+recall)
    Accuracy=(TP+TN)/(TP+TN+FP+FN)
    metric={"TP":TP,"FP":FP,"FN":FN,"TN":TN,"TPR":TPR,"FPR":FPR,"TNR":TNR,"FNR":FNR,"precision":precision,
            "recall":recall,"F1-SCORE":F1_SCORE,"Accuracy":Accuracy}
    with open("evaluate_metric_family","wb") as f:
        pickle.dump(metric,f)


    plt.figure(1)
    plt.title("train&val loss")
    plt.plot(epoch_loss,label="train loss")
    plt.plot(epoch_test_loss,label="test loss")
    plt.legend()

    plt.figure(2)
    plt.title("training&val acc")
    plt.plot(epoch_acc,label="train acc")
    plt.plot(epoch_test_acc,label="test acc")
    plt.legend()

    plt.show()

if __name__=="__main__":
    print("Family Level,特征为函数调用次数,样本数为:",len(all_data))
    start=time.time()
    train(100)
    end=time.time()
    print("训练好费时间:",end-start)

























