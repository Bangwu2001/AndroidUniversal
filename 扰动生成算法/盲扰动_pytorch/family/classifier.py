"""
构建各个分类器,用于测试扰动前后的效果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import joblib
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import dgl
import scipy.sparse as sp

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1)
        self.layer2=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1)
        self.layer3=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.layer4=nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1)
        self.layer5=nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,stride=1)
        self.layer6=nn.Flatten()
        self.layer7=nn.Linear(4,1)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=self.layer6(x)
        x=self.layer7(x)
        out=torch.sigmoid(x)
        return out

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

class MarkovClassifier():
    """
    包含各种基于MamaDroid Family粒度的特征训练好的分类器
    训练模型时,训练数据:恶意样本:正常样本=10000:10000
    输入数据shape:[N,1,11,11],数据类型均为numpyArray
    数据标签:
        0:恶意样本
        1:良性样本

    """

    def __init__(self,data):
        # self.clf_type = clf_type
        self.x = data
        self.classification=['CNN1','CNN2','DNN','RF','KNN1','KNN3','AdaBoost','GCN']

    def get_answer(self,clf_type):
        if clf_type == 'CNN1':
            result = self.CNN1(self.x)
        elif clf_type == 'CNN2':
            result = self.CNN2(self.x)
        elif clf_type == 'DNN':
            result = self.DNN(self.x)
        elif clf_type == 'RF':
            result = self.RF(self.x)
        elif clf_type == 'KNN1':
            result = self.KNN1(self.x)
        elif clf_type == 'KNN3':
            result = self.KNN3(self.x)
        elif clf_type == 'AdaBoost':
            result = self.AdaBoost(self.x)
        elif clf_type == 'GCN':
            result = self.GCN(self.x)
        else:
            assert clf_type in self.classification, 'No corresponding classifier %s, please check again'%(clf_type)
        return result

    def StandScale(self, x):
        for index in range(x.shape[0]):
            for row in range(x.shape[2]):
                sumer = np.sum(x[index, :, row, :])
                if sumer != 0:
                    x[index, :, row, :] /= sumer
        return x

    def CNN1(self, x):
        x = self.StandScale(x)
        x=torch.from_numpy(x).float()
        model=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
        pred=model.forward(x)
        result=np.squeeze((pred>=0.5).int().detach().numpy())
        # print(result)
        return result

    def CNN2(self, x):
        x = self.StandScale(x)
        x = torch.from_numpy(x).float()
        model = torch.load("markov_classifier/cnn.pkl")
        pred = model.forward(x)
        result = np.squeeze((pred >= 0.5).int().detach().numpy())
        return result
    def DNN(self, x):
        x = self.StandScale(x)
        x=x.reshape(x.shape[0],-1)
        x = torch.from_numpy(x).float()
        model = torch.load("markov_classifier/dnn.pkl")
        pred = model.forward(x)
        result = np.squeeze((pred >= 0.5).int().detach().numpy())
        return result
    def RF(self, x):
        x = self.StandScale(x)
        x = x.reshape(x.shape[0], -1)
        model = joblib.load("markov_classifier/RF")
        result = model.predict(x)
        return result
    def KNN1(self, x):
        x = self.StandScale(x)
        x = x.reshape(x.shape[0], -1)
        model = joblib.load("markov_classifier/KNN1")
        result = model.predict(x)
        return result
    def KNN3(self, x):
        x = self.StandScale(x)
        x = x.reshape(x.shape[0], -1)
        model = joblib.load("markov_classifier/KNN3")
        result = model.predict(x)
        return result
    def AdaBoost(self, x):
        x = self.StandScale(x)
        x = x.reshape(x.shape[0], -1)
        model = joblib.load("markov_classifier/adaboost")
        result = model.predict(x)
        return result
    def GCN(self, x):
        model = torch.load("GCNModel/GCN_MAMA_FAMILY_20000.pt")
        model = model.cpu()
        result=[]
        for i in range(x.shape[0]):
            adj_matrix = x[i, 0]
            feature_matrix = np.zeros_like(adj_matrix)
            for i in range(adj_matrix.shape[0]):
                feature_matrix[i, i] = adj_matrix[i, :].sum() + adj_matrix[:, i].sum() - adj_matrix[i, i]
            feature_matrix = sp.coo_matrix(feature_matrix)
            adj_matrix = sp.coo_matrix(adj_matrix)
            g = dgl.from_scipy(adj_matrix)
            g = dgl.add_self_loop(g)
            g.ndata["h"] = torch.from_numpy(feature_matrix.todense())
            pred = (torch.sigmoid(model.forward(g))[0] >= 0.5).int().detach().numpy()[0]
            result.append(pred)
        return np.array(result)





if __name__=="__main__":
    #加载x
    data_path = "dataset/markov_evaluate_1000_X"
    label_path = "dataset/markov_evaluate_1000_Y"
    with open(data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(label_path, "rb") as f:
        train_label = pickle.load(f)

    classifiers = ['CNN1','CNN2','DNN','RF','KNN1','KNN3','AdaBoost','GCN']
    myClasss=MarkovClassifier(train_data)
    for method in classifiers:
        print('Now test method:',method)
        result = myClasss.get_answer(method)
        # print(result)
        accuracy = accuracy_score(result, train_label)
        print('The accuracy of method %s is %f'%(method,accuracy))

    with open('log.txt','a+') as f:
        f.write('the classifier method: '+method+' the accuracy is :'+ str(result)+'\n')