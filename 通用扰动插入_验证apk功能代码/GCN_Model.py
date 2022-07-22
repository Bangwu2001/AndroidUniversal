import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch

class GCN_Model(nn.Module):
    #定义网络的结构
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):
        super(GCN_Model,self).__init__()
        self.conv1 = GraphConv(input_dim,hidden_dim1, activation=nn.ReLU(inplace=True))
        # print(self.conv1)
        self.conv2 = GraphConv(hidden_dim1,hidden_dim2, activation=nn.ReLU(inplace=True))
        self.dense1 = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim3,hidden_dim4),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(hidden_dim4, 1)

    def extract_feature(self, x):
        h = x.ndata['h'].float()
        # print("h",h)
        h1 = self.conv1(x, h)
        # print(print("h1",h1))
        h2 = self.conv2(x, h1)
        x.ndata['h'] = h2

        #平均池化
        hg = dgl.mean_nodes(x, 'h')
        return hg

    def forward(self, x):
        hg = self.extract_feature(x)
        # print("hg",hg)
        hg2 = self.dense1(hg)
        out = self.dense2(hg2)
        return out