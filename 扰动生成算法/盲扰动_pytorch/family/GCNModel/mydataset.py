import dgl as dgl
import torch.utils.data as data
import numpy as np
import os
import torch
import scipy.sparse as sp

def default_loader(feature):
    return torch.from_numpy(feature)

def construct_graph(adj_mat, node_feature):
    g = dgl.from_scipy(adj_mat)
    g = dgl.add_self_loop(g)

    g.ndata['h'] = torch.from_numpy(node_feature)
    return g

def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.unsqueeze(torch.tensor(labels, dtype=torch.float),dim=1)

#Pytorch数据加载
class Mydataset(data.Dataset):
    """
    all_data:列表,里面含有邻接矩阵,特征矩阵,样本label
    """
    def __init__(self, all_data, loader=default_loader, construct_graph=construct_graph):
        self.all_data = all_data
        # self.targets = label_y
        self.loader = loader
        self.construct_graph = construct_graph

    def __getitem__(self, index):

        adj_matrix = self.all_data[index]["adj_matrix"]
        node_features = self.all_data[index]["feature_matrix"].todense()

        graph = self.construct_graph(adj_matrix, node_features)
        target = self.all_data[index]["label"]
        return graph, target
        
    def __len__(self):
        return len(self.all_data)



if __name__=="__main__":
    pass





