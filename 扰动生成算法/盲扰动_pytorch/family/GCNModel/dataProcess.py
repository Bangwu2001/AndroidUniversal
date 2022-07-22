"""family粒度数据处理
每个样本含有三项:
    邻接矩阵
    特征矩阵
    label:one-hot形式"""
import pickle
import numpy as np
import scipy.sparse as sp
def data_process(ori_data_path,ori_label_path,target_data_path):
    """
    以pkl形式存储
    :param ori_data_path: 处理前数据路劲
    :param target_data_path: 处理后数据路径
    :return:
    """
    data=[]
    with open(ori_data_path, "rb") as f:
        ori_data = pickle.load(f)
    with open(ori_label_path,"rb") as f:
        ori_label=pickle.load(f)
    for index in range(ori_data.shape[0]):
        adj_matrix = sp.coo_matrix(ori_data[index,0])
        adj_matrix_array=adj_matrix.todense()
        # 特征矩阵[n*n]
        feature_matrix = np.zeros_like(adj_matrix_array)
        for i in range(adj_matrix.shape[0]):
            feature_matrix[i, i] = adj_matrix_array[i, :].sum() + adj_matrix_array[:, i].sum() - adj_matrix_array[i, i]
            # feature_matrix[i, i]=1
        feature_matrix=sp.coo_matrix(feature_matrix)
        data.append({"adj_matrix":adj_matrix,"feature_matrix":feature_matrix,"label":ori_label[index]})
    with open(target_data_path,"wb") as f:
        pickle.dump(data,f)

if __name__=="__main__":
    ori_data_path="../iter_dataset/markov_evaluate_1000_X_sample_trainLoop2000_1_5000_selfDefined_v3"
    ori_label_path="../iter_dataset/markov_evaluate_1000_Y_sample_trainLoop2000_1_5000_selfDefined_v3"
    target_data_path="../dataset/markov_adv_evaluate_GCN_new"
    data_process(ori_data_path=ori_data_path,ori_label_path=ori_label_path,target_data_path=target_data_path)
    # with open(ori_data_path,"rb") as f:
    #     ori_data=pickle.load(f)
    # with open(ori_label_path,"rb") as f:
    #     ori_label=pickle.load(f)
    # print(ori_label.shape,ori_data.shape)

