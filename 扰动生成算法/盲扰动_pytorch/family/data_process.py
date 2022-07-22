"""
数据处理:
    对既有样本进行划分
        训练目标模型数据:targetF_data:
                       targetF_label:
        训练扰动生成器数据:
                       trainG_data:
                       trainG_label:
        用来测试对抗样本有效性的恶意样本:
                       evaluate_100: 100个能够被目标模型正确分类的恶意样本
                       evaluate_1000: 1000个能够被目标模型正确分类的恶意样本

"""

import pickle
import numpy as np
import torch
import os

def data_process():
    """
    对既有样本进行划分
        训练目标模型数据:targetF_data:
                       targetF_label:
        训练扰动生成器数据:
                       trainG_data:
                       trainG_label:


    :return:
    """
    data_path = "../MamaData/dataset_target_MaMa"
    with open(data_path, "rb") as f:
        ori_data = pickle.load(f)
    # print(ori_data)
    train_data = []
    train_label = []
    for item in ori_data:
        family_call_prob = np.array(item["family_call_prob"].todense())
        family_call_prob=np.expand_dims(family_call_prob,axis=0)

        train_data.append(family_call_prob)
        train_label.append(np.argmax(item["label"]))
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    #Malware_sampel
    Malware_sample=train_data[train_label==0]
    Malware_label=train_label[train_label==0]
    #Bengin sample
    Bengin_sample=train_data[train_label==1]
    Bengin_label=train_label[train_label==1]
    print(Malware_sample.shape)
    print(Malware_label.shape)
    print(Bengin_sample.shape)
    print(Bengin_label.shape)

    #划分
    # targetF_data=np.concatenate([Malware_sample[10000:11000],Bengin_sample[10000:11000]],axis=0)
    # targetF_label=np.concatenate([Malware_label[10000:11000],Bengin_label[10000:11000]],axis=0)
    #
    #
    # trainG_data_small=Malware_sample[12000:13000]
    # trainG_label_samll=Malware_label[12000:13000]

    #数据顺序打乱
    # targetF_index=np.arange(len(targetF_label))
    # np.random.shuffle(targetF_index)
    #
    # targetF_data=targetF_data[targetF_index]
    # targetF_label=targetF_label[targetF_index]

    benginG_data=Bengin_sample[10000:]
    benginG_label=Bengin_label[10000:]



    # print(targetF_data.shape,targetF_label.shape)
    # print(trainG_data_small.shape,trainG_label_samll.shape)
    #
    # #数据存储
    # if not os.path.exists("dataset"):
    #     os.mkdir("dataset")
    # with open("dataset/testF_data","wb") as f:
    #     pickle.dump(targetF_data,f)
    # with open("dataset/testF_label","wb") as f:
    #     pickle.dump(targetF_label,f)
    with open("dataset/benginG_data","wb") as f:
        pickle.dump(benginG_data,f)
    with open("dataset/benginG_label","wb") as f:
        pickle.dump(benginG_label,f)


def select_Malware():
    """
    用来测试对抗样本有效性的恶意样本:
                       evaluate_100: 100个能够被目标模型正确分类的恶意样本
                       evaluate_1000: 1000个能够被目标模型正确分类的恶意样本


    :return:
    """
    data_path="../MamaData/dataset_attack_MaMa"
    with open(data_path,"rb") as f:
        ori_data=pickle.load(f)
    # print(len(ori_data))
    #用来存储既是恶意样本且能够被目标分类模型正确分类为恶意的样本

    X_data=[]
    y_data=[]
    for item in ori_data:
        family_call_prob = np.array(item["family_call_prob"].todense())
        family_call_prob = np.expand_dims(family_call_prob, axis=0)
        X_data.append(family_call_prob)
        y_data.append(np.argmax(item["label"]))
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    print(X_data.shape,y_data.shape)
    Malware_sample=X_data[y_data==0]
    Malware_sample=torch.from_numpy(Malware_sample)
    Malware_sample=Malware_sample.float()
    Markov_Malware_sample=Malware_sample.clone()
    for index in range(Markov_Malware_sample.shape[0]):
        for row in range(Markov_Malware_sample.shape[2]):
            sumer = torch.sum(Markov_Malware_sample[index, :, row, :])
            if sumer != 0:
                Markov_Malware_sample[index,:,row, :] /= sumer
    Malware_label=y_data[y_data==0]
    print(Malware_sample.shape,Malware_label.shape)
    #加载分类模型
    targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
    Malware_pred=np.squeeze(targetF(Markov_Malware_sample))<0.5
    true_Malware_sample=Malware_sample[Malware_pred]
    true_Malware_label=Malware_label[Malware_pred]
    true_Malware_sample=true_Malware_sample.numpy()
    print(true_Malware_sample.shape,true_Malware_label.shape)
    # input()
    #存储数据
    with open("dataset/markov_evaluate_100_X","wb") as f:
        pickle.dump(true_Malware_sample[:100],f)
    with open("dataset/markov_evaluate_100_Y","wb") as f:
        pickle.dump(true_Malware_label[:100],f)
    with open("dataset/markov_evaluate_1000_X","wb") as f:
        pickle.dump(true_Malware_sample[:1000],f)
    with open("dataset/markov_evaluate_1000_Y","wb") as f:
        pickle.dump(true_Malware_label[:1000],f)




if __name__=="__main__":
    pass
    # data_process()
    # select_Malware()