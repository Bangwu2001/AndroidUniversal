#加载数据
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Conv1D,Lambda
import joblib



#加载数据
# train_data_path="dataset/targetF_data"
# train_label_path="dataset/targetF_label"
# with open(train_data_path,"rb") as f:
#     train_data=pickle.load(f)
# with open(train_label_path,"rb") as f:
#     train_label=pickle.load(f)
# print(train_data.shape)
# for index in range(train_data.shape[0]):
#     for row in range(train_data.shape[2]):
#         sumer=np.sum(train_data[index,:,row,:])
#         if sumer!=0:
#             train_data[index,:,row,:]/=sumer
#
# train_data=train_data.reshape(train_data.shape[0],-1)
# print(train_data.shape,train_label.shape)
#
# # #构建模型 AdaBoost
# #
#
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
# clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),random_state=123)
# clf.fit(train_data,train_label)
# res=clf.predict(train_data)
# print(res.shape)
# print(classification_report(train_label,res,target_names=["Malware","Benign"],digits=4))
# #
# #保存权重
# joblib.dump(clf,"markov_classifier/adaboost")
#
clf=joblib.load("markov_classifier/adaboost")
# res=clf.predict(train_data)
# print(res.shape)
# print(classification_report(train_label,res,target_names=["Malware","Benign"],digits=4))


#加载测试数据
#加载训练数据,仅仅用恶意样本
train_data_path="dataset/markov_evaluate_1000_X"
train_label_path="dataset/markov_evaluate_1000_Y"
with open(train_data_path,"rb") as f:
    train_data=pickle.load(f)
with open(train_label_path,"rb") as f:
    train_label=pickle.load(f)
train_label=np.squeeze(train_label)
print(train_data.shape,train_label.shape)
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=np.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer

train_data=train_data.reshape(train_data.shape[0],-1)
print(train_data.shape,train_label.shape)
#
#在测试数据干净样本上测试
print("干净样本评估")
res=clf.predict(train_data)
print(res.shape)
print(accuracy_score(train_label,res))

#
#对抗样本评估

train_data_path="iter_dataset/markov_evaluate_1000_X_sample_trainLoop2000_1_5000_selfDefined_v3"
train_label_path="iter_dataset/markov_evaluate_1000_Y_sample_trainLoop2000_1_5000_selfDefined_v3"
with open(train_data_path,"rb") as f:
    train_data=pickle.load(f)
with open(train_label_path,"rb") as f:
    train_label=pickle.load(f)
train_label=np.squeeze(train_label)
# train_data=train_data.numpy()
for index in range(train_data.shape[0]):
    for row in range(train_data.shape[2]):
        sumer=np.sum(train_data[index,:,row,:])
        if sumer!=0:
            train_data[index,:,row,:]/=sumer

train_data=train_data.reshape(train_data.shape[0],-1)
print(train_data.shape,train_label.shape)

#在测试数据干净样本上测试
res=clf.predict(train_data)
print(res.shape)
print("对抗样本")
print(accuracy_score(train_label,res))
# print(np.sum(res)/len(train_label))
# print(train_label)
# print(res)
