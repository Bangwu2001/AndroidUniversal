"""
对真实apk进行分类测试
①将原始apk逆编译成smali
②从smali中提取函数调用图
③将函数调用图转换为family粒度的调用次数矩阵
④根据不同分类模型进行后续处理进行分类
"""
from SmaliProcess import extract_Mama_family_feature_for_single_apk
from Classifier import MarkovClassifier
import numpy as np
import GCN_Model
import targetF_Model
from Classifier import DNN,CNN
#1.提取调用次数矩阵
apk_path="apk_data/malware_ori.apk"
depress_path="apk_data/depress/"
Family_times_matrix=extract_Mama_family_feature_for_single_apk(apk_path,depress_path)

#2.构建分类器
Family_times_matrix=np.expand_dims((np.expand_dims(Family_times_matrix,axis=0)),axis=0)
# print(Family_times_matrix)
clf_type="KNN1"
classifier=MarkovClassifier(clf_type)
pred_result=classifier.KNN1(Family_times_matrix)
if pred_result==0:
    print("%s 为恶意软件"%(apk_path))
else:
    print("%s 为良性软件"%(apk_path))


