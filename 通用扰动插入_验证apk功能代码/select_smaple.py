from Classifier import MarkovClassifier
import pickle
import numpy as np
#挑选测试apk,用于测试功能完整性
classify=MarkovClassifier("CNN")

#加载数据
with open("dataset_attack_MaMa","rb") as f:
    data=pickle.load(f)

for item in data:
    family_call_prob = np.array(item["family_call_prob"].todense())
    family_call_prob = np.expand_dims(np.expand_dims(family_call_prob, axis=0),axis=0)

    pred=classify.CNN1(family_call_prob)
    print(pred)
    print(item["feature_path"])
    input()