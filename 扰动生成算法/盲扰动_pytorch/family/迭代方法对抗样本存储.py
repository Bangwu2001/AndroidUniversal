import torch
import numpy as np
import pickle

#1.生成对抗样本


with open("dataset/markov_evaluate_1000_X","rb") as f:
    evaluate_data=pickle.load(f)
with open("dataset/markov_evaluate_1000_Y","rb") as f:
    evaluate_label=pickle.load(f)


with open("iter_dataset/markov_universal_1000_trainLoop2000_1_5000_selfDefined_v4","rb") as f:
    per=pickle.load(f)
per=per.int().float()
per=per.numpy()
print(per[0,0,:,:])
# input()
evaluate_data_adv=evaluate_data+per


aver_overhead=np.sum(evaluate_data_adv-evaluate_data)/len(evaluate_data)
print("平均每个样本的扰动量:",aver_overhead)

#扰动量占原始比例
overhead_ratio=np.sum(evaluate_data_adv-evaluate_data)/np.sum(evaluate_data)
print("总的扰动量比例:",overhead_ratio)

with open("iter_dataset/markov_evaluate_1000_X_sample_trainLoop2000_1_5000_selfDefined_v5","wb") as f:
    pickle.dump(evaluate_data_adv,f)


with open("iter_dataset/markov_evaluate_1000_Y_sample_trainLoop2000_1_5000_selfDefined_v5","wb") as f:
    pickle.dump(evaluate_label,f)
