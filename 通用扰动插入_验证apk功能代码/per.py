"""
查看通用扰动调用次数矩阵形式
"""

import pickle




with open("universal_pertubation/per/markov_universal_1000_trainLoop2000_1_1000_selfDefined_v3","rb") as f:
    per=pickle.load(f)
print(per)