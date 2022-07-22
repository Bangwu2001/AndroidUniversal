import torch
from torch.autograd import Variable
import numpy as np
import pickle

with open("iter_dataset/markov_universal_1000_trainLoop2000_1_1000","rb") as f:
    data=pickle.load(f)

print(data.int())

