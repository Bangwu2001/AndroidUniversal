import pickle as pkl
import random
import numpy as np
import os
import scipy.sparse as sp


def load_data(path):
    with open(path,"r") as f:
        return eval(f.read())


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 5 every 10 epochs"""
    # lr = args.lr * (0.2 ** (epoch // 10))
    if epoch <= 30:
        lr = learning_rate
    elif epoch > 30 and epoch <= 40:
        lr = learning_rate * 0.2
    elif epoch > 40 and epoch <= 45:
        lr = learning_rate * 0.2 * 0.5
    elif epoch > 45:
        lr = learning_rate * 0.2 * 0.5 * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

