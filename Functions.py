# kurtosis
import numpy as np
import sklearn.preprocessing
import torch
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler
from torch import nn


def funcKurtosis(y, halfFilterlength):
    y_1 = torch.squeeze(y)
    y_1 = y_1[halfFilterlength:-halfFilterlength]
    y_2 = y_1 - torch.mean(y_1)
    num = len(y_2)
    y_num = torch.sum(torch.pow(y_2, 4)) / num
    std = torch.sqrt(torch.sum(torch.pow(y_2, 2)) / num)
    y_dem = torch.pow(std, 4)
    loss = y_num / y_dem
    return loss




def g_lplq(y, p=1.0, q=2.0):
    p = torch.tensor(p)
    q = torch.tensor(q)
    obj=torch.sign(torch.log(q/p))*(torch.norm(y,p)/torch.norm(y,q)) ** p
    return obj


