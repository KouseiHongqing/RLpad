'''
函数说明: 
Author: hongqing
Date: 2021-07-20 17:30:57
LastEditTime: 2021-07-20 17:52:12
'''

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# a = nn.Sequential(nn.LSTM(4,4,batch_first=True))
a = nn.Sequential(nn.Linear(4,1))
c=torch.from_numpy(np.arange(16).reshape(-1,4,4)).type(torch.float32)
