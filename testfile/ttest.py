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
a = nn.Sequential(nn.Conv2d(7,30,(4,4),padding=1))
cn1 = nn.Conv2d(7,63,6,padding=3)
cn2 = nn.Conv2d(63,63,5,padding=2)
cn3 = nn.Conv2d(63,126,4,padding=1)
cn4 = nn.Conv2d(126,252,3,padding=1)
cn5 = nn.Conv2d(252,2,1,padding=1)
c=torch.randn(1,7,5,6)
a.forward(c).shape
c=torch.from_numpy(np.arange(32).reshape(-1,4,4)).type(torch.float32)

c1 = torch.randn(1,7,6,6)
c2 = torch.randn(1,63,7,7)
c3 = torch.randn(1,63,7,7)
c4 = torch.randn(1,126,6,6)
c5 = torch.randn(1,252,6,6)
q1=cn1.forward(c1)
q2=cn2.forward(c2)
q3=cn3.forward(c3)
q4=cn4.forward(c4)
q5=cn5.forward(c5)

# x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        
