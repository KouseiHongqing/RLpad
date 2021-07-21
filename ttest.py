'''
函数说明: 
Author: hongqing
Date: 2021-07-20 17:30:57
LastEditTime: 2021-07-20 17:52:12
'''

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
# 图像上显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure(figsize=(8,6))
plt.title('训练效果')
plt.xlabel('训练次数/1000epoch')
plt.ylabel('得分')
reward = [1,2,3,6,7,11]
a = np.arange(len(reward))
plt.plot(a, reward)
plt.savefig('pic\\1')
plt.show()
