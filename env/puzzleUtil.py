'''
函数说明: 
Author: hongqing
Date: 2021-07-15 10:58:38
LastEditTime: 2021-07-23 16:22:37
'''
import numpy as np
from torch.nn.functional import one_hot

class Util:
    def __init__(self,nRow=5,nCol=6,colorSize=6):
        self.rowSize = nRow
        self.colSize = nCol
        self.colorSize=colorSize
        self.limit=3
    #重复版面优化
    def boardTrans(self,s):
        record = []
        pos= []
        for i in s:
            if(not i in record):
                record.append(i)
                pos.append(np.where(s==i))
                if(len(record)==self.colorSize):
                    break
        index = 0
        res = np.zeros(s.shape[0])
        for i in pos:
            index+=1
            for j in i:
                res[j]=index
        return res

    def getLimit(self,targetPos):
        res = []
        x,y = targetPos
        size = (self.colSize-1)*self.rowSize
        if(y<self.colSize-1):
            res.append(x*(self.colSize-1)+y)
        if(y>0):
            res.append(x*(self.colSize-1)+y-1)
        if(x<self.rowSize-1):
            res.append(size + x*self.colSize + y)
        if(x>0):
            res.append(size + (x-1)*self.colSize + y)
        return res
    #3x3
    def getLimitTiny(self,targetPos):
        res = []
        x,y = targetPos
        size = (self.colSize-1)*self.rowSize
        if(y<self.colSize-1):
            res.append(x*(self.colSize-1)+y)
        if(y>0):
            res.append(x*(self.colSize-1)+y-1)
        if(x<self.rowSize-1):
            res.append(size + x*self.colSize + y)
        if(x>0):
            res.append(size + (x-1)*self.colSize + y)
        return res

    def onehot(self,data,colorsize=6):
        res = np.array([])
        for i in data:
            tmp=np.zeros(colorsize)
            tmp[int(i-1)]=1
            res=np.concatenate((res,tmp),0)
        return res

    #重复版面优化2d 去除无用combo
    def boardTrans2d(self,s,prune=False):
        record = []
        pos= []
        for k in s:
            for i in k:
                if(not i in record):
                    record.append(i)
                    pos.append(np.where(s==i))
                    if(len(record)==self.colorSize):
                        break
        index = 0
        res = np.zeros(s.shape)
        pruned = 0
        for i in pos:
            if(prune):
                if(len(i[0])<self.limit):
                    pruned +=1
                    continue
            index+=1
            for j in range(i[0].shape[0]):
                x=i[0][j]
                y=i[1][j]
                res[x][y]=index
        return res,pruned
   
    # 补0
    def padding(self,s):
        row,col = s.shape
        pad = np.zeros((col-row,col))
        return np.vstack((s,pad))
        
    def one_hot2d(self,s):
        pos = {}
        res = np.zeros((self.colorSize,s.shape[0],s.shape[1]))
        for i in range(1,self.colorSize+1):
            pos = np.where(s==i)
            for j in range(len(pos[0])):
                x=pos[0][j]
                y=pos[1][j]
                res[i-1][x][y]=1
        return res

    # 自动优化器
    def autoOptim(self,s):
        if(self.rowSize>self.colSize):
            return
        s,pruned = self.boardTrans2d(s,True)
        # self.colorSize-=pruned
        if(self.rowSize<self.colSize):
            s = self.padding(s)

        transS = self.one_hot2d(s)
        return transS,self.colorSize
