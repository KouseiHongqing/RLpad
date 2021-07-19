'''
函数说明: 
Author: hongqing
Date: 2021-07-15 10:58:38
LastEditTime: 2021-07-19 13:06:59
'''
import numpy as np

class Util:
    def __init__(self,nRow=5,nCol=6,colorSize=6):
        self.rowSize = nRow
        self.colSize = nCol
        self.colorSize=colorSize
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
