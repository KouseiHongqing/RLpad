'''
函数说明: 
Author: hongqing
Date: 2021-07-14 10:50:40
LastEditTime: 2021-08-13 15:30:39
'''
import numpy as np
import random
from copy import copy,deepcopy
from queue import Queue
import datetime
time = datetime.datetime.time
class Board():
    def __init__(self,rowSize = 5,colSize =6,colorSize =6,limit =3,limitsteps=60):
        #init 5x6
        self.rowSize = rowSize
        self.colSize = colSize
        self.colorSize = colorSize
        self.limit = limit
        self.limitsteps = limitsteps
        self.steps = 0
        self.maxcomb = 0
        self.dupBoard = {}

        
    ##初始化版面#Full = ture 满combo矩阵 Fall=True 生成0combo矩阵
    def initBoard(self,Fall=True,Full =False):
        self.steps = 0
        #固定生成
        if(Full):
            Fall = False
            bd = np.tile(np.arange(1,self.colorSize+1),int(self.rowSize*self.colSize//self.colorSize))
            np.random.shuffle(bd)
            self.board = bd.reshape(self.rowSize,self.colSize)
        else:
        #随机生成
            self.board = np.random.randint(1,self.colorSize+1,(self.rowSize,self.colSize))
        self.maxcomb = self.getMaxCombo()
        #是否无天降
        if(Fall):
            while(True):
                stepCombo = self.calculate()
                if(stepCombo==0):
                    return
                self.fillBoard()
    
    #联通图是否满足combo条件 返回布尔值并对原数组进行修改(修改处置为-1)
    ##@param res:combo数组,limit:最低消除珠数(默认3)
    def comboDFS(self,res):
        remod = deepcopy(res)
        isCombo = False
        for i in range (self.rowSize):
            for j in range (self.colSize):
                if(remod[i][j]>0):
                    count = 1
                    jj= j +1
                    while(jj<self.colSize and remod[i][jj]>0):
                        count +=1
                        jj +=1
                    if(count>=self.limit):
                        for k in range(count):
                            res[i][j+k] = -1
                            isCombo = True
        for j in range (self.colSize):
            for i in range (self.rowSize):
                if(remod[i][j]>0):
                    count = 1
                    ii = i+1
                    while(ii<len(remod) and remod[ii][j]>0):
                        count+=1
                        ii+=1
                    if(count>=self.limit):
                        for k in range(count):
                            res[i+k][j] = -1
                            isCombo = True
        return isCombo
    #模拟天降 
    def finish(self,):
        for i in range(self.rowSize-1,-1,-1):
            for j in range(self.colSize-1,-1,-1):
                if(self.board[i][j]==0):
                    ii=i-1
                    while(ii>=0):
                        if(self.board[ii][j]>0):
                            self.board[i][j] = self.board[ii][j]
                            self.board[ii][j] = 0
                            break
                        ii -=1
    #0位填随机
    def fillBoard(self,):
        stat = np.argwhere(self.board==0)
        for i,j in stat:
            self.board[i][j] = random.randint(1,self.colorSize)



    def calculate(self,):
        count = 0
        #已访问记录,0=未访问
        visited = np.zeros([self.rowSize,self.colSize])
        #连通集 bfs处理
        unionSet = Queue()
        for i in range(self.rowSize-1,-1,-1):
            for j in range(self.colSize-1,-1,-1):
                #访问过了则跳过
                if(visited[i][j]>0 or self.board[i][j]<=0):
                    continue
                visited[i][j] = 1
                #未访问 添加到访问队列
                unionSet.put([self.board[i][j],i,j])
                # 结果联通图
                res = np.zeros([self.rowSize,self.colSize])
                res[i][j]=1
                #bfs
                while(not unionSet.empty()):
                    x,ii,jj = unionSet.get()
                    #向下
                    if(ii+1<self.rowSize and visited[ii+1][jj]==0):
                        if(self.board[ii+1][jj] == x):
                            unionSet.put([self.board[ii+1][jj],ii+1,jj])
                            res[ii+1][jj]=1
                            visited[ii+1][jj]=1
                    
                    #向左
                    if(jj-1>=0 and visited[ii][jj-1]==0):
                        if(self.board[ii][jj-1] == x):
                            unionSet.put([self.board[ii][jj-1],ii,jj-1])
                            res[ii][jj-1]=1
                            visited[ii][jj-1]=1
                        
                    #向右
                    if(jj+1<self.colSize and visited[ii][jj+1]==0):
                        if(self.board[ii][jj+1] == x):
                            unionSet.put([self.board[ii][jj+1],ii,jj+1])
                            res[ii][jj+1]=1
                            visited[ii][jj+1]=1
                        
                    #向上
                    if(ii-1>=0 and visited[ii-1][jj]==0):
                        if(self.board[ii-1][jj] == x):
                            unionSet.put([self.board[ii-1][jj],ii-1,jj])
                            res[ii-1][jj]=1
                            visited[ii-1][jj]=1
                    
            
                #判断结果联通图是否满足combo条件，若满足修改原数组
                combo = self.comboDFS(res)
                if(combo):
                    count+=1
                    for a in range(self.rowSize-1,-1,-1):
                        for b in range(self.colSize-1,-1,-1):
                            if(res[a][b]<0):
                                self.board[a][b] = 0
            
        
        #如果发生了消除 则进行掉落处理 并且进行递归
        if(count>0):
            self.finish()
            count+=self.calculate()
        return count
    #根据版面算最大combo
    def getMaxCombo(self,):
        mymap = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        for i in self.board:
            for j in i:
                mymap[j]+=1
        res = 0
        for i in mymap:
            if(mymap[i]>=self.limit):
                res += mymap[i]//self.limit
        return res

    #算分
    def score(self,curcombo=0):
        tmp = deepcopy(self.board)
        maxcombo = self.maxcomb
        combo = self.calculate()
        self.board = tmp
        
        if(combo == curcombo):
            return 0,combo
        if(combo < curcombo):
            return -pow(2,curcombo-combo),combo
        #分数计算规则
        reward = pow(2,combo-curcombo)
        if(combo ==maxcombo):
            reward = pow(2,combo-curcombo+1)
        return reward,combo
    
    #根据当前珠子获取可选位置 pos为当前珠
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

    def estimate(self,action,oldPos):
        # 5x6 
        nrow=self.rowSize
        ncol=self.colSize
        if(action<=(ncol-1)*nrow-1):
            row = action // (ncol-1)
            col = action % (ncol-1)
            pos = [row,col]
            targetPos = [row,col+1]
        else:
            action -= (ncol-1)*nrow
            row = action // ncol
            col = action % ncol
            pos = [row,col]
            targetPos = [row+1,col]
        
        self.board[pos[0]][pos[1]],self.board[targetPos[0]][targetPos[1]] = self.board[targetPos[0]][targetPos[1]],self.board[pos[0]][pos[1]]
        if(len(oldPos)<1 or pos==oldPos):
            return targetPos
        return pos

    #pos1为操作珠 isPlay=True的时候 保证操作连贯
    def step(self,pos,action,combo,isPlay=True):
        self.steps+=1
        done = self.steps>=self.limitsteps or combo == self.maxcomb
        # step forward
        targetPos = self.estimate(action,pos)
        #消除检测,算分
        r,combo = self.score(combo)
        s_ = self.board
        canchoose = self.getLimit(targetPos)
        limit=[]
        if(isPlay):
            for i in canchoose:
                if(i != action):
                    limit.append(i)
        return s_, r, done, combo,targetPos,limit


if __name__ == "__main__":
    board  = Board()
    board.initBoard(True,True)
    board.board
    s_, r, done, combo,targetPos,limit = board.step([],26,0)