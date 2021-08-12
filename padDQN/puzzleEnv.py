'''
函数说明: 
Author: hongqing
Date: 2021-07-13 16:22:36
LastEditTime: 2021-08-11 17:07:02
'''
import numpy as np
import pygame
# from pygame.locals import *
# 导入一些常用的函数和常量
from sys import exit
from puzzleBoard import Board
from puzzleUtil import Util
class padEnv:
    def __init__(self, row=5, col=6,color=6,thres=3,fps=0,noDup=True,limitsteps=100):
        self.rowSize = row
        self.colSize = col
        self.colSize = color
        self.fps=fps
        self.animation = fps>0
        assert(row>0 and col >0 and color<=6)
        self.ballwidth=135
        # 几消
        self.thres = thres
        #版面
        self.board = Board(row,col,color,thres,limitsteps)
        self.util = Util(row,col,color)#定义util
        #是否有重复
        if(noDup):
            self.board.initBoardnoDup()
        else:
            self.board.initBoard()
    
    def unwrapped(self,):
        # pygame
        if(self.animation):
            pygame.init()
            self.screen = pygame.display.set_mode((self.ballwidth*self.colSize,self.ballwidth*self.rowSize))
            pygame.display.set_caption("PadAutomation")
            background = pygame.image.load(r'E:\RLpaz\data\normal.png').convert()
            red = pygame.image.load(r'E:\RLpaz\data\red.png').convert()
            green = pygame.image.load(r'E:\RLpaz\data\green.png').convert()
            yellow = pygame.image.load(r'E:\RLpaz\data\yellow.png').convert()
            dark = pygame.image.load(r'E:\RLpaz\data\dark.png').convert()
            blue = pygame.image.load(r'E:\RLpaz\data\blue.png').convert()
            pink = pygame.image.load(r'E:\RLpaz\data\pink.png').convert()
            self.switch = {1: red,
                        2: blue,
                        3: green,
                        4: yellow,
                        5: dark,
                        6: pink,
                        }
            self.fcclock = pygame.time.Clock()

    def render(self,):
        #这里应该填充背景 后续补上
        self.screen.fill(0)
        s = self.board.board
        for i in range(self.rowSize):
            for j in range(self.colSize):
                photo = self.switch[s[i][j]]
                self.screen.blit(photo, (j* self.ballwidth,i*self.ballwidth))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def step(self,pos,action,combo,isPlay=False):
        s_, r, done, combo,targetPos,limit = self.board.step(pos,action,combo,isPlay)
        # self.board.initBoardnoDup()
        if(self.animation):
            ##加入刷新率
            if(self.fps>0):
                self.fcclock.tick(self.fps)
            self.render()
        return s_, r, done, combo,targetPos,limit

    def setBoard(self,board):
        self.board = board

    def reset(self,noDup=True):
        if(noDup):
            self.board.initBoardnoDup()
        else:
            self.board.initBoard()

    def getBoard(self,):
        s = self.board.board
        return self.util.autoOptim(s)
        
if __name__=='__main__':
    env = padEnv(fps=-1)
    env.unwrapped()
    board = Board() # 定义版面
    board.initBoardnoDup()
    while True:
        env.step(1,1,1)
