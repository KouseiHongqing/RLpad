'''
函数说明: 
Author: hongqing
Date: 2021-07-13 16:22:36
LastEditTime: 2021-07-19 11:53:48
'''
import numpy as np
import pygame
from pygame.locals import *
import random
import time
# 导入一些常用的函数和常量
from sys import exit


class padEnv:
    def __init__(self, row=6, col=5):
        self.rowSize = row
        self.colSize = col
        # 三消
        self.thres = 3
        #self.board = np.array(row, col)
        # pygame
        self.board = np.random.randint(1,6,(5,6))
        pygame.init()
        self.screen = pygame.display.set_mode((135*row, 135*col))
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

    def step(self,pos):
        self.screen.fill(0)
        for i in range(5):
            for j in range(6):
                photo = self.switch[pos[i][j]]
                self.screen.blit(photo, (i*135, j*135))
        pygame.display.update()

    def getscreen(self,):
        return self.screen

    def gameStart(self,fps=2):
        if(fps<0):
            return
        fcclock = pygame.time.Clock()
        while True:
            if(fps!=0):
                fcclock.tick(fps)
            self.step(self.board)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    
    def gene():
        res = np.random.randint(1,6,(5,6))
        return res

    def setBoard(self,board):
        self.board = board



# pygame.init()
# screen = pygame.display.set_mode((135*6, 135*5))
# pygame.display.set_caption("PadAutomation")
# background = pygame.image.load('E:\\RLpaz\\data\\normal.png').convert()
# red = pygame.image.load(r'E:\RLpaz\data\red.png').convert()
# green = pygame.image.load(r'E:\RLpaz\data\green.png').convert()
# yellow = pygame.image.load(r'E:\RLpaz\data\yellow.png').convert()
# dark = pygame.image.load(r'E:\RLpaz\data\dark.png').convert()
# blue = pygame.image.load(r'E:\RLpaz\data\blue.png').convert()
# pink = pygame.image.load(r'E:\RLpaz\data\pink.png').convert()
# pos = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1], [
#     3, 4, 5, 6, 1, 2], [4, 5, 6, 1, 2, 3], [5, 6, 1, 2, 3, 4]]


# def gene():
#     res = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1], [
#         3, 4, 5, 6, 1, 2], [4, 5, 6, 1, 2, 3], [5, 6, 1, 2, 3, 4]]
#     for i in range(5):
#         for j in range(6):
#             res[i][j] = random.randint(1, 6)
#     return res


# fcclock = pygame.time.Clock()  # 创建一个时间对象
# i = 0

# switch = {1: red,
#           2: green,
#           3: yellow,
#           4: dark,
#           5: blue,
#           6: pink,
#           }
# while True:
#     # 游戏主循环
#     fcclock.tick(1)
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             # 接收到退出事件后退出程序
#             exit()
#     screen.blit(background, (0, 0))
#     p = gene()
#     for i in range(5):
#         for j in range(6):
#             photo = switch[p[i][j]]
#             screen.blit(photo, (i*135, j*135))
#     # 将背景图画上去
#     pygame.display.update()
