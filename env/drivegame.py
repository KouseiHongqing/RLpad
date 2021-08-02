'''
函数说明: 
Author: hongqing
Date: 2021-08-02 13:17:58
LastEditTime: 2021-08-02 14:15:23
'''
'''
函数说明: 
Author: hongqing
Date: 2021-07-13 15:40:23
LastEditTime: 2021-07-30 17:45:18
'''
import numpy as np
import pygame
from pygame.locals import *
import random

agentWidth = 150
memlen = 1
class Generator:
    def __init__(self) -> None:
        self.speed = 1 #速度
        self.width = 200 #宽度
        self.limit = 500 - self.width #
        self.count = 0 #计数器
        self.iter = 40 #移动间隔
        self.dire = 1 #方向 1向右
        self.pos = 0#位置
    def step(self,):
        if(self.count==self.iter):
            if(random.randint(0,1)==1):
                self.dire = -self.dire
            self.iter = random.randint(20,60)
            self.count = 0
        if((self.pos==self.limit and self.dire==1)or(self.pos==0 and self.dire==-1)):
            self.dire = - self.dire
        assert(self.limit%self.speed==0)
        self.pos += self.dire * self.speed
        self.count+=1
        return self.pos

class Player:
    def __init__(self,) -> None:
        self.width = 100
        self.pos = 200
        self.limit = 500-self.width #宽度
        self.speed = 1
    def step(self,dire):
        self.pos += dire * self.speed
        self.pos =min(self.limit,self.pos)
        self.pos =max(0,self.pos)

class Myenv():
    def __init__(self) -> None:
        self.gen = Generator()
        self.player = Player()
        pygame.init()
        self.screen = pygame.display.set_mode((500, 450))
        pygame.display.set_caption("Kousei's game")
        self.line = pygame.image.load(r'E:\RLpaz\data\line.png')
        self.agent = pygame.image.load(r'E:\RLpaz\data\agent.png')
        self.line = pygame.transform.scale(self.line, (self.gen.width, 1))
        self.agent = pygame.transform.scale(self.agent, (self.player.width, 50))
        self.fcclock = pygame.time.Clock()
        self.record = []
        self.game_restart(400)

    def score(self,epos):
        score = 0
        if(self.player.pos<epos):
            score = min(self.player.width +self.player.pos - epos,self.gen.width)
        else:
            score = self.gen.width - (self.player.pos - epos)
        #快速收敛
        score = max(0,score)
        return score/self.gen.width

    def drivescore(self,epos):
        if (self.player.pos > epos and self.player.pos + self.player.width < epos+self.gen.width):
            return 1
        return -1

    def game_restart(self,steps):
        assert(steps<=400)
        self.record = []
        for __ in range(steps):
            self.record.append(self.gen.step())
        self.player.pos = self.record[-1]+self.gen.width/2
        s = self.record[-1]
        playermid = self.player.pos + self.player.width/2
        emid = np.array(s) + self.gen.width/2
        s = playermid-emid
        return s

    def step(self,auto = False,a=1):
        done = False
        #self.fcclock.tick(100)
        self.screen.fill((255,255,255))
        if auto:
            self.player.step(a)
        else:
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[K_a]:
                    self.player.step(-1)
            else:
                if keys_pressed[K_d]:
                    self.player.step(1)
        playerpos = self.player.pos
        self.record.append(self.gen.step())
        if(len(self.record)>400):
            self.record = self.record[1:]
        for index,i in enumerate(self.record):
            self.screen.blit(self.line, (i,400-index))
        self.screen.blit(self.agent, (playerpos,400))
        s_=self.record[-1]
        playermid = self.player.pos + self.player.width/2
        emid = np.array(s_) + self.gen.width/2
        s_ = playermid-emid
        pygame.display.update()
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        scores = self.drivescore(self.record[0])
        done = scores <0
        return s_,scores,done
    
        
        
if __name__=='__main__':
    env = Myenv()
    while True:
        score,_ = env.step()
        if(score>0):
            print(score)