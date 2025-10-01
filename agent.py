import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import clear_output
import os

class Agent:
    """Q-Learning代理人"""
    
    def __init__(self, maze, initState):
        self.maze = maze
        self.state = initState
        self.actionList = ['up', 'down', 'left', 'right']
        self.actionDict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.initQTable()
    
    def initQTable(self):
        """初始化Q表格"""
        Q = np.zeros(self.maze.shape).tolist()
        for i, row in enumerate(Q):
            for j, _ in enumerate(row):
                Q[i][j] = [0, 0, 0, 0]  # up, down, left, right
        self.QTable = np.array(Q, dtype='f')
    
    def getAction(self, eGreedy=0.8):
        """根據ε-greedy策略選擇動作"""
        if random.random() > eGreedy:
            return random.choice(self.actionList)
        else:
            Qsa = self.QTable[self.state].tolist()
            return self.actionList[Qsa.index(max(Qsa))]
    
    def getNextMaxQ(self, nextState):
        """獲取下一狀態的最大Q值"""
        return max(self.QTable[nextState])
    
    def updateQTable(self, action, nextState, reward, lr=0.7, gamma=0.9):
        """更新Q表格"""
        Qs = self.QTable[self.state]
        Qsa = Qs[self.actionDict[action]]
        Qs[self.actionDict[action]] = (1 - lr) * Qsa + lr * (reward + gamma * self.getNextMaxQ(nextState))
