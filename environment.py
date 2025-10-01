import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import clear_output
import os

class Environment:
    """遊戲環境類別，處理迷宮互動邏輯"""
    
    def __init__(self, maze):
        self.maze = maze
    
    def getNextState(self, state, action):
        """根據當前狀態和動作計算下一個狀態"""
        row, column = state[0], state[1]
        
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            column -= 1
        elif action == 'right':
            column += 1
            
        nextState = (row, column)
        
        try:
            # 超出邊界或撞牆
            if row < 0 or column < 0 or self.maze[row, column] == 1:
                return [state, False]
            # 到達終點
            elif self.maze[row, column] == 3:
                return [nextState, True]
            # 正常移動
            else:
                return [nextState, False]
        except IndexError:
            # 超出邊界
            return [state, False]
    
    def doAction(self, state, action):
        """執行動作並返回獎勵、下一狀態和是否結束"""
        nextState, result = self.getNextState(state, action)
        
        # 沒有移動（撞牆或邊界）
        if nextState == state:
            reward = -10
        # 到達終點
        elif result:
            reward = 100
        # 正常移動
        else:
            reward = -1
            
        return [reward, nextState, result]