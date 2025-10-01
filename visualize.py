import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import clear_output
import os


class MazeVisualizer:
    """迷宮視覺化工具"""
    
    def __init__(self, maze):
        self.maze = maze
        self.colors = ['white', 'black', 'green', 'red', 'lightblue', 'orange']  # path, wall, start, goal, visited, agent
        self.cmap = ListedColormap(self.colors)
    
    def plot_maze(self, title="Maze", path=None):
        """繪製迷宮"""
        plt.figure(figsize=(10, 10))
        maze_display = self.maze.copy().astype(float)
        
        if path:
            for i, pos in enumerate(path):
                if i == 0:  # 起點
                    continue
                elif i == len(path) - 1:  # 終點
                    continue
                elif maze_display[pos] == 0:
                    maze_display[pos] = 4  # 路徑標記 (lightblue)
        
        plt.imshow(maze_display, cmap=self.cmap, vmin=0, vmax=5)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # 添加路徑步數標記
        if path:
            for i, (row, col) in enumerate(path):
                if i > 0 and i < len(path) - 1:  # 不標記起點和終點
                    plt.text(col, row, str(i), ha='center', va='center', 
                            fontsize=8, fontweight='bold', color='darkblue')
        
        plt.show()
    
    def plot_learning_curve(self, steps_history, title="Learning Progress"):
        """繪製學習曲線"""
        plt.figure(figsize=(12, 6))
        plt.plot(steps_history)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Steps to Goal')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def animate_path(self, path, title="Agent Animation", save_gif=False, filename="maze_animation.gif"):
        """動畫顯示代理人移動路徑"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame):
            ax.clear()
            maze_display = self.maze.copy().astype(float)
            
            # 顯示已走過的路徑
            for i in range(min(frame + 1, len(path))):
                pos = path[i]
                if i == 0:  # 起點
                    continue
                elif i == len(path) - 1 and frame >= len(path) - 1:  # 終點
                    continue
                elif maze_display[pos] == 0:
                    maze_display[pos] = 4  # 已訪問路徑
            
            # 顯示當前代理人位置
            if frame < len(path):
                current_pos = path[frame]
                if maze_display[current_pos] != 3:  # 不覆蓋終點
                    maze_display[current_pos] = 5  # 代理人位置 (orange)
            
            ax.imshow(maze_display, cmap=self.cmap, vmin=0, vmax=5)
            ax.set_title(f"{title} - Step {frame + 1}/{len(path)}")
            ax.grid(True, alpha=0.3)
            
            # 添加步數標記
            if frame < len(path):
                current_pos = path[frame]
                ax.text(current_pos[1], current_pos[0], f"Step {frame + 1}", 
                       ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        anim = animation.FuncAnimation(fig, animate, frames=len(path), 
                                     interval=500, repeat=True, blit=False)
        
        if save_gif:
            anim.save(filename, writer='pillow', fps=2)
            print(f"動畫已保存為: {filename}")
        
        plt.show()
        return anim
    
    def print_path_console(self, path, maze_name="Maze"):
        """在控制台以ASCII藝術顯示路徑"""
        print(f"\n🗺️  {maze_name} 路徑視覺化:")
        print("=" * 50)
        
        maze_display = self.maze.copy()
        
        # 標記路徑
        for i, pos in enumerate(path):
            if i == 0:
                continue  # 起點保持原樣
            elif i == len(path) - 1:
                continue  # 終點保持原樣
            elif maze_display[pos] == 0:
                maze_display[pos] = 9  # 路徑標記
        
        # 字符映射
        char_map = {0: '·', 1: '█', 2: 'S', 3: 'G', 9: '○'}
        
        for row in maze_display:
            line = ""
            for cell in row:
                line += char_map.get(cell, '?') + " "
            print(line)
        
        print("\n圖例: S=起點, G=終點, ○=路徑, █=牆壁, ·=空地")
        print(f"總步數: {len(path)} 步")
        print("=" * 50)