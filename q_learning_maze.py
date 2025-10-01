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

def solve_maze_qlearning(maze, episodes=100, lr=0.7, gamma=0.9, epsilon=0.9, epsilon_decay=0.995):
    """使用Q-Learning解決迷宮問題"""
    
    # 找到起始位置
    start_pos = tuple(np.argwhere(maze == 2)[0])
    
    # 創建環境和代理人
    environment = Environment(maze)
    agent = Agent(maze, start_pos)
    
    steps_history = []
    
    print(f"開始訓練 - 迷宮大小: {maze.shape}")
    print(f"起始位置: {start_pos}")
    
    for episode in range(episodes):
        agent.state = start_pos
        steps = 0
        current_epsilon = epsilon * (epsilon_decay ** episode)
        
        while True:
            steps += 1
            
            # 選擇動作
            action = agent.getAction(current_epsilon)
            
            # 執行動作
            reward, nextState, result = environment.doAction(agent.state, action)
            
            # 更新Q表格
            agent.updateQTable(action, nextState, reward, lr, gamma)
            
            # 更新狀態
            agent.state = nextState
            
            # 檢查是否到達終點
            if result:
                steps_history.append(steps)
                if (episode + 1) % 10 == 0:
                    print(f'Episode {episode + 1:3d}: {steps:3d} steps to goal (ε={current_epsilon:.3f})')
                break
            
            # 防止無限循環
            if steps > 1000:
                steps_history.append(1000)
                break
    
    return agent, steps_history

def get_optimal_path(agent, maze, show_steps=False):
    """獲取最優路徑"""
    start_pos = tuple(np.argwhere(maze == 2)[0])
    environment = Environment(maze)
    
    path = [start_pos]
    agent.state = start_pos
    
    if show_steps:
        print(f"\n🚶 代理人移動路徑:")
        print(f"Step 1: {start_pos} (起點)")
    
    for step in range(1000):  # 防止無限循環
        action = agent.getAction(1.0)  # 完全貪婪策略
        reward, nextState, result = environment.doAction(agent.state, action)
        agent.state = nextState
        path.append(nextState)
        
        if show_steps:
            status = "到達終點!" if result else f"移動 {action}"
            print(f"Step {step + 2}: {nextState} ({status})")
        
        if result:
            break
    
    return path

def get_training_path(agent, maze, episode_num=None):
    """獲取訓練過程中的一次完整路徑（用於演示）"""
    start_pos = tuple(np.argwhere(maze == 2)[0])
    environment = Environment(maze)
    
    path = [start_pos]
    agent.state = start_pos
    
    print(f"\n🎮 Episode {episode_num} 路徑追蹤:")
    print(f"起點: {start_pos}")
    
    for step in range(1000):  # 防止無限循環
        action = agent.getAction(0.1)  # 低探索率以展示學習效果
        reward, nextState, result = environment.doAction(agent.state, action)
        agent.state = nextState
        path.append(nextState)
        
        print(f"Step {step + 1}: {action} -> {nextState} (reward: {reward})")
        
        if result:
            print(f"🎯 到達終點! 總步數: {len(path)}")
            break
        
        if step > 50:  # 避免輸出過長
            print("... (路徑過長，省略中間步驟)")
            break
    
    return path

def main():
    """主函數"""
    
    # 定義兩個迷宮
    maze_10x10 = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,0,0,0,2,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,0,1,1,1,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,0,0,0,0,3,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1]
    ])

    maze_25x25 = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,2,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1],
        [1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1],
        [1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0,1,1],
        [1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1],
        [1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    # 創建視覺化工具
    viz_10x10 = MazeVisualizer(maze_10x10)
    viz_25x25 = MazeVisualizer(maze_25x25)
    
    print("=" * 60)
    print("Q-Learning 迷宮求解器")
    print("=" * 60)
    
    # 解決10x10迷宮
    print("\n🎯 解決 10x10 迷宮...")
    agent_10x10, steps_10x10 = solve_maze_qlearning(
        maze_10x10, episodes=200, lr=0.7, gamma=0.9, epsilon=0.9
    )
    
    # 獲取最優路徑並顯示詳細步驟
    print("\n📍 獲取最優路徑...")
    optimal_path_10x10 = get_optimal_path(agent_10x10, maze_10x10, show_steps=True)
    
    print(f"\n✅ 10x10迷宮完成!")
    print(f"   最終路徑長度: {len(optimal_path_10x10)} 步")
    print(f"   平均步數(最後10次): {np.mean(steps_10x10[-10:]):.1f}")
    
    # 顯示控制台路徑視覺化
    viz_10x10.print_path_console(optimal_path_10x10, "10x10 迷宮")
    
    # 解決25x25迷宮
    print("\n🎯 解決 25x25 迷宮...")
    agent_25x25, steps_25x25 = solve_maze_qlearning(
        maze_25x25, episodes=500, lr=0.5, gamma=0.95, epsilon=0.9
    )
    
    # 獲取最優路徑
    print("\n📍 獲取最優路徑...")
    optimal_path_25x25 = get_optimal_path(agent_25x25, maze_25x25, show_steps=False)  # 25x25太長，不顯示詳細步驟
    
    print(f"\n✅ 25x25迷宮完成!")
    print(f"   最終路徑長度: {len(optimal_path_25x25)} 步")
    print(f"   平均步數(最後10次): {np.mean(steps_25x25[-10:]):.1f}")
    
    # 顯示控制台路徑視覺化
    viz_25x25.print_path_console(optimal_path_25x25, "25x25 迷宮")
    
    # 顯示結果
    print("\n" + "=" * 60)
    print("📊 訓練結果摘要")
    print("=" * 60)
    print(f"10x10 迷宮 - 最優解: {len(optimal_path_10x10)} 步")
    print(f"25x25 迷宮 - 最優解: {len(optimal_path_25x25)} 步")
    
    # 視覺化結果（可選）
    try:
        print("\n📈 繪製學習曲線...")
        viz_10x10.plot_learning_curve(steps_10x10, "10x10 Maze Learning Progress")
        viz_25x25.plot_learning_curve(steps_25x25, "25x25 Maze Learning Progress")
        
        print("🗺️  顯示迷宮和最優路徑...")
        viz_10x10.plot_maze("10x10 Maze with Optimal Path", optimal_path_10x10)
        viz_25x25.plot_maze("25x25 Maze with Optimal Path", optimal_path_25x25)
        
        # 動畫顯示
        print("\n🎬 創建動畫...")
        user_input = input("是否要顯示動畫? (y/n): ").lower().strip()
        if user_input == 'y' or user_input == 'yes':
            print("正在創建 10x10 迷宮動畫...")
            anim_10x10 = viz_10x10.animate_path(optimal_path_10x10, "10x10 Maze Agent Animation")
            
            save_gif = input("是否保存為GIF文件? (y/n): ").lower().strip()
            if save_gif == 'y' or save_gif == 'yes':
                viz_10x10.animate_path(optimal_path_10x10, "10x10 Maze Agent Animation", 
                                     save_gif=True, filename="maze_10x10_animation.gif")
            
            # 對於25x25迷宮，由於路徑較長，詢問是否要顯示
            show_25x25 = input("是否要顯示 25x25 迷宮動畫? (路徑較長，可能需要更多時間) (y/n): ").lower().strip()
            if show_25x25 == 'y' or show_25x25 == 'yes':
                print("正在創建 25x25 迷宮動畫...")
                anim_25x25 = viz_25x25.animate_path(optimal_path_25x25, "25x25 Maze Agent Animation")
                
                save_gif_25 = input("是否保存 25x25 動畫為GIF文件? (y/n): ").lower().strip()
                if save_gif_25 == 'y' or save_gif_25 == 'yes':
                    viz_25x25.animate_path(optimal_path_25x25, "25x25 Maze Agent Animation", 
                                         save_gif=True, filename="maze_25x25_animation.gif")
        
    except Exception as e:
        print(f"視覺化功能需要GUI環境: {e}")
    
    # 演示訓練過程中的路徑追蹤
    print("\n🔍 演示訓練過程路徑追蹤...")
    demo_input = input("是否要看訓練過程中的路徑演示? (y/n): ").lower().strip()
    if demo_input == 'y' or demo_input == 'yes':
        print("重新創建代理人進行演示...")
        demo_agent = Agent(maze_10x10, tuple(np.argwhere(maze_10x10 == 2)[0]))
        
        # 簡單訓練幾個回合
        environment = Environment(maze_10x10)
        for episode in range(5):
            demo_agent.state = tuple(np.argwhere(maze_10x10 == 2)[0])
            for _ in range(100):
                action = demo_agent.getAction(0.8)
                reward, nextState, result = environment.doAction(demo_agent.state, action)
                demo_agent.updateQTable(action, nextState, reward)
                demo_agent.state = nextState
                if result:
                    break
        
        # 顯示訓練後的路徑
        demo_path = get_training_path(demo_agent, maze_10x10, "演示")
    
    return agent_10x10, agent_25x25, steps_10x10, steps_25x25

if __name__ == "__main__":
    main()
