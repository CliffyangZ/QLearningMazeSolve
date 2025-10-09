import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import clear_output
import os

from environment import Environment
from agent import Agent
from visualize import MazeVisualizer


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
