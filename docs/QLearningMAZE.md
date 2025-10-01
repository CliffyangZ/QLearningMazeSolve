
建立遊戲環境 (Environment)
除迷宮地圖外，還需要一個環境來對我們做的每個動作給予回饋，就如同小孩的動作與父母的回饋 (之於動作與環境回饋)。建立環境前，需先定義狀態 (State)、動作 (Action) 與獎懲機制 (Reward)。

定義狀態
每一個時間點都會有一個可以描述的狀態，以二維迷宮來說，這個狀態可以定義為 Player 所在的位置 state = (row, column)，若 state 為 (1, 2) 表示 Player 所處狀態為 row = 1 且 column = 2。

定義動作
對迷宮來說，只會有四個動作 up、down、left、right 分別代表上、下、左、右

# Determine the result of an action in this state.
def getNextState(self, state, action):
    row = state[0]
    column = state[1]
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
        # Beyond the boundary or hit the wall.
        if row < 0 or column < 0 or maze[row, column] == 1:
            return [state, False]
        # Goal
        elif maze[row, column] == 2:
            return [nextState, True]
        # Forward
        else:
            return [nextState, False]
    except IndexError as e:
        # Beyond the boundary.
        return [state, False]
up：往上移動，即 row = row - 1。
down：往下移動，即 row = row + 1。
left：往左移動，即 column = column - 1。
right：往右移動，即 column = column + 1。
try ... except：根據現在的狀態 (state) 計算出下一個狀態 (nextState)。
定義獎懲機制
環境執行完對應的動作並取得下一個狀態後，即可針對這些資訊來計算對應的獎懲分數

# Execute action.
def doAction(self, state, action):
    nextState, result = self.getNextState(state, action)
    # No move
    if nextState == state:
        reward = -10
    # Goal
    elif result:
        reward = 100
    # Forward
    else:
        reward = -1
    return [reward, nextState, result]
No move： 碰到牆或是邊界，即執行完動作還停在原地，回饋 -10 分。
Forward： 可以移動，但沒有到達終點，回饋 -1 分。
Goal：到達終點，回饋 100 分。
遊戲環境的使用方法
實作出環境後，我們便只需跟環境說：「Hi，我現在的狀態是 (1, 2)，我要做的動作是 up，請問得到多少回饋分數 (reward)？下個狀態會是什麼 (nextState)？遊戲結束了嗎 (result)？」

# Give the action to the Environment to execute
state = (1, 2)
action = 'up'
environment = Environment()
reward, nextState, result = environment.doAction(state, action)
建立代理人 (Agent)
要真的實現 AI 玩遊戲之前，需要建立一個代理人，讓代理人代替真人去玩遊戲，換句話說：就是讓 AI 自己 (Agent) 與遊戲環境 (Environment) 去互動。而我們的 Q Learning 就是實作在代理人這端，讓代理人可以根據 Q Table 與當前狀態，來決定下一個要執行的動作是什麼，在過程中不停的透過決策與獎勵來更新手上的 Q Table，最後精通遊戲的玩法，即為 Q Learning 的精髓所在。

建立 Q Table
Q Table 是一個狀態 (state) 與動作 (action) 的對應表，紀錄下每個決策預期可護得的獎勵，我們的狀態是 (row, column) 而動作是 up、down、left、right，因此可以將 Q Table 格式定義如下

# 先定義變量
(row, column) 為 Player 所在迷宮的位置
up    := 依過去經驗，在 (row, column) 狀態下執行 up    後，預期可護得的分數
down  := 依過去經驗，在 (row, column) 狀態下執行 down  後，預期可護得的分數
left  := 依過去經驗，在 (row, column) 狀態下執行 left  後，預期可護得的分數
right := 依過去經驗，在 (row, column) 狀態下執行 right 後，預期可護得的分數

# Q Table 為
Q(row, column) := [up, down, left, right]
Q Table 格式不是只能使用陣列的型式表示，只要資料結構能滿足記錄 state 和 action 的對應，都是很好的！

因為剛開始 AI 對環境還不了解，每個狀態、每個動作對 AI 來說都是一樣的，因此將 Q Table 中所有的分數初始化為 0

def initQTable(self):
    Q = np.zeros(self.maze.shape).tolist()
    for i, row in enumerate(Q):
        for j, _ in enumerate(row):
            Q[i][j] = [0, 0, 0, 0] # up, down, left, right
    self.QTable = np.array(Q, dtype='f')
選擇動作
Q Table 代表以往決策的經驗，因此 AI 可以使用當前的狀態去查表得知應該要執行什麼動作，才有可能獲得較高的分數

def getAction(self, eGreddy=0.8):
    if random.random() > eGreddy:
        return random.choice(self.actionList)
    else:
        Qsa = self.QTable[self.state].tolist()
        return self.actionList[Qsa.index(max(Qsa))]
eGreddy：為一機率值，若 random.random() > eGreddy 成立則不參考 Q Table，隨機選一個動作做為決策，這項機制是防止代理人進入有可能出現的無窮迴圈。
更新 Q Table
這裡即為 Q Learning 演算法的核心，更新 Q Table

def updateQTable(self, action, nextState, reward, lr=0.7, gamma=0.9):
    Qs = self.QTable[self.state]
    Qsa = Qs[self.actionDict[action]]
    Qs[self.actionDict[action]] = (1 - lr) * Qsa + lr * (reward + gamma *(self.getNextMaxQ(nextState)))
Qsa 對應公式為 
lr 對應公式為 
reward 對應公式為 
gamma 對應公式為 
getNextMaxQ(nextState) 對應公式為 
代理人與遊戲環境互動
使用 while True 讓代理人待在玩遊戲的迴圈內，直至到達終點 if result，詳細說明可參考程式內註解

initState = (np.where(maze==-1)[0][0], np.where(maze==-1)[1][0])
# Create an Agent
agent = Agent(maze, initState)
# Create a game Environment
environment = Environment()
for j in range(0, 30):
    agent.state = initState
    time.sleep(0.1)
    i = 0
    while True:
        i += 1
        # Get the next step from the Agent
        action = agent.getAction(0.9)
        # Give the action to the Environment to execute
        reward, nextState, result = environment.doAction(agent.state, action)
        # Update Q Table based on Environmnet's response
        agent.updateQTable(action, nextState, reward)
        # Agent's state changes
        agent.state = nextState
        if result:
            print(f' {j+1:2d} : {i} steps to the goal.')
            break
結語
強化學習能從無任何資訊去學出一套規則，就像人從小就會觀察這個世界給我們的回饋，不論這樣的回饋是來自父母、朋友、同學、師長、甚至是陌生人，都會對我們人生中的決策產生一定的影響。每個人手裡都握有一個 Q Table，也都正在努力更新、優化這張表、盡可能地嘗試各種不一樣的機會，都是希望能透過現在的經驗，引導未來的自己邁向心中的那個理想。

wall := 1
path := 0
start := 2
goal := 3

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
 [1,1,1,1,1,1,1,1,1,1,1,1]])

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
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
 
