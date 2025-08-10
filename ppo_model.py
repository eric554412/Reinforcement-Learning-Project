import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_train import StockEnvTrain
import os
from env_trade import StockEnvTrade
import random
from env_validation import StockEnvValidation
os.makedirs("results", exist_ok=True)

'''
流程圖
## PPO 主訓練流程圖（從匯入資料到模型更新）
[0] 讀入 df 並建立環境 StockEnvTrain
        ↓
[1] 初始化模型 ActorCritic & Optimizer
        ↓
[2] for episode in range(num_episodes):
        ↓
[3] env.reset() 初始化一集狀態
        ↓
[4] while not done:
        - 模型 forward 得到 dist, value
        - 從 dist.sample() 取得動作
        - 跟環境互動 step(action)
        - 儲存 state, action, log_prob, reward, done, value 進 memory
        ↓
[5] 一集結束 → compute_returns_and_advantage()
        - 使用 GAE 推回 returns, advantages
        ↓
[6] 呼叫 ppo_update()
        - 多輪 mini-batch 更新策略與價值網路
        - 計算 actor_loss、critic_loss、entropy bonus
        - loss.backward() + optimizer.step()
        ↓
[7] 清空 memory，回到 episode 開頭進行下一集
        ↓
[8] 所有 episode 完成後 → 儲存模型 torch.save()

torch:
只要要丟到torch裡面計算的都要是torch.tensor模樣
torch.stack就是把torch.tensor堆疊到一起例如很多個state
torch.zero_grad() 清空梯度
torch.backward() 反向計算梯度
torch.step() 根據梯度更新 model 的參數
何謂梯度下降:torch.backward()模型會針對每一個參數, 去計算這個參數對loss的影響(導數),再根據這個影響程度去調整參數方向,目的是讓loss變小。
detach:拔掉梯度插頭, PyTorch 就知道這塊是「資料備份」，不是訓練參與者。
'''


def save_model(model, model_name, iteration):
    model_dir = f"models/{model_name}/"
    os.makedirs(model_dir, exist_ok=True)
    save_path = f"{model_dir}/{model_name}_{iteration}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")
    return save_path

def load_model(model, model_name, iteration):
    model_dir = f"models/{model_name}/"
    load_path = f"{model_dir}/{model_name}_{iteration}.pth"
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"模型已載入: {load_path}")
    else:
        print(f"找不到模型: {load_path}")
    return model

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, action, log_prob, reward, done, value):
        '''
        把資料把資料從記憶圖中斷開,否則PyTorch會追蹤整個episode的梯度,導致記憶體爆炸
        '''
        self.states.append(state.detach())  
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())
    
    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []


class ActorCritic(nn.Module):
    '''
    nn.ModuleList讓actor_mean之類的可以當作傳參數計算
    '''
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        '''
        建構Actor(策略網路), Critic(價值網路)
        '''
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  #最後為action_dim, 因為他是輸出動作平均值
        )
        #PPO連續動作的重點, 不是輸出動作而是學習動作的機率分佈
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim)) 
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  #最後為一, 因為他是輸出一個值V(s)
        )
    
    def forward(self, state):
        mean = self.actor_mean(state)                 #策略分布的 mean
        std = self.actor_log_std.exp()                #策略分佈的 std
        dist = torch.distributions.Normal(mean, std)  #建立連續動作分佈
        value = self.critic(state)                    #評估 V(s)
        return dist, value

def compute_returns_and_advantage(rewards, values, dones, gamma = 0.99, gae_lambda = 0.95):
    '''
    計算TD target給Critic學的東西
    計算Advantage給Actor學的東西
    GAE 是從最後一步往回推, 要用backward loop
    '''
    returns = []
    advantages = []
    G = 0               #累積TD target
    A = 0               #累積GAE advantage
    
    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = rewards[i]
            A = 0          #設這個為零(預設最後一項+1為0)
        else:
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            A = delta + gamma * gae_lambda * (1 - dones[i]) * A
            td_target = rewards[i] + gamma * values[i + 1] * (1 - dones[i])
            G = td_target
        
        advantages.insert(0, A)
        returns.insert(0 ,G)
    
    return torch.tensor(returns, dtype = torch.float32), torch.tensor(advantages, dtype = torch.float32)

def ppo_update(memory, model, optimizer, clip_epsilon = 0.1, ent_coef = 0.005, vf_coef = 0.5, n_epochs = 4, batch_size = 32, max_grad_norm = 0.5):
    '''
    ppo:Mini-batch & 多步學習, 每128個state更新一次, 每32個state訓練一次, 這樣每次訓練4輪, 總共訓練16次
    '''
    #取出 Memory 中的資料
    states = torch.stack(memory.states)
    actions = torch.stack(memory.actions)
    old_log_probs = torch.stack(memory.log_probs)
    values = torch.stack(memory.values).squeeze()
    
    #計算 returns & advantages, returns給critic學V(s), advantage給actor學 π(a|s)
    returns, advantages = compute_returns_and_advantage(memory.rewards, memory.values + [values[-1]], memory.dones)
    #advantage 正規化將梯度縮放統一	讓 Actor 的學習穩定, 避免 scale 影響策略更新, 是 PPO 的常用技巧
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    #Mini-batch(把整集資料切成一小塊一小塊（batch）丟進模型訓練) + 多輪訓練(用同一批資料重複訓練好幾輪（Epoch))
    for epoch in range(n_epochs):                      #多輪訓練
        for i in range(0, len(states), batch_size):    #Mini-batch
            #把這一段的狀態的St, at, A, TD_targert抽出來
            batch_states = states[i : i + batch_size]
            batch_actions = actions[i : i + batch_size]
            batch_old_log_probs = old_log_probs[i : i + batch_size]
            batch_advantages = advantages[i : i + batch_size]
            batch_returns = returns[i : i + batch_size]
            
            dist, values = model(batch_states) #用下一期的狀態讓模型forward
            new_log_probs = dist.log_prob(batch_actions).sum(dim = 1) #計算現在策略對這些舊動作的 log 機率
            ratio = torch.exp(new_log_probs - batch_old_log_probs)    #計算重要性比率
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) # 限制策略更新幅度：不要讓策略變化太大
            actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean() #加負號因為actor要最大化reward
            critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)  #最小化MSE_loss
            entropy_loss = dist.entropy().sum(dim = 1).mean()            #這一整批資料中，policy的平均entropy(探索度)，有多高
            #根據每筆資料計算總損失函數(通常是mean),策略學習,價值學習,探索平衡
            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_loss #減掉entropy_loss鼓勵探索
           
            optimizer.zero_grad() #清空梯度, 準備反向傳播
            loss.backward()       #根據loss對模型參數求梯度（反向傳播）
            optimizer.step()      #根據梯度更新 model 的參數 

        
def train_ppo(env, model, optimizer, model_name = 'PPO', iteration = 1, num_episodes=10, n_step = 128):
    memory = Memory()
    progress = tqdm(range(num_episodes), desc='Training PPO')
    rewards = []

    for episode in progress:
        state = torch.tensor(env.reset(), dtype=torch.float32)#每集開始讓環境回到初始狀態（例如資產為初始值、持倉清空）
        done = False                                          #控制是否 episode 結束                      
        step_count = 0                                        # 每集初始化步數
        episode_reward = 0                                    # 每集初始化獎勵
        while not done:
            #開始每天進行策略決策：模型推理 & 抽樣動作
            dist, value = model(state)
            action = dist.sample()
            log_probs = dist.log_prob(action).sum()        #計算這次的策略機率,存入memory成為舊策略機率   
            #與環境互動
            action_numpy = action.detach().numpy()
            next_state, reward, done, info = env.step(action_numpy)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            memory.add(state, action, log_probs, reward, done, value)
            state = next_state  #更新為新環境繼續跑迴圈
            episode_reward += reward
            step_count += 1
            #每n_step次更新一次模型
            if (step_count % n_step == 0) and len(memory.states) > 0:
                ppo_update(memory, model, optimizer)
                memory.clear()  #清空記憶體
        #這集結束後, 計算最後一個狀態的價值函數, 存入memory
        if len(memory.states) > 0:
            with torch.no_grad():
                _, last_value = model(state)
                memory.values.append(last_value)
        #跑完一個episode, 呼叫ppo_update並且使用這集資料來更新策略與價值網路
            ppo_update(memory, model, optimizer)
            memory.clear()
        progress.set_postfix(reward=episode_reward, asset=info['total_asset'])
        rewards.append(episode_reward)

    save_path = save_model(model, model_name, iteration)
    return rewards


              
def test_ppo(env, model_name = 'PPO', iteration = 1):
    state_dim = env.observation_space.shape[0]
    action_dim =env.action_space.shape[0]
    
    model = ActorCritic(state_dim, action_dim)
    model = load_model(model, model_name, iteration)
    model.eval()
    
    state = torch.tensor(env.reset(), dtype = torch.float32)
    
    done = False
    total_reward = 0
    asset_curve = []
    
    while not done:
        with torch.no_grad():
            dist, value = model(state)
            action = dist.mean   #測試時使用平均值（不是 sample)
        next_state, reward, done, info = env.step(action.numpy())
        next_state = torch.tensor(next_state, dtype = torch.float32)
        total_reward += reward
        state = next_state   
        asset_curve.append(info['total_asset'])
    
    print(f"Test total asset: {info['total_asset']}")
    print(f"Test cost: {info['cost']}")
    print(f"Test sharpe: {info['sharpe']}")
    print('*' * 20)
    # plt.plot(asset_curve)
    # plt.title("PPO equity curve")
    # plt.xlabel("trading day")
    # plt.ylabel("equity")
    # plt.grid(True)
    # plt.show()
    return asset_curve




      

            


if __name__ == '__main__':
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/data.csv')
    train_df = df.loc[(df['datadate'] > 20090101) & (df['datadate'] < 20160101), :].reset_index(drop=True)
    test_df = df.loc[(df['datadate'] > 20160101), :].reset_index(drop = True)
    
    env = StockEnvTrain(train_df, 'PPO')
    # trade_env = StockEnvTrade(vadilation_df)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    
    # train_ppo(env, model, optimizer, num_episodes = 10)
    # test_ppo(env, model_name = 'PPO', iteration = 1)
    valid_env = StockEnvValidation(test_df, 'PPO', iteration = 0)
    train_ppo(env, model, optimizer, model_name = 'PPO', iteration = 0, num_episodes = 10)
    test_ppo(valid_env, model_name = 'PPO', iteration = 0)
