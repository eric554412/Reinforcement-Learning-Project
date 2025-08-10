import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from env_train import StockEnvTrain
from env_trade import StockEnvTrade

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

class ReplayBuffer:
    def __init__(self, capacity = 100000):
        self.buffer = []
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        '''
        如果緩衝區緩衝區超過容量, 則刪除最舊的數據
        新經驗加入尾端
        '''
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        ''''
        隨機抽取batch_size的經驗
        '''
        indices = np.random.choice(len(self.buffer), batch_size, replace = False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype = torch.float32),
            torch.tensor(np.array(actions),  dtype = torch.float32),
            torch.tensor(np.array(rewards), dtype = torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype = torch.float32),
            torch.tensor(np.array(dones), dtype = torch.float32).unsqueeze(1)
        )
    
    def size(self):
        return len(self.buffer)

class OrnsteinUhlenbeckNoise:
    '''
    Ornstein-Uhlenbeck噪音, 用於連續動作空間的探索
    '''
    def __init__(self, action_dim, mu = 0.0, theta = 0.15, sigma = 0.5):
        self.mu = mu                            #OU過程的均值
        self.theta = theta                      #OU過程均值回歸速度
        self.sigma = sigma                      #OU過程的波動幅度
        self.state = np.ones(action_dim) * mu   #每個維度初始值都相同
    
    def reset(self):
        self.state = np.ones_like(self.state) * self.mu
    
    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx     
        return self.state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action = 1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), # 第一個線性層，將狀態映射到隱藏層
            nn.ReLU(),             
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),# 輸出層，將隱藏層映射到動作維度
            nn.Tanh()                 # Tanh 函數，將輸出壓縮到 [-1, 1]
        )
    
    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.net(state) 

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), # 輸入層: 狀態 + 動作, DDPG要同時考慮狀態和動作
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 輸出層: Q值
        )
    
    def forward(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        # 將狀態和動作連接起來
        x = torch.cat([state, action], dim = 1) #狀態和動作拼接
        return self.net(x)

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        '''
        取得目標網路和當前網路的所有權重, 並用zip將兩者打包
        target_param: 目標網路(策略更新結果)的權重
        source_param: 當前網路(策略訓練結果)的權重
        利用目標網路還有當前網路的權重來更新
        '''
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def train_ddpg(env, actor, critic, actor_target, critic_target,
               actor_optimizer, critic_optimizer, model_name="DDPG", iteration = 1,
               num_episodes=10, batch_size=64, gamma=0.99, tau=0.005):
    
    os.makedirs('results', exist_ok = True)
    #添加噪音，適合連續控制的問題
    noise = OrnsteinUhlenbeckNoise(action_dim = env.action_space.shape[0])
    
    replay_buffer = ReplayBuffer()
    progress = tqdm(range(num_episodes), desc = 'Training DDPG')
    rewards = []
    
    for episode in progress:
        state = torch.tensor(env.reset(), dtype = torch.float32)
        done = False
        epiosde_reward = 0

        while not done:
            action = actor(state).detach().numpy().squeeze()    
            noise_sample = noise()
            action = np.clip(action + noise_sample, -1, 1)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype = torch.float32)
            replay_buffer.add(state.numpy(), action, reward, next_state.numpy(), done)
            state = next_state
            epiosde_reward += reward
  
            #當經驗回放緩衝區中的樣本數大於批次大小時，才進行模型更新。
            if replay_buffer.size() > batch_size :
                #從 Replay Buffer 中隨機取出一批數據
                states, actions, batch_rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                #critic更新
                with torch.no_grad():
                    next_actions = actor_target(next_states) #使用 actor target計算下一狀態 action
                    #使用 Critic Target 計算下一步 Q 值, 避免直接使用當前網路，防止更新過快
                    target_q = batch_rewards + gamma * (1 - dones) * critic_target(next_states, next_actions)
                #計算當前狀態的Q值(預期回報)
                q_value = critic(states, actions)
                critic_loss = nn.MSELoss()(q_value, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                #actor更新
                actor_loss = -critic(states, actor(states)).mean() #最大化Q值即最小化 -Q
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                #軟更新目標網路
                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)
        rewards.append(epiosde_reward)
        progress.set_postfix(reward = epiosde_reward)
    actor_path = save_model(actor, model_name + "_actor", iteration)
    critic_path = save_model(critic, model_name + "_critic", iteration)
    
    # plt.plot(rewards)
    # plt.title('DDPG reward')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()
    return actor_path, critic_path

def test_ddpg(env, model_name = "DDPG", iteration = 1):
    '''
    載入訓練好的actor網路
    '''
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim, max_action = 1)
    actor = load_model(actor, model_name + "_actor", iteration)
    actor.eval()
    
    state = torch.tensor(env.reset(), dtype = torch.float32)
    done = False
    asset_curve = []
    
    while not done:
        with torch.no_grad():
            action = actor(state).numpy().reshape(-1)
            # print(f"動作 shape: {action.shape}, 動作: {action}")
        next_state, reward, done, info = env.step(action)
        state = torch.tensor(next_state, dtype = torch.float32)
        asset_curve.append(info['total_asset'])
    
    # plt.plot(asset_curve)
    # plt.title('DDPG equity curve')
    # plt.xlabel('trading day')
    # plt.ylabel('equity')
    # plt.grid(True)
    # plt.show()
    print(f"Test total asset: {info['total_asset']}")
    print(f"Test cost: {info['cost']}")
    print(f"Test sharpe: {info['sharpe']}")
    print('*' * 20)
    return asset_curve 


if __name__ == '__main__':
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/data.csv')
    train_df = df.loc[df['datadate'] < 20160101, :].reset_index(drop = True)
    vadilation_df = df.loc[(df['datadate'] >= 20160101) & (df['datadate'] < 20160401), :].reset_index(drop=True)
    test_df = df.loc[df['datadate'] >= 20160101, :].reset_index(drop = True)
    
    env = StockEnvTrain(train_df, model_name = "DDPG")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    
    # 建立目標網路（Target Network），初始化為與當前網路相同
    actor_target = Actor(state_dim, action_dim)
    critic_target = Critic(state_dim, action_dim)
    # 使用load_state_dict()將當前網路的權重複製給目標網路。
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    
    # 設定優化器
    actor_optimizer = optim.Adam(actor.parameters(), lr = 1e-3)
    critic_optimizer =optim.Adam(critic.parameters(), lr = 1e-3)
    
    # 設定超參數
    num_episodes = 10
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    
    # 訓練DDPG
    # train_ddpg(env, actor, critic, actor_target, critic_target,
    #            actor_optimizer, critic_optimizer,
    #            num_episodes = num_episodes, batch_size = batch_size,
    #            gamma = gamma, tau = tau)   
    
    # 測試DDPG
    env_test = StockEnvTrain(train_df, 'DDPG')
    test_ddpg(env_test, 'DDPG', 1)