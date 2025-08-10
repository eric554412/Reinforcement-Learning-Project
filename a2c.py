import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_trade import StockEnvTrade
from env_train import StockEnvTrain
from env_validation import StockEnvValidation
import os

'''
實作A2C的Actor和Critic
PPO架構是A2C的改良版,也是用GAE來計算優勢
只要改update and train就好了
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
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, action, state, log_prob, reward, done, value):
        self.actions.append(action)
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.actions, self.states, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_prob = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        mean = self.actor_mean(state)
        std = self.actor_log_prob.exp()
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(state)
        return dist, value

def compute_returns_and_advantages(rewards, values, dones, gamma = 0.99, gae_lam = 0.97):
    '''
    A2C也是用GAE來計算優勢
    '''
    returns = []
    advantages = []
    G = 0
    A = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = rewards[i]
            A = 0
        else:
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            A = delta + gamma * gae_lam * A
            td_target = rewards[i] + gamma * values[i + 1] * (1 - dones[i])
            G = td_target
        
        returns.insert(0, G)
        advantages.insert(0, A)
    
    return torch.tensor(returns, dtype = torch.float32), torch.tensor(advantages, dtype = torch.float32)

def a2c_update(memory, model, optimizer, ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5):
    '''
    A2C的更新,只有更新一次(無mini-batch, 無clipping)
    '''
    states = torch.stack(memory.states)
    actions = torch.stack(memory.actions)
    old_log_probs = torch.stack(memory.log_probs)
    value = torch.stack(memory.values).squeeze()
    #計算 returns和 advantage   
    returns, advantage = compute_returns_and_advantages(memory.rewards, memory.values, memory.dones)
    std = advantage.std(unbiased = False) 
    advantage = (advantage - advantage.mean()) / (std + 1e-8)
    
    dist, value = model(states)
    new_log_probs = dist.log_prob(actions).sum(dim = 1)
    
    #計算loss
    actor_loss = -(new_log_probs * advantage.detach()).mean()
    critic_loss = nn.MSELoss()(value.squeeze(-1), returns.detach())
    entropy_loss = dist.entropy().sum(dim = 1).mean()
    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

def train_a2c(env, model, optimizer, model_name = 'A2C', iteration = 1, num_episodes = 10, n_step = 10):
    memory = Memory()
    progress = tqdm(range(num_episodes), desc = 'Training A2C')
    for episode in progress:
        state = torch.tensor(env.reset(), dtype = torch.float32)
        done = False
        step_count = 0
        
        while not done:
            if torch.isnan(state).any() or torch.isinf(state).any():
                print("⚠️ Detected NaN or Inf in state:")
                print(state)
                break
            dist, value = model(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            action_nump = action.detach().numpy()
            next_state, reward, done, info = env.step(action_nump)
            next_state = torch.tensor(next_state, dtype = torch.float32)
            memory.add(action, state, log_prob, reward, done, value)
            step_count += 1
            state = next_state
            # 每n_step更新一次
            if (step_count % n_step == 0) and (len(memory.states) > 0):
                with torch.no_grad():
                #計算最後一個狀態的value
                    _, value = model(state)
                    memory.values.append(value)
                a2c_update(memory, model, optimizer)
                memory.clear()
        if len(memory.states) > 0:
            with torch.no_grad():
                #計算最後一個狀態的value
                _, value = model(state)
                memory.values.append(value)
            a2c_update(memory, model, optimizer)
            memory.clear()
        
        progress.set_postfix(total_asset = info['total_asset'], cost = info['cost'], sharpe = info['sharpe'])
    #儲存模型
    save_path = save_model(model, model_name, iteration)
    return save_path
    
def test_a2c(env, model_name = 'A2C', iteration = 1):

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    model = load_model(model, model_name, iteration)
    model.eval()
    
    state = torch.tensor(env.reset(), dtype = torch.float32)
    done = False
    asset_curve = []
    
    while not done:
        with torch.no_grad():
            dist, value = model(state)
            action = dist.mean
        next_state, reward, done, info = env.step(action.numpy())
        next_state = torch.tensor(next_state, dtype = torch.float32)
        state = next_state
        asset_curve.append((info['total_asset'], info['cost'], info['sharpe']))
    asset_curve = np.array(asset_curve)
    print(f'Test total asset: {asset_curve[-1][0]}')
    print(f'Test cost: {asset_curve[-1][1]}')
    print(f'Test sharpe: {asset_curve[-1][2]}')
    print('*' * 20)
    # plt.plot(asset_curve[:, 0])
    # plt.title('Total Asset Curve')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Total Asset')
    # plt.show()
        
                    


if __name__ == '__main__':
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/data.csv')
    test_df = df.loc[(df['datadate'] < 20160101), :].reset_index(drop = True)
    train_df = df.loc[(df['datadate'] >= 20160101), :].reset_index(drop = True)
    
    env = StockEnvTrain(train_df, 'A2C')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    train_a2c(env, model, optimizer, num_episodes = 10)
    trade_env = StockEnvValidation(test_df, 'A2C', iteration = 0)
    test_a2c(trade_env, 'A2C', 0)