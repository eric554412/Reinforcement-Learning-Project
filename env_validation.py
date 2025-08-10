import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

# 每次交易最多100股
HMAX_NORMALIZE = 100
# 初始化資產
INITIAL_ACCOUNT_BALANCE = 1000000
# 股票數量
STOCK_DIM = 30
# 交易手續費 (0.1%)
TRANSACTION_COST = 0.001
# 獎勵因子
REWARD_SCALING = 1e-4

class StockEnvValidation(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, model_name, turbulence_threshold = 140, day=0, iteration = ''):
        self.day = day
        self.df = df
        self.iteration = iteration
        self.model_name = model_name
        self.turbulence = 0
        self.turbulence_threshold = turbulence_threshold
        
        self.result_dir = os.path.join("results", "validation", f"{self.model_name}_{self.iteration}")
        os.makedirs(self.result_dir, exist_ok = True)
        
        # 動作空間：每隻股票可以買或賣的最大數量
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # 狀態空間
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(211,))
        
        # 初始化狀態
        self.data = self.df[self.df["datadate"] == self.df["datadate"].unique()[self.day]]
        self.terminal = False
        self.reset()

    def sell_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                sell_amount = min(abs(action), self.state[index + STOCK_DIM + 1])
                self.state[0] += self.state[index + 1] * sell_amount * (1 - TRANSACTION_COST)
                self.state[index + STOCK_DIM + 1] -= sell_amount
                self.cost += self.state[index + 1] * sell_amount * TRANSACTION_COST
                self.trade += 1
            else:
                pass
        else:
            # 如果震盪過大，清空所有持倉
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * (1 - TRANSACTION_COST)
                self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * TRANSACTION_COST
                self.state[index + STOCK_DIM + 1] = 0
                self.trade += 1
            else:
                pass
        
        
    def buy_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_COST)
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_COST
            self.trade += 1
        else:
            pass
        
        

    def step(self, actions):
        if self.day >= len(self.df["datadate"].unique()) - 1:
            self.terminal = True

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plot_path = os.path.join(self.result_dir, f'account_value_validation_{self.iteration}.png')
            plt.savefig(plot_path)
            plt.close()

            # 計算最終資產
            end_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(2 * STOCK_DIM + 1)]))

            df_total_value = pd.DataFrame(self.asset_memory, columns = ['account_value'])
            csv_path = os.path.join(self.result_dir, f"account_value_validation_{self.iteration}.csv")
            df_total_value.to_csv(csv_path, index = False)
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4**0.5) * (df_total_value['daily_return'].mean()) / df_total_value['daily_return'].std()

            info = {
                "total_asset": end_total_asset,
                "cost": self.cost,
                'sharpe': sharpe if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0
            }
            return self.state, self.reward, self.terminal, info

        else:
            actions = actions * HMAX_NORMALIZE
            # 判斷是否需要清倉：根據震盪閾值或特定條件清倉
            if self.turbulence > self.turbulence_threshold:
                # 假設震盪超過140時清倉，設定全部賣出最多100股
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)
                # 震盪過大時,不進行交易
            begin_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(2 * STOCK_DIM + 1)]))
            # 先賣後買
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self.sell_stock(index, actions[index])
            for index in buy_index:
                self.buy_stock(index, actions[index])

            # 更新狀態
            self.day += 1
            self.data = self.df[self.df["datadate"] == self.df["datadate"].unique()[self.day]]
            self.turbulence = self.data["turbulence"].values[0]

            self.state = [self.state[0]] + \
                        self.data["adjcp"].tolist() + \
                        list(self.state[(STOCK_DIM + 1):(2 * STOCK_DIM + 1)]) + \
                        self.data["macd"].tolist() + \
                        self.data["rsi"].tolist() + \
                        self.data["cci"].tolist() + \
                        self.data["adx"].tolist() + \
                        self.data["vr"].tolist()
                        
            end_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(2 * STOCK_DIM + 1)]))

            self.asset_memory.append(end_total_asset)
            self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
            self.reward_memory.append(self.reward)

            info = {
                "total_asset": end_total_asset,
                "cost": self.cost,
                'sharpe': None
            }
            return self.state, self.reward, self.terminal, info
   
    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.cost = 0
        self.trade = 0
        self.turbulence = 0
        self.terminal = False
        self.reward_memory = []

        self.data = self.df[self.df["datadate"] == self.df["datadate"].unique()[self.day]]
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data["adjcp"].tolist() + \
                     [0] * STOCK_DIM + \
                     self.data["macd"].tolist() + \
                     self.data["rsi"].tolist() + \
                     self.data["cci"].tolist() + \
                     self.data["adx"].tolist() + \
                     self.data["vr"].tolist()
        return self.state