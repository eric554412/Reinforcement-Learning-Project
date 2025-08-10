import os, time, numpy as np, pandas as pd, torch
from env_train import StockEnvTrain
from env_validation import StockEnvValidation
from env_trade import StockEnvTrade
from ppo_model import train_ppo, test_ppo, ActorCritic as PPOActor
from a2c import train_a2c, test_a2c, ActorCritic as A2CActor
from ddpg import train_ddpg, test_ddpg, Actor as DDPGActor, Critic as DDPGCritic
import matplotlib.pyplot as plt


def get_validation_sharpe(model_name, iteration):
    '''
    計算valid期間的sharpe ratio
    '''
    csv_path = os.path.join(
        "results", "validation",
        f"{model_name}_{iteration}",
        f"account_value_validation_{iteration}.csv")
    
    if not os.path.exists(csv_path):
        print(f"找不到驗證結果:{csv_path}")
        return -9e9
    
    df = pd.read_csv(csv_path)
    df.columns = ['account_value']
    df['daily_return'] = df['account_value'].pct_change(1)
    std = df['daily_return'].std()
    if std == 0 or np.isnan(std):
        return 0
    sharpe = (4 ** 0.5) * df['daily_return'].mean() / std  #63個交易日，大約為一季
    return sharpe

def train_and_validate(train_df, val_df, it, turbulence_threshold):
    '''
    訓練策略並輸出valid期間sharpe最高的演算法
    '''
    env_train = StockEnvTrain(train_df, model_name = "tmp")
    s_dim, a_dim = env_train.observation_space.shape[0], env_train.action_space.shape[0]
    #A2C
    a2c = A2CActor(s_dim, a_dim)
    train_a2c(env_train, a2c, torch.optim.Adam(a2c.parameters(), 1e-4), iteration = it, num_episodes = 50)
    test_a2c(StockEnvValidation(val_df, 'A2C', turbulence_threshold = turbulence_threshold, iteration = it), iteration = it)
    sharpe_a2c = get_validation_sharpe('A2C', iteration = it)
    #PPO
    ppo = PPOActor(s_dim, a_dim)
    train_ppo(env_train, ppo, torch.optim.Adam(ppo.parameters(), 1e-4), iteration = it, num_episodes = 50)
    test_ppo(StockEnvValidation(val_df, 'PPO', turbulence_threshold = turbulence_threshold, iteration = it), iteration = it)
    sharpe_ppo = get_validation_sharpe('PPO', iteration = it)
    #DDPG
    act = DDPGActor(s_dim, a_dim)
    cri = DDPGCritic(s_dim, a_dim)
    train_ddpg(env_train, act, cri, act, cri,
               torch.optim.Adam(act.parameters(), 1e-3),
               torch.optim.Adam(cri.parameters(), 1e-3),
               iteration = it, num_episodes = 50)
    test_ddpg(StockEnvValidation(val_df, 'DDPG', turbulence_threshold = turbulence_threshold, iteration = it), iteration = it)
    sharpe_ddpg = get_validation_sharpe('DDPG', iteration = it)
    
    best = max([("PPO", sharpe_ppo),
            ("A2C", sharpe_a2c),
            ("DDPG", sharpe_ddpg)], key = lambda x: x[1])[0]
    
    print(f"Iter: {it}: Sharpe PPO={sharpe_ppo} "
          f"A2C={sharpe_a2c} DDPG={sharpe_ddpg} → Best = {best}")
    return best



def run_ensemble(df):
    df['datadate_dt'] = pd.to_datetime(df['datadate'].astype(str))
    dates = sorted(df['datadate'].unique())
    
    rebalance = 63
    train_start = 20090101
    val_start_idx = np.where(np.array(dates) == 20151001)[0][0]
    val_end_idx = val_start_idx + rebalance
    trade_end_idx = val_end_idx + rebalance

    insample_start_date = pd.to_datetime("2009-01-01")
    insample_end_date = pd.to_datetime("2015-10-01")

    it = 0
    while True:
        if trade_end_idx > len(dates):
            print(f"\n✔ 剩餘資料不足 {rebalance} 天，終止滾動視窗。")
            break
        print(f'\n----------Iteration {it}----------')
        print(f'Train period: {train_start} ~ {dates[val_start_idx - 1]}')
        print(f'Validation period: {dates[val_start_idx]} ~ {dates[val_end_idx - 1]}')
        print(f'Trade period: {dates[val_end_idx]} ~ {dates[trade_end_idx]}')

        val_start = dates[val_start_idx]
        val_end = dates[val_end_idx]
        trade_end = dates[trade_end_idx]

        train_df = df[(df['datadate'] >= train_start) & (df['datadate'] < val_start)]
        val_df = df[(df['datadate'] >= val_start) & (df['datadate'] < val_end)]
        trade_df = df[(df['datadate'] >= val_end) & (df['datadate'] < trade_end)]


        val_start_date = pd.to_datetime(str(val_start))
        if val_start_date >= insample_end_date + pd.DateOffset(years = 5):
            insample_start_date += pd.DateOffset(years=5)
            insample_end_date += pd.DateOffset(years=5)
            print(f" insample 滾動至: {insample_start_date.date()} ~ {insample_end_date.date()}")

        insample_turbulence = df[(df['datadate_dt'] >= insample_start_date) & (df['datadate_dt'] < insample_end_date)]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.9)
        robust_threshold_09 = 96
    
        lookback = 63
        hist_start_idx = max(val_start_idx - lookback, 0)
        hist_start = dates[hist_start_idx]
        historical_turbulence = df[(df['datadate'] >= hist_start) & (df['datadate'] < val_start)]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_mean = historical_turbulence.turbulence.mean()

        if historical_mean > insample_turbulence_threshold:
            turbulence_threshold = robust_threshold_09
            print(f"進入風控模式：historical_mean = {historical_mean:.2f} > threshold = {insample_turbulence_threshold:.2f}, robust_threshold = {robust_threshold_09}")
        else:
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            print(f"市場穩定：使用寬鬆 threshold = {turbulence_threshold:.2f}, {historical_mean}")
        
        # 訓練並驗證模型
        best_model = train_and_validate(train_df, val_df, it, turbulence_threshold)
        env_trade = StockEnvTrade(trade_df, best_model, iteration=it, initial=False, turbulence_threshold=turbulence_threshold)

        if best_model == "PPO":
            test_ppo(env_trade, "PPO", iteration=it)
        elif best_model == "A2C":
            test_a2c(env_trade, "A2C", iteration=it)
        elif best_model == "DDPG":
            test_ddpg(env_trade, "DDPG", iteration=it)

        # 更新索引
        val_start_idx += rebalance
        val_end_idx += rebalance
        trade_end_idx += rebalance
        it += 1





if __name__ == '__main__':
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/tmba_project/done_df_new.csv')   
    t0 = time.time()
    run_ensemble(df)
    print(f'\nTotal elapsed: {round((time.time() - t0) / 60, 2)} min')
    # df['datadate'] = pd.to_datetime(df['datadate'], errors = 'coerce', format = '%Y%m%d')
    # plt.plot(df['datadate'], df['turbulence'])
    # plt.show()
    
    
    




