import pandas as pd
import os
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None) 


def get_account_value(model_directories):
    df_account_value = pd.DataFrame()
    all_files = []
    
    for model_directory in model_directories:
        print(f"檢查資料夾: {model_directory}")
        files_in_directory = os.listdir(model_directory)
        print(f"資料夾內檔案: {files_in_directory}")
        for file_name in files_in_directory:
            if file_name.endswith(".csv") and 'account_value_trade' in file_name:
                match = re.search(r'account_value_trade_(\w+)_(\d+).csv$', file_name)
                if match:
                    model_name = match.group(1)
                    it_value = int(match.group(2))
                    print(f"找到檔案: {file_name}, 模型名稱: {model_name}, it 值: {it_value}")
                    all_files.append((model_directory, file_name, model_name, it_value))
    if not all_files:
        print("沒有找到符合的檔案")
    # 按照it值排序
    all_files.sort(key = lambda x: x[3])
    for model_directory, file_name, model_name, it_value in all_files:
        #讀取csv檔案，第二欄命名為account_value
        temp = pd.read_csv(os.path.join(model_directory, file_name), header = None, names = ['index', 'account_value'])
        temp['account_value'] = pd.to_numeric(temp['account_value'], errors = 'coerce')
        temp['iteration'] = it_value
        temp['model_name'] = model_name
        df_account_value = pd.concat([df_account_value, temp], ignore_index = True)
        
    return df_account_value

def sharpe_ratio(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    std = df['daily_return'].std()
    sharpe = (252 ** 0.5) * df['daily_return'].mean() / std
    return sharpe

def plot_account_value(df_strategy, df_djia):
    df_strategy['datadate'] = pd.to_datetime(df_strategy['datadate'], format='%Y%m%d')
    df_djia['Date'] = pd.to_datetime(df_djia['Date'], format='%Y-%m-%d')
    df_strategy['daily_return'] = df_strategy['account_value'].pct_change(1)
    df_strategy['cumulative_return'] = (1 + df_strategy['daily_return']).cumprod()
    df_djia = df_djia.rename(columns={
        'close': 'close_djia',
        'high': 'high_djia',
        'low': 'low_djia',
        'open': 'open_djia',
        'volume': 'volume_djia',
        'daily_return': 'daily_return_djia',
        'cumulative_return': 'cumulative_return_djia',
        'Date': 'datadate',
        'drawdown': 'drawdown_djia' 
    })
    merge = pd.merge(df_strategy, df_djia, on='datadate')
    plot_risk_metrics(merge)
    max_dd_idx_djia = merge['drawdown_djia'].idxmin()
    max_dd_date_djia = merge.loc[max_dd_idx_djia, 'datadate']
    max_dd_value_djia = merge.loc[max_dd_idx_djia, 'cumulative_return_djia']
    
    max_dd_idx_strat = merge['drawdown'].idxmin()
    max_dd_date_strat = merge.loc[max_dd_idx_strat, 'datadate']
    max_dd_value_strat = merge.loc[max_dd_idx_strat, 'cumulative_return']

    plt.figure(figsize=(12, 6))
    plt.plot(merge['datadate'], merge['cumulative_return'], label='Ensemble Strategy')
    plt.plot(merge['datadate'], merge['cumulative_return_djia'], label='DJIA Index')
    
    plt.scatter(max_dd_date_djia, max_dd_value_djia, color='red', zorder = 5)
    plt.annotate(f'Max DD\n{max_dd_date_djia.date()}',
                 xy = (max_dd_date_djia, max_dd_value_djia),
                 xytext=(max_dd_date_djia, max_dd_value_djia + 0.1),
                 arrowprops = dict(facecolor = 'red', arrowstyle = '->'),
                 ha = 'center', fontsize = 9)
    
    plt.scatter(max_dd_date_strat, max_dd_value_strat, color = 'darkred', zorder = 5)
    plt.annotate(f'Max DD\n{max_dd_date_strat.date()}',
                 xy = (max_dd_date_strat, max_dd_value_strat),
                 xytext = (max_dd_date_strat, max_dd_value_strat + 0.1),
                 arrowprops = dict(facecolor = 'red', arrowstyle = '->'),
                 ha='center', fontsize=9)

    plt.title('Ensemble Strategy vs DJIA (with Max Drawdown)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



def plot_risk_metrics(df, window=126): 
    df = df.copy()
    df = df.sort_values('datadate')

   
    df['volatility'] = df['daily_return'].rolling(window).std()
    df['volatility_djia'] = df['daily_return_djia'].rolling(window).std()

    df['sharpe'] = (df['daily_return'].rolling(window).mean()) / df['daily_return'].rolling(window).std()
    df['sharpe_djia'] = (df['daily_return_djia'].rolling(window).mean()) / df['daily_return_djia'].rolling(window).std()

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df['datadate'], df['volatility'], label='Strategy Volatility', color='orange')
    axs[0].plot(df['datadate'], df['volatility_djia'], label='DJIA Volatility', color='gray')
    axs[0].axhline(y=df['volatility'].mean(), color='blue', linestyle='--', label='Avg Strategy Vol')
    axs[0].set_title(f'Rolling Volatility ({window} days)')
    axs[0].set_ylabel('Volatility')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df['datadate'], df['sharpe'], label='Strategy Sharpe', color='darkorange')
    axs[1].plot(df['datadate'], df['sharpe_djia'], label='DJIA Sharpe', color='gray')
    axs[1].axhline(y=df['sharpe'].mean(), color='blue', linestyle='--', label='Avg Strategy Sharpe')
    axs[1].axhline(y=0, color='black', linewidth=1)
    axs[1].set_title(f'Rolling Sharpe Ratio ({window} days)')
    axs[1].set_ylabel('Sharpe Ratio')
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()





def calculate_cr_ar(df):
    initial_value = df['account_value'].iloc[0]
    final_value = df['account_value'].iloc[-1]
    n_days = len(df)
    cr = ((final_value / initial_value) - 1)
    ar = (((1 + cr) ** (252 / n_days)) - 1) 
    return cr, ar 

def calculate_max_drawdown(df):
    df['datadate'] = pd.to_datetime(df['datadate'], format = '%Y%m%d')
    df['daily_return'] = df['account_value'].pct_change(1)
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() #累積乘積
    df['peak'] = df['cumulative_return'].cummax() # 累積最大值
    df['drawdown'] = df['cumulative_return'] / df['peak'] - 1 # 回撤
    max_drawdoen = df['drawdown'].min() # 最大回撤
    return max_drawdoen, df

def annual_volatility(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    annual_volatility = df['daily_return'].std() * (252 ** 0.5)
    return annual_volatility

if __name__ == '__main__':
    
    model_directories = [
        '/Users/huyiming/Downloads/python練習/tmba_project/results/trade/A2C',
        '/Users/huyiming/Downloads/python練習/tmba_project/results/trade/PPO',
        '/Users/huyiming/Downloads/python練習/tmba_project/results/trade/DDPG'
    ]
    df_account_value = get_account_value(model_directories)
    df_account_value = df_account_value.dropna().reset_index(drop = True)
    print(df_account_value.shape)
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/tmba_project/done_df_new.csv')
    df_djia = pd.read_csv('/Users/huyiming/Downloads/python練習/tmba_project/dji_cr.csv')
    df_date = df[(df['datadate'] < 20241007) & (df['datadate'] >= 20151231)]
    datadate = df_date['datadate'].unique()
    
    print(datadate.shape)
    
    df_account_value['datadate'] = datadate
    df_account_value_paper  = df_account_value[df_account_value['datadate'] <= 20200508]
    # print(df_account_value.head(10))
    sharpe = sharpe_ratio(df_account_value_paper)
    print(f"Sharpe Ratio: {sharpe:.4f}")
    cr, ar = calculate_cr_ar(df_account_value_paper)
    print(f"累積報酬:{cr:.4%}, 年化報酬:{ar:.4%}")
    mdd, mdd_df = calculate_max_drawdown(df_account_value_paper)
    print(f"最大回撤:{mdd:.4%}")
    a_vol = annual_volatility(df_account_value_paper)
    print(f"年化波動率:{a_vol:.4%}")
    plot_account_value(mdd_df, df_djia)
    