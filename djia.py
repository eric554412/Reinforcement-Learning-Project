import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt



def calculate_cr_ar(df):
    initial = df['close'].iloc[0]
    final = df['close'].iloc[-1]
    n_days = len(df)
    cr = ((final / initial) - 1)
    ar = (((1 + cr) ** (252 / n_days)) - 1) 
    return cr, ar

def sharpe_ratio(df):
    df['daily_return'] = df['close'].pct_change(1)
    std = df['daily_return'].std()
    sharpe = (252 ** 0.5) * df['daily_return'].mean() / std
    return sharpe

def annual_volatility(df):
    df['daily_return'] = df['close'].pct_change(1)
    std = df['daily_return'].std()
    annual_volatility = std * (252 ** 0.5)
    return annual_volatility

def calculate_max_drawdown(df):
    df['daily_return'] = df['close'].pct_change(1)
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    df['peak'] = df['cumulative_return'].cummax()
    df['drawdown'] = df['cumulative_return'] / df['peak'] - 1
    max_drawdown = df['drawdown'].min()
    plt.plot(df['Date'], df['cumulative_return'])
    plt.title('Cumulative Return with Transaction Cost')
    plt.xlabel('Date')
    plt.ylabel('cumulative_return')
    plt.show()
    return max_drawdown, df

    

if __name__ == "__main__":
    # dji = yf.download("^DJI", start="2015-12-31", end="2024-10-05")
    # dji.to_csv("/Users/huyiming/Downloads/python練習/tmba_project/dji.csv")
    df = pd.read_csv("/Users/huyiming/Downloads/python練習/tmba_project/dji.csv", skiprows = 2)
    df.rename(columns = {"Unnamed: 1": "close", "Unnamed: 2": "high", "Unnamed: 3": "low", "Unnamed: 4": "open", "Unnamed: 5": "volume"}, inplace = True)
    df = df.sort_values(by = ['Date'])
    cr, ar = calculate_cr_ar(df)
    print(f"累積報酬:{cr:.4%}, 年化報酬:{ar:.4%}")
    sharpe = sharpe_ratio(df)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    annual_vol = annual_volatility(df)
    print(f"Annual Volatility: {annual_vol:.2%}")
    mdd, df_final = calculate_max_drawdown(df)
    print(f"最大回撤:{mdd:.4%}")
    # df_final.to_csv('/Users/huyiming/Downloads/python練習/tmba_project/dji_cr.csv')
