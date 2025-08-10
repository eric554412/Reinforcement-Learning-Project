import pandas as pd
import numpy as np
import talib

pd.set_option("display.max_rows", None)


def calculate_vr(close, volume, n = 10):
    up_vol = []
    down_vol = []
    flat_vol = []
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            up_vol.append(volume[i])
            down_vol.append(0)
            flat_vol.append(0)
        elif close[i] < close[i - 1]:
            up_vol.append(0)
            down_vol.append(volume[i])
            flat_vol.append(0)
        else:
            up_vol.append(0)
            down_vol.append(0)
            flat_vol.append(volume[i])
    up_vol.insert(0, 0)
    down_vol.insert(0, 0)
    flat_vol.insert(0, 0)
    av = pd.Series(up_vol).rolling(n).sum()
    cv = pd.Series(down_vol).rolling(n).sum()
    bv = pd.Series(flat_vol).rolling(n).sum()
    denominator = cv + bv / 2
    denominator = denominator.replace(0, np.nan)
    vr = (av + bv / 2) / denominator * 100
    return vr

def apply_indicators(x):
    x = x.copy()
    x['atr'] = talib.ATR(x['high'], x['low'], x['adjcp'], timeperiod=14)
    x['vol_5d'] = talib.SMA(x['volume'], timeperiod=5)
    upper, middle, lower = talib.BBANDS(x['adjcp'], timeperiod=20, nbdevup=2, nbdevdn=2)
    x['upper_band'] = upper
    x['middle_band'] = middle
    x['lower_band'] = lower
    x = x.sort_values(by = 'datadate').reset_index(drop = True)
    x['vr'] = calculate_vr(x['adjcp'].values, x['volume'].values, n = 10)
    first_valid = x['vr'].first_valid_index()
    if first_valid is not None:
        x.loc[first_valid:, 'vr'] = x.loc[first_valid: ,'vr'].fillna(method = 'ffill').fillna(method = 'bfill')
    return x

if __name__ == '__main__':
    df = pd.read_csv('/Users/huyiming/Downloads/python練習/processed_data_with_turbulence.csv')
    df_before = df.copy()
    df_before = df_before.sort_values(by = ['tic', 'datadate']).reset_index(drop = True)
    df_tec = df_before.groupby('tic').apply(apply_indicators).reset_index(drop = True)
    df_tec = df_tec.sort_values(['datadate', 'tic'])
    df_tec = df_tec.reset_index(drop = True)
    df_tec['vr'] = df_tec.groupby('tic')['vr'].transform(lambda x: x.fillna(method = 'ffill').fillna(method = 'bfill'))
    df_tec['vr'] = df_tec['vr'].fillna(df_tec['vr'].median())
    df_tec.to_csv('/Users/huyiming/Downloads/python練習/tmba_project/done_df_new_feature.csv', index=False)

                



