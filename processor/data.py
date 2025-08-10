import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from tqdm import tqdm

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(file_name)
    return _data

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical indicators
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = []
    rsi = []
    cci = []
    dx = []

    for i in range(len(unique_ticker)):
        # macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        macd.append(temp_macd)
        
        # rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        rsi.append(temp_rsi)
        
        # cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        cci.append(temp_cci)
        
        # adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        dx.append(temp_dx)

    # Concatenate all lists into one dataframe
    df['macd'] = pd.concat(macd, ignore_index=True)
    df['rsi'] = pd.concat(rsi, ignore_index=True)
    df['cci'] = pd.concat(cci, ignore_index=True)
    df['adx'] = pd.concat(dx, ignore_index=True)

    return df

def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    
    # start after a year
    start = 252
    turbulence_index = [0] * start
    
    count = 0
    for i in tqdm(range(start, len(unique_date))):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]
        cov_temp = hist_price.cov()
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # Avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        
        turbulence_index.append(turbulence_temp)
    
    turbulence_index = pd.DataFrame({'datadate': df_price_pivot.index, 'turbulence': turbulence_index})
    return turbulence_index

def add_turbulence(df):
    """
    add turbulence index from a precalculated dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)
    return df

def preprocess_data(file_name: str):
    """data preprocessing pipeline"""
    
    df = load_dataset(file_name=file_name)
    
    # get data after 2009
    df = df[df.datadate >= 20090000]
    dow_30 = [
        "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GS", "HD",
        "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
        "PFE", "PG", "RTX", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "XOM"
    ]
    df = df[df['tic'].isin(dow_30)]
    
    # calculate adjusted price
    df_preprocess = calcualte_price(df)
    
    # add technical indicators using stockstats
    df_final = add_technical_indicator(df_preprocess)
    
    # add turbulence index
    df_final = add_turbulence(df_final)
    
    # fill the missing values at the beginning
    df_final.fillna(method='bfill', inplace=True)
    
    df_final.to_csv("/Users/huyiming/Downloads/python練習/processed_data_with_turbulence.csv")
    return df_final

if __name__ == "__main__":
    file_name = "/Users/huyiming/Downloads/python練習/tmba_project/21266592.csv" 
    df_processed = preprocess_data(file_name)
    print(df_processed.tail()) 




    