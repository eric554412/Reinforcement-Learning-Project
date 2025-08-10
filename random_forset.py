import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/huyiming/Downloads/python練習/tmba_project/done_df_new_feature.csv")  # 包含 30 檔股票，欄位有 tic、adjcp、rsi、macd 等

df = df[df['datadate'] < 20161001]

df['return_5d'] = df.groupby('tic')['adjcp'].pct_change(1)
df['target'] = (df['return_5d'] > 0).astype(int) 

features = ['macd', 'rsi', 'cci', 'vr', 'adx', 'atr', 'vol_5d', 'upper_band', 'middle_band', 'lower_band']
X = df[features].fillna(0)
y = df['target']

mask = ~y.isna()
X = X[mask]
y = y[mask]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)


importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

feature_importance.plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

top_k = 5
selected_features = feature_importance.head(top_k).index.tolist()
print("最重要的前 {} 個特徵:".format(top_k))
print(selected_features)
