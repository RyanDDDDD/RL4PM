import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 下载SP500和VIX指数数据（2000-2020）
sp500 = yf.download('^GSPC', start='2000-01-01', end='2020-12-31')
vix = yf.download('^VIX', start='2000-01-01', end='2020-12-31')

# 数据预处理和对齐
sp500['log_return'] = np.log(sp500['Close']).diff()

# 设置滚动窗口大小
window_size = 250

# 基础滚动特征
sp500['roll_mean'] = sp500['log_return'].rolling(window=window_size).mean()
sp500['roll_std'] = sp500['log_return'].rolling(window=window_size).std()

# 新增特征1：滚动动量 (Momentum)
sp500['momentum'] = sp500['Close'].pct_change(periods=window_size)

# 新增特征2：真实波动幅度（ATR）
sp500['TR'] = np.maximum.reduce([
    sp500['High'] - sp500['Low'],
    abs(sp500['High'] - sp500['Close'].shift(1)),
    abs(sp500['Low'] - sp500['Close'].shift(1))
])
sp500['ATR'] = sp500['TR'].rolling(window=window_size).mean()

# 新增特征3：VIX指数（市场情绪）
sp500['VIX'] = vix['Close']

# 删除NaN，特征对齐
sp500.dropna(inplace=True)

# 特征矩阵构建
features = ['roll_mean', 'roll_std', 'momentum', 'ATR', 'VIX']
X = sp500[features].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# HMM模型训练（4个状态）
hmm_model = GaussianHMM(n_components=4, covariance_type='full', n_iter=500, random_state=42)
hmm_model.fit(X_scaled)

# 预测市场状态
states = hmm_model.predict(X_scaled)
sp500['HMM_state'] = states

# 可视化市场状态
plt.figure(figsize=(16, 7))
for i in range(hmm_model.n_components):
    idx = sp500['HMM_state'] == i
    plt.plot(sp500.index[idx], sp500['Close'][idx], '.', label=f'State {i}')

plt.title('HMM Market States (Enhanced Features)')
plt.xlabel('Date')
plt.ylabel('S&P 500 Close Price')
plt.legend()
plt.show()
