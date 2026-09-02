import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from joblib import dump

# Ensure output directories exist
os.makedirs('output/data', exist_ok=True)
os.makedirs('output/scaler', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/model', exist_ok=True)
os.makedirs('checkpoints/seeds', exist_ok=True)

ticker = 'BTC-USD'
start_date = '2020-01-01'
end_date = '2026-01-01'
time_window = 90  # 3-Month Sequence Lookback Window

print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    if ticker in df.columns.levels[1]:
        df = df.xs(ticker, axis=1, level='Ticker')
    else:
        df.columns = df.columns.get_level_values(-1)

# Drop redundant 'Close' if 'Adj Close' is present
if 'Close' in df.columns and 'Adj Close' in df.columns:
    df = df.drop(columns=['Close'])
elif 'Close' in df.columns and 'Adj Close' not in df.columns:
    df['Adj Close'] = df['Close']
    df = df.drop(columns=['Close'])

# Filter invalid volume rows
df = df[df['Volume'] > 0].dropna()
print(f"Data count after filtering volume and missing values: {len(df)}")

# ==========================================
# 1. TECHNICAL INDICATORS & STATIONARY ALPHA FEATURE ENGINEERING
# ==========================================
close_s = df['Adj Close'].squeeze()
high_s = df['High'].squeeze()
low_s = df['Low'].squeeze()
open_s = df['Open'].squeeze()
vol_s = df['Volume'].squeeze()

# A. Stationary Multi-Horizon Log Returns & Price Action
df['Log_Return_1'] = np.log(close_s / (close_s.shift(1) + 1e-9)).fillna(0.0)
df['Log_Return_3'] = np.log(close_s / (close_s.shift(3) + 1e-9)).fillna(0.0)
df['Log_Return_5'] = np.log(close_s / (close_s.shift(5) + 1e-9)).fillna(0.0)
df['Log_Return_10'] = np.log(close_s / (close_s.shift(10) + 1e-9)).fillna(0.0)
df['High_Low_Ratio'] = ((high_s - low_s) / (close_s + 1e-9)).fillna(0.0)
df['Close_Open_Ratio'] = ((close_s - open_s) / (open_s + 1e-9)).fillna(0.0)

# B. Moving Average & EMA Ribbon Trend Alignment
sma10 = close_s.rolling(10, min_periods=1).mean()
sma25 = close_s.rolling(25, min_periods=1).mean()
df['Dist_SMA10'] = ((close_s - sma10) / (sma10 + 1e-9)).fillna(0.0)
df['Dist_SMA25'] = ((close_s - sma25) / (sma25 + 1e-9)).fillna(0.0)

ema_9 = close_s.ewm(span=9, min_periods=1).mean()
ema_21 = close_s.ewm(span=21, min_periods=1).mean()
ema_50 = close_s.ewm(span=50, min_periods=1).mean()
df['Dist_EMA9'] = ((close_s - ema_9) / (close_s + 1e-9)).fillna(0.0)
df['EMA9_EMA21_Ratio'] = ((ema_9 - ema_21) / (close_s + 1e-9)).fillna(0.0)
df['EMA21_EMA50_Ratio'] = ((ema_21 - ema_50) / (close_s + 1e-9)).fillna(0.0)

# C. Normalized MACD Indicators
ema_12 = close_s.ewm(span=12, min_periods=1).mean()
ema_26 = close_s.ewm(span=26, min_periods=1).mean()
macd_line = ema_12 - ema_26
macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
macd_hist = macd_line - macd_signal
df['MACD_Line_Norm'] = (macd_line / (close_s + 1e-9)).fillna(0.0)
df['MACD_Signal_Norm'] = (macd_signal / (close_s + 1e-9)).fillna(0.0)
df['MACD_Hist_Norm'] = (macd_hist / (close_s + 1e-9)).fillna(0.0)

# D. Average True Range (ATR 14) Normalized
tr1 = high_s - low_s
tr2 = (high_s - close_s.shift(1)).abs()
tr3 = (low_s - close_s.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr_14 = tr.rolling(14, min_periods=1).mean()
df['ATR_Norm'] = (atr_14 / (close_s + 1e-9)).fillna(0.0)

# E. ADX & Directional Movement (+DI, -DI)
up_move = high_s.diff()
down_move = -low_s.diff()
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
plus_dm_s = pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean()
minus_dm_s = pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean()
plus_di = 100.0 * (plus_dm_s / (atr_14 + 1e-9))
minus_di = 100.0 * (minus_dm_s / (atr_14 + 1e-9))
dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
adx = dx.rolling(14, min_periods=1).mean().fillna(0.0)
df['ADX_Norm'] = (adx / 100.0).clip(0.0, 1.0)
df['DI_Diff_Norm'] = ((plus_di - minus_di) / 100.0).clip(-1.0, 1.0)

# F. RSI (14) Normalized to [-1.0, 1.0]
delta = close_s.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1/14, min_periods=1).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=1).mean()
rs = avg_gain / (avg_loss + 1e-9)
rsi = 100.0 - (100.0 / (1.0 + rs))
rsi = rsi.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df['RSI_Norm'] = (rsi - 50.0) / 50.0

# G. Bollinger Bands Normalized Position & Width
std_dev = close_s.rolling(20, min_periods=1).std(ddof=0).fillna(0)
m_band = close_s.rolling(20, min_periods=1).mean()
upper_bband = m_band + 2 * std_dev
lower_bband = m_band - 2 * std_dev
band_width = upper_bband - lower_bband
df['Band_Pos'] = ((close_s - lower_bband) / (band_width + 1e-9)).clip(-1.0, 2.0).fillna(0.5)
df['Band_Width_Norm'] = (band_width / (close_s + 1e-9)).fillna(0.0)

# H. Garman-Klass Volatility
log_hl = np.log(np.maximum(high_s / (low_s + 1e-9), 1e-9))
log_co = np.log(np.maximum(close_s / (open_s + 1e-9), 1e-9))
gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
df['GK_Vol'] = np.sqrt(np.maximum(gk_var, 0.0))
df['GK_Vol_14'] = df['GK_Vol'].rolling(14, min_periods=1).mean()

# I. Chaikin Money Flow (CMF 20)
hl_diff = (high_s - low_s).replace(0, np.nan)
mf_multiplier = ((close_s - low_s) - (high_s - close_s)) / (hl_diff + 1e-9)
mf_multiplier = mf_multiplier.fillna(0.0)
mf_volume = mf_multiplier * vol_s
df['CMF'] = mf_volume.rolling(20, min_periods=1).sum() / (vol_s.rolling(20, min_periods=1).sum() + 1e-9)
df['CMF'] = df['CMF'].replace([np.inf, -np.inf], np.nan).ffill().bfill()

# J. Stochastic Oscillator Normalized to [-1.0, 1.0]
lowest_low_14 = low_s.rolling(14, min_periods=1).min()
highest_high_14 = high_s.rolling(14, min_periods=1).max()
stoch_range = (highest_high_14 - lowest_low_14).replace(0, np.nan)
stoch_k = ((close_s - lowest_low_14) / (stoch_range + 1e-9)) * 100.0
stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).ffill().bfill()
stoch_d = stoch_k.rolling(3, min_periods=1).mean().ffill().bfill()
df['Stoch_K_Norm'] = (stoch_k - 50.0) / 50.0
df['Stoch_D_Norm'] = (stoch_d - 50.0) / 50.0

# K. Volume Relative & On-Balance Volume (OBV)
vol_sma20 = vol_s.rolling(20, min_periods=1).mean()
df['Volume_Pct_Change'] = (vol_s.pct_change()).clip(-2.0, 5.0).fillna(0.0)
df['Volume_SMA_Ratio'] = ((vol_s / (vol_sma20 + 1e-9)) - 1.0).clip(-2.0, 5.0).fillna(0.0)

obv_direction = np.sign(close_s.diff()).fillna(0.0)
obv = (obv_direction * vol_s).cumsum()
obv_sma20 = obv.rolling(20, min_periods=1).mean()
obv_std20 = obv.rolling(20, min_periods=1).std(ddof=0).replace(0, np.nan).fillna(1.0)
df['OBV_Norm'] = ((obv - obv_sma20) / (obv_std20 + 1e-9)).clip(-3.0, 3.0).fillna(0.0)

# L. Calendar / Cyclical Features
day_of_week = df.index.dayofweek
day_of_month = df.index.day
month = df.index.month

df['DayOfWeek_Sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
df['DayOfWeek_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
df['Month_Sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
df['Month_Cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)
df['DayOfMonth_Sin'] = np.sin(2 * np.pi * (day_of_month - 1) / 31.0)
df['DayOfMonth_Cos'] = np.cos(2 * np.pi * (day_of_month - 1) / 31.0)
df['Is_Weekend'] = (day_of_week >= 5).astype(float)

# ==========================================
# 2. TARGET CREATION (DIRECTIONAL MOVEMENT)
# ==========================================
# Next day close price, return, and binary direction
df['Next_Adj_Close'] = close_s.shift(-1)
df['Next_Return'] = (df['Next_Adj_Close'] - close_s) / (close_s + 1e-9)
# Target: 1 for UP, 0 for DOWN (or flat)
df['Target_Direction'] = (df['Next_Adj_Close'] > close_s).astype(int)

# Drop last row since it doesn't have Next_Adj_Close
df = df.dropna()

print(f"\nDataset shape after indicator calculations: {df.shape}")
up_count = (df['Target_Direction'] == 1).sum()
down_count = (df['Target_Direction'] == 0).sum()
print(f"Target Distribution: UP={up_count} ({up_count/len(df)*100:.2f}%), DOWN={down_count} ({down_count/len(df)*100:.2f}%)")

# Save full processed tabular dataset
df.to_csv('output/data/dataset.csv', index=True)

# Complete List of 35 Stationary Feature Columns
feature_cols = [
    'Log_Return_1', 'Log_Return_3', 'Log_Return_5', 'Log_Return_10',
    'High_Low_Ratio', 'Close_Open_Ratio',
    'Dist_SMA10', 'Dist_SMA25', 'Dist_EMA9', 'EMA9_EMA21_Ratio', 'EMA21_EMA50_Ratio',
    'MACD_Line_Norm', 'MACD_Signal_Norm', 'MACD_Hist_Norm',
    'ATR_Norm', 'ADX_Norm', 'DI_Diff_Norm',
    'RSI_Norm', 'Band_Pos', 'Band_Width_Norm',
    'GK_Vol', 'GK_Vol_14', 'CMF', 'Stoch_K_Norm', 'Stoch_D_Norm',
    'Volume_Pct_Change', 'Volume_SMA_Ratio', 'OBV_Norm',
    'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos',
    'DayOfMonth_Sin', 'DayOfMonth_Cos', 'Is_Weekend'
]

print(f"Number of stationary feature columns: {len(feature_cols)}")

# ==========================================
# 3. TRAIN / TEST SPLIT & ROBUST SCALING
# ==========================================
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Training set: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
print(f"Testing set : {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")

# Fit RobustScaler ONLY on training feature set
scaler = RobustScaler(quantile_range=(25.0, 75.0))
train_features_scaled = scaler.fit_transform(train_df[feature_cols])
test_features_scaled = scaler.transform(test_df[feature_cols])

# Save scaler and feature names
dump(scaler, "output/scaler/feature_scaler.joblib")
dump(feature_cols, "output/scaler/feature_columns.joblib")
print("Saved RobustScaler to output/scaler/feature_scaler.joblib")

# Save tabular arrays (allows dynamic window evaluation in PSO)
np.savez(
    "output/data/train_tabular.npz",
    features=train_features_scaled,
    target=train_df['Target_Direction'].values,
    prices=train_df['Adj Close'].values,
    next_prices=train_df['Next_Adj_Close'].values,
    returns=train_df['Next_Return'].values,
    dates=train_df.index.strftime('%Y-%m-%d').to_numpy(dtype='U10')
)

np.savez(
    "output/data/test_tabular.npz",
    features=test_features_scaled,
    target=test_df['Target_Direction'].values,
    prices=test_df['Adj Close'].values,
    next_prices=test_df['Next_Adj_Close'].values,
    returns=test_df['Next_Return'].values,
    dates=test_df.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
    train_tail_features=train_features_scaled[-260:]  # Buffer for lookback sequence creation (supports up to window=256)
)

# ==========================================
# 4. DEFAULT SEQUENCE GENERATION (WINDOW=90)
# ==========================================
def create_sequences_from_arrays(features, target, window):
    X, y = [], []
    for i in range(len(features) - window + 1):
        X.append(features[i : (i + window)])
        y.append(target[i + window - 1])
    return np.array(X), np.array(y)

# For training sequences
X_train, y_train = create_sequences_from_arrays(
    train_features_scaled,
    train_df['Target_Direction'].values,
    time_window
)

# For testing sequences (concatenate train buffer to avoid losing initial test samples)
test_input_features = np.vstack([
    train_features_scaled[-time_window+1:],
    test_features_scaled
])
test_input_target = np.concatenate([
    train_df['Target_Direction'].values[-time_window+1:],
    test_df['Target_Direction'].values
])

X_test, y_test = create_sequences_from_arrays(
    test_input_features,
    test_input_target,
    time_window
)

print(f"\nDefault Sequence Shapes (window={time_window}):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

np.savez("output/data/train_data.npz", X=X_train, y=y_train)
np.savez("output/data/test_data.npz", X=X_test, y=y_test)
print("Data preprocessing completed successfully.")