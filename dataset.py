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
time_window = 60  # Default sequence window

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
df_filtered = df[df['Volume'] > 0].dropna()
print(f"Data count after filtering volume and missing values: {len(df_filtered)}")
df = df_filtered.copy()

# ==========================================
# 1. TECHNICAL INDICATORS & STATIONARY FEATURE ENGINEERING
close_s = df['Adj Close'].squeeze()
high_s = df['High'].squeeze()
low_s = df['Low'].squeeze()
open_s = df['Open'].squeeze()
vol_s = df['Volume'].squeeze()

# A. Stationary Price & Return Ratios
df['Log_Return_1'] = np.log(close_s / (close_s.shift(1) + 1e-9)).fillna(0.0)
df['Log_Return_3'] = np.log(close_s / (close_s.shift(3) + 1e-9)).fillna(0.0)
df['Log_Return_5'] = np.log(close_s / (close_s.shift(5) + 1e-9)).fillna(0.0)
df['Dist_SMA10'] = ((close_s - sma10) / (sma10 + 1e-9)).fillna(0.0)
df['Dist_SMA25'] = ((close_s - sma25) / (sma25 + 1e-9)).fillna(0.0)
df['MACD_Line_Norm'] = (macd_line / (close_s + 1e-9)).fillna(0.0)
df['MACD_Signal_Norm'] = (macd_signal / (close_s + 1e-9)).fillna(0.0)
df['MACD_Hist_Norm'] = (macd_hist / (close_s + 1e-9)).fillna(0.0)

# D. RSI (14) Normalized to [-1.0, 1.0]
rsi = 100.0 - (100.0 / (1.0 + rs))
rsi = rsi.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# E. Bollinger Bands Normalized
std_dev = close_s.rolling(20, min_periods=1).std(ddof=0).fillna(0)
m_band = close_s.rolling(20, min_periods=1).mean()
upper_bband = m_band + 2 * std_dev
lower_bband = m_band - 2 * std_dev
band_width = upper_bband - lower_bband
df['Band_Pos'] = ((close_s - lower_bband) / (band_width + 1e-9)).clip(-1.0, 2.0).fillna(0.5)
log_hl = np.log(np.maximum(high_s / (low_s + 1e-9), 1e-9))
log_co = np.log(np.maximum(close_s / (open_s + 1e-9), 1e-9))
df['GK_Vol'] = np.sqrt(np.maximum(gk_var, 0.0))
df['GK_Vol_14'] = df['GK_Vol'].rolling(14, min_periods=1).mean()

# G. Chaikin Money Flow (CMF 20)
hl_diff = (high_s - low_s).replace(0, np.nan)
mf_multiplier = ((close_s - low_s) - (high_s - close_s)) / (hl_diff + 1e-9)
mf_multiplier = mf_multiplier.fillna(0.0)
lowest_low_14 = low_s.rolling(14, min_periods=1).min()
highest_high_14 = high_s.rolling(14, min_periods=1).max()
stoch_k = ((close_s - lowest_low_14) / (stoch_range + 1e-9)) * 100.0
stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).ffill().bfill()
stoch_d = stoch_k.rolling(3, min_periods=1).mean().ffill().bfill()
df['Stoch_K_Norm'] = (stoch_k - 50.0) / 50.0
df['Stoch_D_Norm'] = (stoch_d - 50.0) / 50.0

# I. Volume Relative Metrics
df['Volume_Pct_Change'] = (vol_s.pct_change()).clip(-2.0, 5.0).fillna(0.0)
df['Volume_SMA_Ratio'] = ((vol_s / (vol_sma20 + 1e-9)) - 1.0).clip(-2.0, 5.0).fillna(0.0)

# J. Calendar / Cyclical Features
day_of_week = df.index.dayofweek
day_of_month = df.index.day
month = df.index.month

df['DayOfWeek_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
df['Month_Sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
df['Month_Cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)
df['DayOfMonth_Sin'] = np.sin(2 * np.pi * (day_of_month - 1) / 31.0)
df['DayOfMonth_Cos'] = np.cos(2 * np.pi * (day_of_month - 1) / 31.0)
df['Is_Weekend'] = (day_of_week >= 5).astype(float)
df['Next_Adj_Close'] = close_s.shift(-1)
df['Next_Return'] = (df['Next_Adj_Close'] - close_s) / (close_s + 1e-9)
print(f"\nDataset shape after indicator calculations: {df.shape}")
up_count = (df['Target_Direction'] == 1).sum()
down_count = (df['Target_Direction'] == 0).sum()
print(f"Target Distribution: UP={up_count} ({up_count/len(df)*100:.2f}%), DOWN={down_count} ({down_count/len(df)*100:.2f}%)")

# Save full processed tabular dataset
df.to_csv('output/data/dataset.csv', index=True)

# Feature columns list (All Stationary)
feature_cols = [
    'Log_Return_1', 'Log_Return_3', 'Log_Return_5', 'Log_Return_10',
    'High_Low_Ratio', 'Close_Open_Ratio',
    'Dist_SMA10', 'Dist_SMA25',
    'MACD_Line_Norm', 'MACD_Signal_Norm', 'MACD_Hist_Norm',
    'RSI_Norm', 'Band_Pos', 'Band_Width_Norm',
    'GK_Vol', 'GK_Vol_14', 'CMF', 'Stoch_K_Norm', 'Stoch_D_Norm',
    'Volume_Pct_Change', 'Volume_SMA_Ratio',
    'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos',
]

print(f"Number of feature columns: {len(feature_cols)}")

# ==========================================
# 3. TRAIN / TEST SPLIT & ROBUST SCALING
# ==========================================
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Training set: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
print(f"Testing set : {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")

# Fit RobustScaler ONLY on training feature set
train_features_scaled = scaler.fit_transform(train_df[feature_cols])
test_features_scaled = scaler.transform(test_df[feature_cols])

# Save scaler and feature names
dump(scaler, "output/scaler/feature_scaler.joblib")
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
# 4. DEFAULT SEQUENCE GENERATION (WINDOW=30)
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