import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
from joblib import dump

os.makedirs('output/data/plots', exist_ok=True)
os.makedirs('output/scaler', exist_ok=True)

ticker = 'BTC-USD'
start_date = '2020-01-01'
end_date = '2026-09-01'
time_window = 120

df = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date, auto_adjust=False))

if isinstance(df.columns, pd.MultiIndex):
    if ticker in df.columns.levels[1]:
        df = df.xs(ticker, axis=1, level='Ticker')
    else:
        df.columns = df.columns.get_level_values(-1)

if 'Close' in df.columns and 'Adj Close' in df.columns:
    df = df.drop(columns=['Close'])
elif 'Close' in df.columns and 'Adj Close' not in df.columns:
    df['Adj Close'] = df['Close']
    df = df.drop(columns=['Close'])

df_filtered = df[df['Volume'] > 0]
print(f"Data count after filtering volume and missing values: {len(df_filtered)}")
df = df_filtered.copy()

close_s = df['Adj Close'].squeeze()
high_s = df['High'].squeeze()
low_s = df['Low'].squeeze()
open_s = df['Open'].squeeze()
vol_s = df['Volume'].squeeze()

df['Log_Return_1'] = np.log(close_s / (close_s.shift(1) + 1e-9)).fillna(0.0)
df['Log_Return_3'] = np.log(close_s / (close_s.shift(3) + 1e-9)).fillna(0.0)
df['Log_Return_5'] = np.log(close_s / (close_s.shift(5) + 1e-9)).fillna(0.0)
df['Log_Return_10'] = np.log(close_s / (close_s.shift(10) + 1e-9)).fillna(0.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.25)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['Log_Return_1'], label='Log Return 1', color='blue')
ax2.plot(df.index, df['Log_Return_3'], label='Log Return 3', color='red')
ax2.plot(df.index, df['Log_Return_5'], label='Log Return 5', color='green')
ax2.plot(df.index, df['Log_Return_10'], label='Log Return 10', color='yellow')
ax2.set_title('Log Return')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/log_return.png", dpi=300, bbox_inches='tight')
plt.close()

df['High_Low_Ratio'] = ((high_s - low_s) / (close_s + 1e-9)).fillna(0.0)
df['Close_Open_Ratio'] = ((close_s - open_s) / (open_s + 1e-9)).fillna(0.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.25)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['High_Low_Ratio'], label='High Low Ratio', color='blue')
ax2.plot(df.index, df['Close_Open_Ratio'], label='Close Open Ratio', color='red')
ax2.set_title('Price Ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/price_ratio_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

df['SMA10'] = close_s.rolling(10, min_periods=1).mean()
df['SMA25'] = close_s.rolling(25, min_periods=1).mean()
df['SMA50'] = close_s.rolling(50, min_periods=1).mean()

df['Dist_SMA10'] = ((close_s - df['SMA10']) / (df['SMA10'] + 1e-9)).fillna(0.0)
df['Dist_SMA25'] = ((close_s - df['SMA25']) / (df['SMA25'] + 1e-9)).fillna(0.0)
df['Dist_SMA50'] = ((close_s - df['SMA50']) / (df['SMA50'] + 1e-9)).fillna(0.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.25)
ax1.plot(df.index, df['SMA10'], label='SMA 10', color='blue')
ax1.plot(df.index, df['SMA25'], label='SMA 25', color='red')
ax1.plot(df.index, df['SMA50'], label='SMA 50', color='green')
ax1.set_title('Simple Moving Average')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['Dist_SMA10'], label='SMA 10', color='blue')
ax2.plot(df.index, df['Dist_SMA25'], label='SMA 25', color='red')
ax2.plot(df.index, df['Dist_SMA50'], label='SMA 50', color='green')
ax2.set_title('Distance From Close to SMA')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.savefig(f"output/data/plots/moving_averages_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

ema_12 = close_s.ewm(span=12, min_periods=1).mean()
ema_26 = close_s.ewm(span=26, min_periods=1).mean()
macd_line = ema_12 - ema_26
macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
macd_hist = macd_line - macd_signal
df['MACD_Line_Norm'] = (macd_line / (close_s + 1e-9)).fillna(0.0)
df['MACD_Signal_Norm'] = (macd_signal / (close_s + 1e-9)).fillna(0.0)
df['MACD_Hist_Norm'] = (macd_hist / (close_s + 1e-9)).fillna(0.0)


plt.figure(figsize=(24, 12))
plt.plot(df.index, df['MACD_Line_Norm'], label='MACD Line', color='blue')
plt.plot(df.index, df['MACD_Signal_Norm'], label='MACD Signal', color='red')
plt.axhline(0, color='black', linewidth=1, linestyle='-')
plt.title('MACD Indicator')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"output/data/plots/macd_indicator{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

# D. RSI (14) Normalized to [-1.0, 1.0]
delta = close_s.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1/14, min_periods=1).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=1).mean()
rs = avg_gain / (avg_loss + 1e-9)
rsi = 100.0 - (100.0 / (1.0 + rs))
rsi = rsi.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df['RSI_Norm'] = (rsi - 50.0) / 50.0

plt.figure(figsize=(24, 12))
plt.plot(df.index, df['RSI_Norm'], label='Normalized RSI', color='blue')
plt.title('RSI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"output/data/plots/rsi_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

# E. Bollinger Bands Normalized
std_dev = close_s.rolling(20, min_periods=1).std(ddof=0).fillna(0)
m_band = close_s.rolling(20, min_periods=1).mean()
upper_bband = m_band + 2 * std_dev
lower_bband = m_band - 2 * std_dev
band_width = upper_bband - lower_bband
df['Band_Pos'] = ((close_s - lower_bband) / (band_width + 1e-9)).clip(-1.0, 2.0).fillna(0.5)
df['Band_Width_Norm'] = (band_width / (close_s + 1e-9)).fillna(0.0)

plt.figure(figsize=(24, 12))
plt.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.25)
plt.plot(df.index, upper_bband, label='Upper BBand', color='blue')
plt.plot(df.index, lower_bband, label='Lower BBand', color='red')
plt.fill_between(df.index, upper_bband, lower_bband, color='purple', alpha=0.25)
plt.title('Bollinger Bands')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"output/data/plots/bollinger_band_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

# F. Garman-Klass Volatility
log_hl = np.log(np.maximum(high_s / (low_s + 1e-9), 1e-9))
log_co = np.log(np.maximum(close_s / (open_s + 1e-9), 1e-9))
gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
df['GK_Vol'] = np.sqrt(np.maximum(gk_var, 0.0))
df['GK_Vol_14'] = df['GK_Vol'].rolling(14, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.6)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['GK_Vol'], label='Garman-Klass 1', color='blue')
ax2.plot(df.index, df['GK_Vol_14'], label='Garman-Klass 14', color='red')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_title('Chaikin Money Flow')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/garman_klass_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

# G. Chaikin Money Flow (CMF 20)
hl_diff = (high_s - low_s).replace(0, np.nan)
mf_multiplier = ((close_s - low_s) - (high_s - close_s)) / (hl_diff + 1e-9)
mf_multiplier = mf_multiplier.fillna(0.0)
mf_volume = mf_multiplier * vol_s
df['CMF'] = mf_volume.rolling(20, min_periods=1).sum() / (vol_s.rolling(20, min_periods=1).sum() + 1e-9)
df['CMF'] = df['CMF'].replace([np.inf, -np.inf], np.nan).ffill().bfill()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.6)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['CMF'], label='CMF', color='green')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_title('Chaikin Money Flow')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/chaikin_money_flow_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

# H. Stochastic Oscillator Normalized to [-1.0, 1.0]
lowest_low_14 = low_s.rolling(14, min_periods=1).min()
highest_high_14 = high_s.rolling(14, min_periods=1).max()
stoch_range = (highest_high_14 - lowest_low_14).replace(0, np.nan)
stoch_k = ((close_s - lowest_low_14) / (stoch_range + 1e-9)) * 100.0
stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).ffill().bfill()
stoch_d = stoch_k.rolling(3, min_periods=1).mean().ffill().bfill()
df['Stoch_K_Norm'] = (stoch_k - 50.0) / 50.0
df['Stoch_D_Norm'] = (stoch_d - 50.0) / 50.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.6)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['Stoch_K_Norm'], label='Stoch K', color='blue')
ax2.plot(df.index, df['Stoch_D_Norm'], label='Stoch D', color='red')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_title('Stochastic Oscillator')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/stochastic_oscillator_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()


# I. Volume Relative Metrics
vol_sma20 = vol_s.rolling(20, min_periods=1).mean()
df['Volume_Pct_Change'] = (vol_s.pct_change()).clip(-2.0, 5.0).fillna(0.0)
df['Volume_SMA_Ratio'] = ((vol_s / (vol_sma20 + 1e-9)) - 1.0).clip(-2.0, 5.0).fillna(0.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.6)
ax1.set_title('Close Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(df.index, df['Volume_Pct_Change'], label='RVOL', color='blue')
ax2.plot(df.index, df['Volume_SMA_Ratio'], label='RVOL SMA 20', color='red', alpha=0.25)
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_title('RVOL')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig(f"output/data/plots/RVOL_{ticker}.png", dpi=300, bbox_inches='tight')
plt.close()

df['Next_Adj_Close'] = close_s.shift(-1)
df['Next_Log_Return'] = np.log(df['Next_Adj_Close'] / close_s)
df.dropna(subset=['Next_Adj_Close']).copy()

df = df.dropna()

print(f"\nDataset shape after indicator calculations: {df.shape}")

# Save full processed tabular dataset
df.to_csv(f"output/data/dataset_{ticker}.csv", index=True)

feature_cols = [
    'Log_Return_1', 'Log_Return_3', 'Log_Return_5', 'Log_Return_10',
    'High_Low_Ratio', 'Close_Open_Ratio',
    'Dist_SMA10', 'Dist_SMA25', 'Dist_SMA50',
    'MACD_Line_Norm', 'MACD_Signal_Norm', 'MACD_Hist_Norm',
    'RSI_Norm', 'Band_Pos', 'Band_Width_Norm',
    'GK_Vol', 'GK_Vol_14', 'CMF', 'Stoch_K_Norm', 'Stoch_D_Norm',
    'Volume_Pct_Change', 'Volume_SMA_Ratio',
]

def mrmr_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_features_to_select: int = 15,
    method: str = "MID",
    random_state: int = 42
):
    """
    Minimum Redundancy Maximum Relevance (mRMR) Feature Selection using scikit-learn.

    Parameters:
    -----------
    X : np.ndarray
        Scaled training feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target array of shape (n_samples,).
    feature_names : list of str
        List of candidate feature names.
    n_features_to_select : int
        Number of top features to select.
    method : str
        'MID' (Mutual Information Difference) or 'MIQ' (Mutual Information Quotient).
    random_state : int
        Random seed for sklearn mutual_info_regression.

    Returns:
    --------
    selected_indices : list of int
        Indices of selected features.
    selected_features : list of str
        Names of selected features in order of selection.
    selection_history : list of dict
        Step-by-step logs with relevance, redundancy, and mRMR scores.
    """
    n_samples, n_features = X.shape
    k = min(n_features_to_select, n_features)

    # 1. Relevance: I(f_i; Y) using scikit-learn mutual_info_classif for directional classification
    if len(np.unique(y)) <= 2:
        relevance = mutual_info_classif(X, y, random_state=random_state)
    else:
        relevance = mutual_info_regression(X, y, random_state=random_state)

    # 2. Redundancy: Pairwise Pearson correlation matrix among scaled features
    corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    selected_indices = []
    remaining_indices = list(range(n_features))
    selection_history = []

    # Step 1: Select feature with highest relevance to target
    first_idx = int(np.argmax(relevance))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    selection_history.append({
        'step': 1,
        'index': first_idx,
        'feature': feature_names[first_idx],
        'relevance': float(relevance[first_idx]),
        'redundancy': 0.0,
        'score': float(relevance[first_idx])
    })

    # Steps 2..k: Greedily select features balancing relevance and redundancy
    for step in range(2, k + 1):
        best_score = -np.inf
        best_idx = None
        best_rel = 0.0
        best_red = 0.0

        for cand_idx in remaining_indices:
            rel = relevance[cand_idx]
            red = np.mean([corr_matrix[cand_idx, sel_idx] for sel_idx in selected_indices])

            if method.upper() == "MIQ":
                score = rel / (red + 1e-9)
            else:  # MID
                score = rel - red

            if score > best_score:
                best_score = score
                best_idx = cand_idx
                best_rel = rel
                best_red = red

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        selection_history.append({
            'step': step,
            'index': best_idx,
            'feature': feature_names[best_idx],
            'relevance': float(best_rel),
            'redundancy': float(best_red),
            'score': float(best_score)
        })

    selected_features = [feature_names[i] for i in selected_indices]
    return selected_indices, selected_features, selection_history

split_percentage = 0.80
split_index = int(len(df) * split_percentage)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

scaler = RobustScaler(quantile_range=(25.0, 75.0))
train_scaled = scaler.fit_transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

n_features_to_select = 10
selected_indices, selected_features, selection_history = mrmr_feature_selection(
    X=train_scaled,
    y=train_df['Adj Close'].values,
    feature_names=feature_cols,
    n_features_to_select=n_features_to_select,
    method="MID",
    random_state=42
)

print("\nStep-by-Step mRMR Selection Summary:")
print(f"{'Step':<5} | {'Feature Name':<20} | {'Relevance (MI)':<15} | {'Redundancy':<12} | {'mRMR Score':<12}")
print("-" * 72)
for h in selection_history:
    print(f"{h['step']:<5} | {h['feature']:<20} | {h['relevance']:<15.4f} | {h['redundancy']:<12.4f} | {h['score']:<12.4f}")

train_scaled = train_scaled[:, selected_indices]
test_scaled = test_scaled[:, selected_indices]

dump(scaler, "output/scaler/feature_scaler.joblib")
dump(selected_features, "output/scaler/feature_columns.joblib")
print(f"\nSaved RobustScaler & selected feature names ({len(selected_features)} features) to output/scaler/")


# Plot and save mRMR Feature Ranking
plt.figure(figsize=(12, 6))
plot_features = [h['feature'] for h in selection_history]
plot_scores = [h['score'] for h in selection_history]
plt.barh(plot_features[::-1], plot_scores[::-1], color='steelblue')
plt.title(f'Top {len(selected_features)} Features Selected by mRMR (MID Scheme)')
plt.xlabel('mRMR Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/data/plots/mrmr_feature_selection.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved mRMR selection plot to output/data/plots/mrmr_feature_selection.png")

def create_sequences_from_arrays(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : (i + window)])
        # Target: Harga 'Adj Close' (kolom pertama) pada hari berikutnya
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences_from_arrays(
    train_scaled,
    time_window
)

X_test, y_test = create_sequences_from_arrays(
    test_scaled,
    time_window
)

print(f"\nDefault Sequence Shapes (window={time_window}):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

np.savez(
    "output/data/train_tabular.npz",
    features=train_scaled,
    prices=train_df['Adj Close'].values,
    next_prices=train_df['Next_Adj_Close'].values,
    returns=train_df['Next_Log_Return'].values,
    dates=train_df.index.astype(str).values
)
np.savez(
    "output/data/test_tabular.npz",
    features=test_scaled,
    prices=test_df['Adj Close'].values,
    next_prices=test_df['Next_Adj_Close'].values,
    returns=test_df['Next_Log_Return'].values,
    dates=test_df.index.astype(str).values
)

np.savez("output/data/train_data.npz", X=X_train, y=y_train)
np.savez("output/data/test_data.npz", X=X_test, y=y_test)
