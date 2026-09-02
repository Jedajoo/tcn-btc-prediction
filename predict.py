import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import utils as ut

ticker = 'BTC-USD'
start_date = '2023-01-01'
end_date = '2026-01-01'

print(f"Downloading evaluation data for {ticker} from {start_date} to {end_date}...")
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

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

df = df[df['Volume'] > 0].dropna()

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

# Next day target
df['Next_Adj_Close'] = close_s.shift(-1)
df['Next_Return'] = (df['Next_Adj_Close'] - close_s) / (close_s + 1e-9)
df['Target_Direction'] = (df['Next_Adj_Close'] > close_s).astype(int)

df = df.dropna()

# ==========================================
# 2. SCALING AND SEQUENCE CREATION
# ==========================================
scaler = load("output/scaler/feature_scaler.joblib")
feature_cols = load("output/scaler/feature_columns.joblib")

# Load best hyperparameters to get time_window and calibrated threshold
optimal_thresh = 0.50
if os.path.exists("output/model/best_hyperparameters.joblib"):
    best_hp = load("output/model/best_hyperparameters.joblib")
    time_window = best_hp.get("time_window", 90)
    optimal_thresh = float(best_hp.get("optimal_threshold", 0.50))
else:
    time_window = 90

scaled_features = scaler.transform(df[feature_cols])

X_eval, y_eval = ut.create_sequences(scaled_features, df['Target_Direction'].values, time_window)
eval_dates = df.index[time_window-1:]
eval_prices = df['Adj Close'].values[time_window-1:]
eval_next_prices = df['Next_Adj_Close'].values[time_window-1:]
eval_returns = df['Next_Return'].values[time_window-1:]

# ==========================================
# 3. ENSEMBLE PREDICTION
# ==========================================
SEED_CHECKPOINT_DIR = "checkpoints/seeds"
seed_list = [42, 43, 44, 45, 46]
all_prob_preds = []

for seed in seed_list:
    seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
    if os.path.exists(seed_model_path):
        model = tf.keras.models.load_model(seed_model_path, compile=False)
        prob = model.predict(X_eval, verbose=0).ravel()
        all_prob_preds.append(prob)
        del model
        tf.keras.backend.clear_session()

if len(all_prob_preds) == 0:
    print("Warning: No seed models found. Falling back to single model output/model/model_tcn_pso.keras")
    model = tf.keras.models.load_model("output/model/model_tcn_pso.keras", compile=False)
    all_prob_preds = [model.predict(X_eval, verbose=0).ravel()]
    del model
    tf.keras.backend.clear_session()

print(f"Running inference with {len(all_prob_preds)} ensemble prediction(s)...")
ensemble_probs = np.mean(all_prob_preds, axis=0)
pred_directions = (ensemble_probs >= optimal_thresh).astype(int)
confidences = np.abs(ensemble_probs - optimal_thresh) * 2.0 * 100.0

# ==========================================
# 4. RESULTS SUMMARY & EVALUATION
# ==========================================
results_df = pd.DataFrame({
    'Date': eval_dates.strftime('%Y-%m-%d'),
    'Current_Price': eval_prices,
    'Next_Price': eval_next_prices,
    'Return_%': eval_returns * 100.0,
    'P(Up)': np.round(ensemble_probs, 4),
    'Prediction': np.where(pred_directions == 1, 'UP', 'DOWN'),
    'Confidence_%': np.round(confidences, 2),
    'Actual': np.where(y_eval == 1, 'UP', 'DOWN'),
    'Correct': np.where(pred_directions == y_eval, '✓', '✗')
})

print(f"\n--- Recent Directional Predictions (Threshold = {optimal_thresh:.3f}) ---")
print(results_df.tail(15).to_string(index=False))

# Evaluation Metrics
acc = accuracy_score(y_eval, pred_directions)
auc_val = roc_auc_score(y_eval, ensemble_probs)
prec = precision_score(y_eval, pred_directions, zero_division=0)
rec = recall_score(y_eval, pred_directions, zero_division=0)
f1 = f1_score(y_eval, pred_directions, zero_division=0)

print("\n--- Model Evaluation Summary ---")
print(f"Decision Thresh : {optimal_thresh:.4f}")
print(f"Accuracy        : {acc * 100:.2f}%")
print(f"ROC-AUC         : {auc_val:.4f}")
print(f"Precision       : {prec * 100:.2f}%")
print(f"Recall          : {rec * 100:.2f}%")
print(f"F1-Score        : {f1:.4f}")

# Latest Prediction Forecast
latest_row = results_df.iloc[-1]
print(f"\n==========================================")
print(f"LATEST FORECAST FOR NEXT TRADING DAY:")
print(f"Asset       : {ticker}")
print(f"Last Price  : ${latest_row['Current_Price']:,.2f}")
print(f"Predicted   : {latest_row['Prediction']}")
print(f"Probability : P(Up) = {latest_row['P(Up)'] * 100:.2f}%")
print(f"Confidence  : {latest_row['Confidence_%']:.2f}%")
print(f"==========================================")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(14, 8))

# Subplot 1: Price and Predicted Direction Signals
ax1 = plt.subplot(2, 1, 1)
ax1.plot(eval_dates, eval_prices, label=f'{ticker} Price', color='black', alpha=0.7)
up_idx = np.where(pred_directions == 1)[0]
down_idx = np.where(pred_directions == 0)[0]
ax1.scatter(eval_dates[up_idx], eval_prices[up_idx], color='green', marker='^', s=30, label=f'Predicted UP (P >= {optimal_thresh:.2f})', alpha=0.6)
ax1.scatter(eval_dates[down_idx], eval_prices[down_idx], color='red', marker='v', s=30, label=f'Predicted DOWN (P < {optimal_thresh:.2f})', alpha=0.6)
ax1.set_title(f'{ticker} Directional Predictions (PSO-TCN Ensemble)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Confidence and Probability
ax2 = plt.subplot(2, 1, 2)
ax2.plot(eval_dates, ensemble_probs * 100.0, label='Probability of UP (%)', color='teal', lw=1.5)
ax2.axhline(optimal_thresh * 100.0, color='red', linestyle='--', label=f'Threshold ({optimal_thresh*100:.1f}%)')
ax2.fill_between(eval_dates, optimal_thresh * 100.0, ensemble_probs * 100.0, where=(ensemble_probs >= optimal_thresh), color='green', alpha=0.15)
ax2.fill_between(eval_dates, optimal_thresh * 100.0, ensemble_probs * 100.0, where=(ensemble_probs < optimal_thresh), color='red', alpha=0.15)
ax2.set_title('Prediction Probability P(UP) & Calibrated Decision Boundary')
ax2.set_xlabel('Date')
ax2.set_ylabel('Probability (%)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plots/predict_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()