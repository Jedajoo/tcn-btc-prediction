from IPython.display import display
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from joblib import load
import utils as ut

ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2026-01-01'
time_window = 30 
# Menggunakan 30 timestep sebelumnya untuk memprediksi harga pada timestep berikutnya

df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
df = df.drop(columns=['Close'])

# Check if there are any zeros or negative values in Volume
zeros_count = (df['Volume'] == 0).sum()
negatives_count = (df['Volume'] < 0).sum()

print(f"Jumlah Volume bernilai 0: {zeros_count.values[0]}")
print(f"Jumlah Volume bernilai negatif: {negatives_count.values[0]}")

# Menghilangkan Volume yang <= 0
df_filtered = df[df['Volume'] > 0]
df_filtered = df_filtered.dropna()

print(f"Jumlah baris sebelum filter: {len(df)}")
print(f"Jumlah baris setelah filter: {len(df_filtered)}")
display(df_filtered.head())

df = df_filtered

# Check for missing values
missing_values = df_filtered.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Remove rows with missing values
df_cleaned = df_filtered.dropna()

df = pd.DataFrame(df_cleaned.xs(ticker, axis=1, level='Ticker'))

print(f"\nJumlah baris sebelum menghapus missing values: {len(df_filtered)}")
print(f"Jumlah baris setelah menghapus missing values: {len(df_cleaned)}")

# Meratakan kolom MultiIndex
# Mengambil level terakhir dari MultiIndex untuk mendapatkan nama kolom yang datar
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)

# Mengubah indeks 'Date' menjadi kolom biasa

# Menghitung SMA dengan jendela dinamis (expanding window) di awal
# min_periods=1 memastikan perhitungan dimulai dari data pertama yang tersedia
close_s = df['Adj Close'].squeeze()

df['SMA10'] = close_s.rolling(10).mean()
df['SMA25'] = close_s.rolling(25).mean()

ema_12 = close_s.ewm(span=12, min_periods=1).mean()
ema_26 = close_s.ewm(span=26, min_periods=1).mean()

# Menghitung MACD Line
df['MACD Line'] = ema_12 - ema_26
df['MACD Signal'] = df['MACD Line'].ewm(span=9, min_periods=1).mean()

# Menghitung RSI dengan jendela dinamis
delta = close_s.diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# Menggunakan Wilder's Smoothing (EWM) dengan min_periods=1
avg_gain = gain.ewm(alpha=1/14, min_periods=1).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=1).mean()

# Menghitung RSI
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Mengisi RSI yang bernilai 0 dan NaN dengan RSI positif terdekat
# 1. Ganti semua 0 dengan NaN agar dapat diisi oleh ffill/bfill
df['RSI'] = df['RSI'].replace(0, np.nan)

# 2. Gunakan ffill (forward fill) untuk mengisi NaN dengan nilai positif terdekat sebelumnya
# 3. Gunakan bfill (backward fill) untuk mengisi NaN di awal (jika ada) dengan nilai positif terdekat setelahnya
df['RSI'] = df['RSI'].replace([np.inf, -np.inf], np.nan)

print("RSI berhasil dihitung tanpa missing values, dengan nilai 0 diisi oleh RSI positif terdekat.")

std_dev = close_s.rolling(20).std(ddof=0)
m_band = close_s.rolling(20).mean()

# Menambahkan Bollinger Bands ke DataFrame
df['Upper BBand'] = m_band + 2 * std_dev
df['Lower BBand'] = m_band - 2 * std_dev

df['Band Width'] = df['Upper BBand'] - df['Lower BBand']
df['Band %'] = (close_s - df['Lower BBand']) / df['Band Width']
df[['Band Width', 'Band %']] = df[['Band Width', 'Band %']].replace(0, np.nan)
df[['Band Width', 'Band %']] = df[['Band Width', 'Band %']].replace([np.inf, -np.inf], np.nan)

print("Bollinger Bands berhasil ditambahkan ke DataFrame.")

df = df.dropna()

print("Jumlah missing values setelah perbaikan:")
print(df.isnull().sum())

# Memperbaiki height_ratios agar berjumlah 4 sesuai dengan jumlah subplot (ax1, ax2, ax3, ax4)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 2]})

# Plot Harga dan Moving Averages
ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='black', alpha=0.6)
ax1.plot(df.index, df['SMA10'], label='SMA 10', linestyle='--')
ax1.plot(df.index, df['SMA25'], label='SMA 25', linestyle='--')
ax1.set_title(f'Analisis Teknikal {ticker}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot MACD
ax2.plot(df.index, df['MACD Line'], label='MACD Line', color='blue')
ax2.plot(df.index, df['MACD Signal'], label='Signal Line', color='red')
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.set_title('MACD Indicator')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot RSI
ax3.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
ax3.axhline(70, color='red', linestyle='--', alpha=0.5) # Overbought
ax3.axhline(30, color='green', linestyle='--', alpha=0.5) # Oversold
ax3.set_ylim(0, 100)
ax3.set_title('RSI Indicator')
ax3.legend()
ax3.grid(True, alpha=0.3)

#Plot Bband
ax4.plot(df.index, df['Upper BBand'], label='Upper Band', color='blue')
ax4.plot(df.index, df['Lower BBand'], label='Lower Band', color='blue')
ax4.plot(df.index, df['Adj Close'], label='Close', color='green')
ax4.fill_between(df.index, df['Upper BBand'], df['Lower BBand'], color='blue', alpha=0.1)
ax4.set_title('Bollinger Bands')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

display(df.head())

# Define the full list of technical and price features
columns_to_scale = [
    'Adj Close', 'High', 'Low', 'Open', 'Volume',
    'SMA10', 'SMA25', 'MACD Line', 'MACD Signal',
    'RSI', 'Upper BBand', 'Lower BBand', 'Band Width', 'Band %'
]

# Save the scaler if needed for inverse transformations later

# Tentukan ukuran training set (70% dari total data)
train_size = int(len(df) * 0.7)

# Bagi data menjadi training dan testing set berdasarkan urutan waktu
train_raw = df.iloc[:train_size]
test_raw = df.iloc[train_size:]

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw[columns_to_scale])

test_scaled = scaler.transform(test_raw[columns_to_scale])

train_df = pd.DataFrame(train_scaled, columns=columns_to_scale, index=train_raw.index)

test_df = pd.DataFrame(test_scaled, columns=columns_to_scale, index=test_raw.index)

print(f"Ukuran Training Set: {len(train_df)} baris")
print(f"Ukuran Testing Set: {len(test_df)} baris")

print("\nHead Training Set:")
display(train_df.head())

print("\nHead Testing Set:")
display(test_df.head())

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i : (i + window)].values)
        # Target: Harga 'Adj Close' (kolom pertama) pada hari berikutnya
        y.append(data.iloc[i + window, 0])
    return np.array(X), np.array(y)


# Membuat sekuens untuk data testing
test_input = pd.concat([
    train_df.tail(time_window),
    test_df
])

X_train, y_train = create_sequences(train_df, time_window)
X_test, y_test = create_sequences(test_input, time_window)

def inverse_transform_target(scaled_val, scaler_obj, n_features):
    # Buat array dummy dengan jumlah kolom yang sama saat scaling (14 kolom)
    dummy = np.zeros((len(scaled_val), n_features))
    # Masukkan nilai yang ingin di-inverse ke kolom pertama (Adj Close)
    dummy[:, 0] = scaled_val.flatten()
    # Lakukan inverse transform
    inverse = scaler_obj.inverse_transform(dummy)
    # Ambil kembali kolom pertama
    return inverse[:, 0]

best_model = load_model("output/model/model_tcn_pso.keras")

predictions = best_model.predict(X_test)
pred_prices = ut.inverse_transform_target(predictions, scaler, 14)
actual_prices = ut.inverse_transform_target(y_test, scaler, 14) 

mae = mean_absolute_error(actual_prices, pred_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
mape = ut.mean_absolute_percentage_error(actual_prices, pred_prices)
r2 = r2_score(actual_prices, pred_prices)

print("\n--- Evaluasi ---")
print(f"MAE  : {mae:.5f}")
print(f"RMSE : {rmse:.5f}")
print(f"MAPE : {mape:.2f}%")
print(f"R2   : {r2:.5f}")

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label='Actual', color='black')
plt.plot(pred_prices, label='Predicted (PSO-TCN)', color='green')
plt.title(f'Prediksi Harga Saham {ticker}')
plt.legend()
plt.grid(True)

plt.savefig('output/plots/evaluationAAPL.png', dpi=300, bbox_inches='tight')
plt.show()