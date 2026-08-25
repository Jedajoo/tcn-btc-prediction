import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN
from IPython.display import display

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i : (i + window)].values)
        # Target: Harga 'Adj Close' (kolom pertama) pada hari berikutnya
        y.append(data.iloc[i + window, 0])
    return np.array(X), np.array(y)

def create_train_test(data):
    # Tentukan ukuran training set (70% dari total data)
    train_size = int(len(data) * 0.7)

    # Bagi data menjadi training dan testing set berdasarkan urutan waktu
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]

    print(f"Ukuran Training Set: {len(train_df)} baris")
    print(f"Ukuran Testing Set: {len(test_df)} baris")

    print("\nHead Training Set:")
    display(train_df.head())

    print("\nHead Testing Set:")
    display(test_df.head())

    train_df.to_csv('train.csv', index=True)
    test_df.to_csv('test.csv', index=True)

    # Membuat sekuens untuk data training
    time_window = 30
    X_train, y_train = create_sequences(train_df, time_window)

    # Membuat sekuens untuk data testing
    X_test, y_test = create_sequences(test_df, time_window)

    print(f"Bentuk X_train: {X_train.shape} (Samples, Time Steps, Features)")
    print(f"Bentuk y_train: {y_train.shape}")
    print(f"Bentuk X_test: {X_test.shape}")
    print(f"Bentuk y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def inverse_transform_target(scaled_val, scaler_obj, n_features):
    """
    Mengembalikan nilai yang sudah di-scale (MinMaxScaler) ke harga asli.
    Asumsi: Target ('Adj Close') berada pada indeks kolom 0.
    """
    # Buat array dummy dengan jumlah kolom yang sama saat scaling (14 kolom)
    dummy = np.zeros((len(scaled_val), n_features))
    # Masukkan nilai yang ingin di-inverse ke kolom pertama (Adj Close)
    dummy[:, 0] = scaled_val.flatten()
    # Lakukan inverse transform
    inverse = scaler_obj.inverse_transform(dummy)
    # Ambil kembali kolom pertama
    return inverse[:, 0]

# 1. Membangun Arsitektur Model TCN
# TCN() layer menangani konvolusi temporal, dilasi, dan residual connections
def build_tcn_model(input_shape, n_filters, k_size, dropout, learning_rate, n_future=1):

    inputs = layers.Input(shape=input_shape)
    x = TCN(
        nb_filters=int(n_filters),
        kernel_size=int(k_size),
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=dropout,
        return_sequences=False
    )(inputs)

    # =========================
    # IMPROVED DENSE BLOCK
    # =========================
    
    # Dense 1 (lebih besar untuk menangkap kompleksitas)
    x = layers.Dense(128)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)

    # Dense 2
    x = layers.Dense(64)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(32)(x)
    x = layers.Activation('relu')(x)

    # =========================
    # OUTPUT LAYER
    # =========================
    outputs = layers.Dense(n_future, activation='linear')(x)

    # =========================
    # MODEL
    # =========================
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber()
    )
    return model

# Define Early Stopping callback
def get_early_stopping(patience=100):
    return EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=patience,         # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100
