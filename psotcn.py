import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [
                tf.config.LogicalDeviceConfiguration(
                    memory_limit=3072  # MB = 3 GB
                )
            ]
        )

        logical_gpus = tf.config.list_logical_devices('GPU')

        print(
            f"GPU tersedia: {len(gpus)} physical, "
            f"{len(logical_gpus)} logical"
        )

    except RuntimeError as e:
        print(e)

from joblib import load, dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import keras
import gc
import utils as ut
import random

ticker = "BTC-USD"

train = np.load("train_data.npz")
X_train = train["X"]
y_train = train["y"]

test = np.load("test_data.npz")
X_test = test["X"]
y_test = test["y"]

input_shape = (X_train.shape[1], X_train.shape[2])

n_splits = 5

tscv = TimeSeriesSplit(n_splits=n_splits)

folds = list(tscv.split(X_train))

def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)

def objective_function(params, folds, X_train, y_train):

    n_filters, dropout, log_lr = params

    # =========================
    # DECODE PARAMETER
    # =========================

    n_filters = int(
        16 * round(n_filters / 16)
    )

    n_filters = max(32, min(256, n_filters))

    k_size = 2

    dropout = float(dropout)

    learning_rate = 10 ** log_lr

    fold_losses = []

    # =========================
    # WALK-FORWARD FOLDS
    # =========================

    for fold_idx, (train_idx, val_idx) in enumerate(folds):

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # =========================
        # RESET RANDOMNESS
        # =========================

        seed = 42 + fold_idx
        set_seed(seed)

        # =========================
        # RESET MODEL
        # =========================

        tf.keras.backend.clear_session()

        model = ut.build_tcn_model(
            input_shape,
            n_filters,
            k_size,
            dropout,
            learning_rate,
            n_future=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        # =========================
        # TRAIN
        # =========================

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=128,
            shuffle=False,
            callbacks=[early_stop],
            verbose=1
        )

        # =========================
        # BEST FOLD LOSS
        # =========================

        best_val_loss = min(
            history.history['val_loss']
        )

        fold_losses.append(best_val_loss)

        keras.backend.clear_session()
        del model
        gc.collect()

    # =========================
    # FITNESS
    # =========================

    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)

    fitness = mean_loss + 0.2 * std_loss

    print(
        f"F={n_filters}, "
        f"K={k_size}, "
        f"Drop={dropout:.4f}, "
        f"LR={learning_rate:.6f}"
    )

    print(
        f"folds={np.round(fold_losses, 6)} | "
        f"mean={mean_loss:.6f} | "
        f"std={std_loss:.6f} | "
        f"fitness={fitness:.6f}"
    )

    return fitness


# =========================
# PARAMETER PSO
# =========================
n_particles = 30
n_iterations = 50
c1 = 1.8
c2 = 2.2

# 🔥 boundaries (log scale untuk LR)
boundaries = [
    (32, 256),      # n_filters
    (0.01, 0.3),    # dropout
    (-4, -3)      # log10(lr)
]

# velocity max (20% range)
v_max = [(b[1] - b[0]) * 0.2 for b in boundaries]

set_seed(42)

particles = []

for _ in range(n_particles):

    position = [
        random.uniform(*boundaries[0]),
        random.uniform(*boundaries[1]),
        random.uniform(*boundaries[2])
    ]

    velocity = [0.0] * 4

    particles.append({
        'position': position,
        'velocity': velocity,
        'pbest_position': position.copy(),
        'pbest_value': float('inf')
    })

gbest_position = None
gbest_value = float('inf')

print("Memulai PSO...")

# =========================
# LOOP PSO
# =========================
for iter_count in range(n_iterations):

    # inertia decay
    w = 0.9 - (0.5 * iter_count / n_iterations)

    print(f"\nIterasi {iter_count+1}/{n_iterations} | w={w:.3f}")

    # ===== Evaluasi =====
    for i, particle in enumerate(particles):
        print(f"Partikel {i+1}/{n_particles}")

        current_position = particle['position']
        current_value = objective_function(current_position, folds, X_train, y_train)

        # update pbest
        if current_value < particle['pbest_value']:
            particle['pbest_value'] = current_value
            particle['pbest_position'] = current_position.copy()

        # update gbest
        if current_value < gbest_value:
            gbest_value = current_value
            gbest_position = current_position.copy()

    # ===== Update velocity & position =====
    for particle in particles:
        for j in range(len(particle['position'])):

            r1 = random.random()
            r2 = random.random()

            # velocity update
            cognitive = c1 * r1 * (particle['pbest_position'][j] - particle['position'][j])
            social = c2 * r2 * (gbest_position[j] - particle['position'][j])

            v = (w * particle['velocity'][j]) + cognitive + social

            # 🔥 velocity clamping
            v = max(-v_max[j], min(v_max[j], v))
            particle['velocity'][j] = v

            # update position
            new_pos = particle['position'][j] + v

            # ===== boundary handling =====
            if j == 0:  # n_filters
                new_pos = int(16 * round(new_pos / 16))
                new_pos = max(boundaries[0][0], min(boundaries[0][1], new_pos))

            elif j == 1:
                KERNEL_OPTIONS = [2, 3, 5]

                new_pos = min(
                    KERNEL_OPTIONS,
                    key=lambda k: abs(k - new_pos)
                )

            elif j == 2:  # dropout
                new_pos = max(boundaries[2][0], min(boundaries[2][1], new_pos))

            elif j == 3:  # log_lr
                new_pos = max(boundaries[3][0], min(boundaries[3][1], new_pos))

            particle['position'][j] = new_pos


print("\nPSO Selesai!")
print(f"Best Loss: {gbest_value:.6f}")

# =========================
# FINAL MODEL
# =========================
best_n_filters = int(gbest_position[0])
best_k_size = int(gbest_position[1])
best_dropout = float(gbest_position[2])
best_lr = 10 ** gbest_position[3]

print("\nBest Parameter:")
print(best_n_filters, best_k_size, best_dropout, best_lr)

print("\nTraining final model...")

seed_losses = []
seed_models = []
seed_histories = []

for seed in range(42, 47):

    print(f"\nTraining seed {seed}")

    set_seed(seed)

    tf.keras.backend.clear_session()

    model = ut.build_tcn_model(
        input_shape,
        best_n_filters,
        best_k_size,
        best_dropout,
        best_lr,
        n_future=1
    )

    early_stopping = ut.get_early_stopping()

    history = model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    best_val_loss = min(
        history.history['val_loss']
    )

    seed_losses.append(best_val_loss)
    seed_models.append(model)
    seed_histories.append(history)

    print(
        f"Seed {seed} → "
        f"best val loss = {best_val_loss:.6f}"
    )

mean_loss = np.mean(seed_losses)
std_loss = np.std(seed_losses)

fitness = mean_loss + 0.2 * std_loss

print("\nMultiple Seed Result")
print(f"Losses : {seed_losses}")
print(f"Mean   : {mean_loss:.6f}")
print(f"Std    : {std_loss:.6f}")
print(f"Fitness: {fitness:.6f}")

best_seed_idx = np.argmin(seed_losses)

best_model = seed_models[best_seed_idx]
best_history = seed_histories[best_seed_idx]

best_model.save("model_tcn_pso.keras")

if history is None:
    raise RuntimeError("Training did not produce a history.")

else:
    # =========================
    # PLOT LOSS
    # =========================
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('TCN Training Loss (PSO Tuned)')
    plt.legend()
    plt.show()

    # =========================
    # EVALUASI
    # =========================
    scaler = load('minmax_scaler.joblib')

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

    # =========================
    # VISUALISASI
    # =========================

    plt.figure(figsize=(14, 6))
    plt.plot(actual_prices, label='Actual', color='black')
    plt.plot(pred_prices, label='Predicted (PSO-TCN)', color='green')
    plt.title(f'Prediksi Harga Saham {ticker}')
    plt.legend()
    plt.grid(True)

    plt.savefig('evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    dump(predictions, 'model_PSO_TCN3.joblib')