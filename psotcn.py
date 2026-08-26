import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle

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

    n_filters = int(
        16 * round(n_filters / 16)
    )

    n_filters = max(32, min(256, n_filters))

    k_size = 2

    dropout = float(dropout)

    learning_rate = 10 ** log_lr

    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"Folds: {fold_idx}")

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        seed = 42 + fold_idx
        set_seed(seed)

        tf.keras.backend.clear_session()

        print(f"  filters={n_filters}, kernel={k_size}, dropout={dropout:.4f}, lr={learning_rate:.6f}")

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
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            shuffle=False,
            callbacks=[early_stop],
            verbose=1
        )

        best_val_loss = min(
            history.history['val_loss']
        )

        fold_losses.append(best_val_loss)

        keras.backend.clear_session()
        del model
        gc.collect()

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

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PSO_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "pso_state.pkl"
)

def save_pso_checkpoint(iter_count, particle_idx, particles, gbest_position, gbest_value):
    state = {
        "iter_count": iter_count,
        "particle_idx": particle_idx,

        "particles": particles,

        "gbest_position": gbest_position,
        "gbest_value": gbest_value,

        # Random state
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }

    temp_file = PSO_CHECKPOINT + ".tmp"

    with open(temp_file, "wb") as f:
        pickle.dump(state, f)

    # Atomic replace
    os.replace(temp_file, PSO_CHECKPOINT)

    print(
        f"Checkpoint tersimpan "
        f"(iterasi={iter_count + 1}, particle={particle_idx + 1})"
    )


def load_pso_checkpoint():
    if not os.path.exists(PSO_CHECKPOINT):
        return None

    with open(PSO_CHECKPOINT, "rb") as f:
        state = pickle.load(f)

    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])

    print(
        f"Checkpoint ditemukan: "
        f"iterasi={state['iter_count'] + 1}, "
        f"particle={state['particle_idx'] + 1}"
    )

    return state

n_particles = 10
n_iterations = 20
c1 = 1.8
c2 = 2.2

boundaries = [
    (32, 256),      # n_filters
    (0.01, 0.3),    # dropout
    (-4, -3)      # log10(lr)
]

v_max = [(b[1] - b[0]) * 0.2 for b in boundaries]

set_seed(42)

checkpoint = load_pso_checkpoint()

if checkpoint is None:
    particles = []

    for _ in range(n_particles):

        position = [
            random.uniform(*boundaries[0]),
            random.uniform(*boundaries[1]),
            random.uniform(*boundaries[2])
        ]

        velocity = [0.0] * 3

        particles.append({
            'position': position,
            'velocity': velocity,
            'pbest_position': position.copy(),
            'pbest_value': float('inf')
        })

    gbest_position = None
    gbest_value = float('inf')

    start_iter = 0
    start_particle = 0

else:

    particles = checkpoint["particles"]

    gbest_position = checkpoint["gbest_position"]
    gbest_value = checkpoint["gbest_value"]

    start_iter = checkpoint["iter_count"]
    start_particle = checkpoint["particle_idx"] + 1

print("Memulai PSO...")

for iter_count in range(start_iter, n_iterations):

    # inertia decay
    w = 0.9 - (0.5 * iter_count / n_iterations)

    print(f"\nIterasi {iter_count+1}/{n_iterations} | w={w:.3f}")

    if iter_count == start_iter:
        particle_start = start_particle
    else:
        particle_start = 0

    for i in range(particle_start, n_particles):

        particle = particles[i]

        print(f"Partikel {i + 1}/{n_particles}")

        current_position = particle['position']

        current_value = objective_function(
            current_position,
            folds,
            X_train,
            y_train
        )

        # update pbest
        if current_value < particle['pbest_value']:
            particle['pbest_value'] = current_value
            particle['pbest_position'] = (current_position.copy())

        # update gbest
        if current_value < gbest_value:
            gbest_value = current_value
            gbest_position = (current_position.copy())

        save_pso_checkpoint(
            iter_count,
            i,
            particles,
            gbest_position,
            gbest_value
        )

    for particle in particles:
        for j in range(len(particle['position'])):

            r1 = random.random()
            r2 = random.random()

            # velocity update
            cognitive = c1 * r1 * (particle['pbest_position'][j] - particle['position'][j])
            social = c2 * r2 * (gbest_position[j] - particle['position'][j])

            v = (w * particle['velocity'][j]) + cognitive + social

            # velocity clamping
            v = max(-v_max[j], min(v_max[j], v))
            particle['velocity'][j] = v

            # update position
            new_pos = particle['position'][j] + v

            # ===== boundary handling =====
            if j == 0:  # n_filters
                new_pos = int(16 * round(new_pos / 16))
                new_pos = max(boundaries[0][0], min(boundaries[0][1], new_pos))

            elif j == 1:  # dropout
                new_pos = max(boundaries[1][0], min(boundaries[1][1], new_pos))

            elif j == 2:  # log_lr
                new_pos = max(boundaries[2][0], min(boundaries[2][1], new_pos))

            particle['position'][j] = new_pos

    start_particle = 0

    save_pso_checkpoint(
        iter_count + 1,
        -1,
        particles,
        gbest_position,
        gbest_value
    )


print("\nPSO Selesai!")
print(f"Best Loss: {gbest_value:.6f}")

best_n_filters = int(gbest_position[0])
best_dropout = float(gbest_position[1])
best_lr = 10 ** gbest_position[2]
kernel_size = 2

print("\nBest Parameter:")
print(best_n_filters, best_dropout, best_lr)

print("\nTraining final model...")

seed_losses = []
seed_models = []
seed_histories = []

SEED_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_DIR,
    "seeds"
)

os.makedirs(
    SEED_CHECKPOINT_DIR,
    exist_ok=True
)



for seed in range(42, 47):
    seed_model_path = os.path.join(SEED_CHECKPOINT_DIR,f"model_seed_{seed}.keras")
    seed_loss_path = os.path.join(SEED_CHECKPOINT_DIR,f"loss_seed_{seed}.npy")
    seed_loss_path = os.path.join(SEED_CHECKPOINT_DIR, f"loss_seed_{seed}.npy")
    seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

    if (os.path.exists(seed_model_path) and os.path.exists(seed_loss_path)):

        print(
            f"Seed {seed} sudah selesai. "
            f"Skip."
        )

        best_val_loss = float(np.load(seed_loss_path))
        loaded_model = tf.keras.models.load_model(seed_model_path)

        with open(seed_history_path, "rb") as f:
            loaded_history = pickle.load(f)

        seed_losses.append(best_val_loss)
        seed_models.append(loaded_model)
        seed_histories.append(loaded_history)
        continue

    print(f"\nTraining seed {seed}")
    set_seed(seed)
    tf.keras.backend.clear_session()

    model = ut.build_tcn_model(
        input_shape,
        best_n_filters,
        kernel_size,
        best_dropout,
        best_lr,
        n_future=1
    )

    early_stopping = ut.get_early_stopping(patience=100)

    history = model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    best_val_loss = min(history.history['val_loss'])

    model.save(seed_model_path)
    np.save(seed_loss_path, best_val_loss)

    with open(seed_history_path, "wb") as f:
        pickle.dump(history.history, f)

    seed_losses.append(best_val_loss)
    seed_models.append(model)
    seed_histories.append(history)

    print(f"Seed {seed} → " f"best val loss = {best_val_loss:.6f}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

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

best_model.save("output/model/model_tcn_pso.keras")

# --- Penanganan dinamis untuk objek History atau Dictionary ---
if hasattr(best_history, 'history'):
    # Jika best_history adalah objek Keras History (training baru)
    train_loss = best_history.history['loss']
    val_loss = best_history.history['val_loss']
else:
    # Jika best_history sudah berupa dictionary (di-load dari pickle)
    train_loss = best_history['loss']
    val_loss = best_history['val_loss']

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title(f'TCN Training Loss (Best Seed {42 + best_seed_idx})')
plt.legend()
plt.savefig('output/plots/train_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Plotting menggunakan best_history
plt.figure(figsize=(10, 4))
plt.plot(best_history['loss'], label='Train Loss')
plt.plot(best_history['val_loss'], label='Val Loss')
plt.title(f'TCN Training Loss (Best Seed {42 + best_seed_idx})')
plt.legend()
plt.savefig('output/plots/train_loss2.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluasi Prediksi
scaler = load('output/scaler/train_scaler.joblib')

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

plt.savefig('output/plots/evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

dump(predictions, 'output/model/model_PSO_TCN3.joblib')