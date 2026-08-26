import os
import gc
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import dump, load
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import utils as ut

# Configure GPU memory if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]
        )
        print(f"GPU configured: {len(gpus)} physical device(s)")
    except RuntimeError as e:
        print(f"GPU configuration notice: {e}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

ticker = "BTC-USD"

# ==========================================
# 1. LOAD TABULAR PREPROCESSED DATA
# ==========================================
train_tab = np.load("output/data/train_tabular.npz")
test_tab = np.load("output/data/test_tabular.npz")

train_features = train_tab["features"]
train_target = train_tab["target"]
train_prices = train_tab["prices"]
train_returns = train_tab["returns"]
train_dates = train_tab["dates"]

test_features = test_tab["features"]
test_target = test_tab["target"]
test_prices = test_tab["prices"]
test_returns = test_tab["returns"]
test_dates = test_tab["dates"]

n_features = train_features.shape[1]
print(f"Loaded tabular data: Train samples={len(train_features)}, Test samples={len(test_features)}, Features={n_features}")

# Discrete allowed sets for kernel_size and time_window
# Strict constraints:
# 1. time_window strictly in [64, 128, 256]
# 2. kernel_size strictly in [2, 3, 5] (no 4)
# 3. Paired search strictly where time_window >= receptive_field
# 4. batch_size strictly fixed to 32
ALLOWED_WINDOWS = [64, 128, 256]
ALLOWED_KERNELS = [2, 3, 5]
FIXED_BATCH_SIZE = 32
DEFAULT_DILATIONS = [1, 2, 4, 8, 16]

# Build all valid pairs (k, w) where w >= receptive_field(k)
VALID_KW_PAIRS = []
for w in ALLOWED_WINDOWS:
    for k in ALLOWED_KERNELS:
        rf = ut.compute_receptive_field(k, DEFAULT_DILATIONS)
        if w >= rf:
            VALID_KW_PAIRS.append({
                "kernel_size": k,
                "time_window": w,
                "receptive_field": rf,
                "dilations": DEFAULT_DILATIONS
            })

print(f"\nFixed Batch Size: {FIXED_BATCH_SIZE}")
print("--- Valid (Kernel Size, Time Window) Pairs (Window >= Receptive Field) ---")
for idx, pair in enumerate(VALID_KW_PAIRS):
    print(f"  [{idx}] Kernel = {pair['kernel_size']}, Window = {pair['time_window']:3d} | Receptive Field = {pair['receptive_field']:3d} ({pair['time_window']} >= {pair['receptive_field']})")

def decode_hyperparameters(p):
    """
    Decodes continuous PSO particle position into concrete hyperparameters.
    """
    # 0. n_filters (multiple of 16 in [32, 256])
    n_filters = int(16 * round(p[0] / 16.0))
    n_filters = max(32, min(256, n_filters))

    # 1. dropout in [0.01, 0.40]
    dropout = float(np.clip(p[1], 0.01, 0.40))

    # 2. learning_rate
    log_lr = float(np.clip(p[2], -4.5, -2.5))
    learning_rate = 10.0 ** log_lr

    # 3. Paired (kernel_size, time_window) search strictly satisfying window >= receptive_field
    pair_idx = int(np.clip(np.floor(p[3]), 0, len(VALID_KW_PAIRS) - 1))
    kw_pair = VALID_KW_PAIRS[pair_idx]
    k_size = kw_pair["kernel_size"]
    time_window = kw_pair["time_window"]
    receptive_field = kw_pair["receptive_field"]
    dilations = kw_pair["dilations"]

    # 4. weight_decay
    log_wd = float(np.clip(p[4], -5.0, -2.0))
    weight_decay = 10.0 ** log_wd

    return {
        "n_filters": n_filters,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "kernel_size": k_size,
        "time_window": time_window,
        "receptive_field": receptive_field,
        "dilations": dilations,
        "batch_size": FIXED_BATCH_SIZE,
        "weight_decay": weight_decay,
        "pair_idx": pair_idx,
        "log_lr": log_lr,
        "log_wd": log_wd
    }

# ==========================================
# 2. PSO OBJECTIVE FUNCTION
# ==========================================
def objective_function(params, train_features, train_target):
    hp = decode_hyperparameters(params)
    
    # Generate sequences dynamically based on time_window
    X_train, y_train = ut.create_sequences(train_features, train_target, hp["time_window"])
    
    if len(X_train) < 100:
        return 999.0  # Invalid window penalty
        
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = list(tscv.split(X_train))

    fold_losses = []
    input_shape = (hp["time_window"], n_features)

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        set_seed(42 + fold_idx)
        tf.keras.backend.clear_session()

        model = ut.build_tcn_attention_model(
            input_shape=input_shape,
            n_filters=hp["n_filters"],
            k_size=hp["kernel_size"],
            dropout=hp["dropout"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            dilations=hp["dilations"]
        )

        early_stop = ut.get_early_stopping(patience=20, monitor='val_loss')

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=hp["batch_size"],
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        best_val_loss = min(history.history['val_loss'])
        fold_losses.append(best_val_loss)

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    fitness = mean_loss + 0.2 * std_loss

    print(
        f"  [Eval] Filters={hp['n_filters']}, K={hp['kernel_size']}, "
        f"Window={hp['time_window']} (RF={hp['receptive_field']}), "
        f"Drop={hp['dropout']:.3f}, LR={hp['learning_rate']:.5f}, "
        f"Batch={hp['batch_size']}, WD={hp['weight_decay']:.5f} | "
        f"ValLoss={mean_loss:.4f} (std={std_loss:.4f}) | Fitness={fitness:.4f}"
    )

    return fitness

# ==========================================
# 3. PSO CHECKPOINTING & BOUNDARIES
# ==========================================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
PSO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pso_state.pkl")

# Boundaries for the 5 parameters:
# [0] n_filters: [32, 256]
# [1] dropout: [0.01, 0.40]
# [2] log_lr: [-4.5, -2.5]
# [3] kw_pair_idx: [0.0, len(VALID_KW_PAIRS) - 1e-4] (maps to valid (K, W) pairs)
# [4] log_weight_decay: [-5.0, -2.0]
boundaries = [
    (32, 256),
    (0.01, 0.40),
    (-4.5, -2.5),
    (0.0, len(VALID_KW_PAIRS) - 1e-4),
    (-5.0, -2.0)
]

v_max = [(b[1] - b[0]) * 0.20 for b in boundaries]

def save_pso_checkpoint(iter_count, particle_idx, particles, gbest_position, gbest_value):
    state = {
        "iter_count": iter_count,
        "particle_idx": particle_idx,
        "particles": particles,
        "gbest_position": gbest_position,
        "gbest_value": gbest_value,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state()
    }
    temp_file = PSO_CHECKPOINT + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(state, f)
    os.replace(temp_file, PSO_CHECKPOINT)

def load_pso_checkpoint():
    if not os.path.exists(PSO_CHECKPOINT):
        return None
    try:
        with open(PSO_CHECKPOINT, "rb") as f:
            state = pickle.load(f)
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        print(f"Resuming PSO from checkpoint: Iteration {state['iter_count'] + 1}, Particle {state['particle_idx'] + 1}")
        return state
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return None

# PSO Configuration
n_particles = 20
n_iterations = 30
c1 = 1.8
c2 = 2.2

set_seed(42)
checkpoint = load_pso_checkpoint()

if checkpoint is None or len(checkpoint.get("particles", [{}])[0].get("position", [])) != len(boundaries):
    particles = []
    for _ in range(n_particles):
        pos = [random.uniform(b[0], b[1]) for b in boundaries]
        vel = [0.0] * len(boundaries)
        particles.append({
            'position': pos,
            'velocity': vel,
            'pbest_position': pos.copy(),
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

# ==========================================
# 4. PSO OPTIMIZATION LOOP
# ==========================================
print(f"\nStarting PSO Optimization ({n_iterations} iterations, {n_particles} particles)...")

for iter_count in range(start_iter, n_iterations):
    w = 0.9 - (0.5 * iter_count / n_iterations)
    print(f"\n--- PSO Iteration {iter_count + 1}/{n_iterations} (Inertia w={w:.3f}) ---")

    p_start = start_particle if iter_count == start_iter else 0

    for i in range(p_start, n_particles):
        particle = particles[i]
        print(f"Evaluating Particle {i + 1}/{n_particles}:")
        
        current_val = objective_function(
            particle['position'],
            train_features,
            train_target
        )

        if current_val < particle['pbest_value']:
            particle['pbest_value'] = current_val
            particle['pbest_position'] = particle['position'].copy()

        if current_val < gbest_value:
            gbest_value = current_val
            gbest_position = particle['position'].copy()
            print(f"  >>> New Global Best Loss: {gbest_value:.5f}")

        save_pso_checkpoint(iter_count, i, particles, gbest_position, gbest_value)

    # Velocity and position updates
    for particle in particles:
        for j in range(len(boundaries)):
            r1, r2 = random.random(), random.random()
            cog = c1 * r1 * (particle['pbest_position'][j] - particle['position'][j])
            soc = c2 * r2 * (gbest_position[j] - particle['position'][j])
            v = (w * particle['velocity'][j]) + cog + soc
            v = max(-v_max[j], min(v_max[j], v))
            particle['velocity'][j] = v

            new_pos = particle['position'][j] + v
            new_pos = max(boundaries[j][0], min(boundaries[j][1], new_pos))
            particle['position'][j] = new_pos

    start_particle = 0
    save_pso_checkpoint(iter_count + 1, -1, particles, gbest_position, gbest_value)

print("\n==========================================")
print("PSO OPTIMIZATION COMPLETED!")
best_hp = decode_hyperparameters(gbest_position)
print(f"Best Fitness Value : {gbest_value:.5f}")
print("Optimal Hyperparameters:")
for k, v in best_hp.items():
    if not k.startswith(('log_', 'k_', 'b_')):
        print(f"  - {k:15s}: {v}")
print("==========================================\n")

# Save best hyperparameters
dump(best_hp, "output/model/best_hyperparameters.joblib")

# ==========================================
# 5. MULTI-SEED ENSEMBLE TRAINING (5 SEEDS)
# ==========================================
print("Training Multi-Seed Ensemble Models with Optimal Parameters...")

best_window = best_hp["time_window"]
X_train_final, y_train_final = ut.create_sequences(train_features, train_target, best_window)

# Build test sequences with lookback buffer
test_input_features = np.vstack([
    train_features[-best_window+1:],
    test_features
])
test_input_target = np.concatenate([
    train_target[-best_window+1:],
    test_target
])
X_test_final, y_test_final = ut.create_sequences(test_input_features, test_input_target, best_window)

SEED_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "seeds")
os.makedirs(SEED_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

seed_list = [42, 43, 44, 45, 46]
seed_models = []
seed_histories = []
seed_val_losses = []

plt.figure(figsize=(10, 5))

for seed in seed_list:
    seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
    seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

    if os.path.exists(seed_model_path) and os.path.exists(seed_history_path):
        print(f"Loading existing trained model for Seed {seed}...")
        model = tf.keras.models.load_model(seed_model_path)
        with open(seed_history_path, "rb") as f:
            history = pickle.load(f)
    else:
        print(f"\n--- Training Model with Seed {seed} ---")
        set_seed(seed)
        tf.keras.backend.clear_session()

        model = ut.build_tcn_attention_model(
            input_shape=(best_window, n_features),
            n_filters=best_hp["n_filters"],
            k_size=best_hp["kernel_size"],
            dropout=best_hp["dropout"],
            learning_rate=best_hp["learning_rate"],
            weight_decay=best_hp["weight_decay"],
            dilations=best_hp.get("dilations", [1, 2, 4, 8, 16])
        )

        early_stopping = ut.get_early_stopping(patience=40, monitor='val_loss')

        history_obj = model.fit(
            X_train_final,
            y_train_final,
            epochs=200,
            batch_size=best_hp["batch_size"],
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping],
            verbose=1
        )

        history = history_obj.history
        model.save(seed_model_path)
        with open(seed_history_path, "wb") as f:
            pickle.dump(history, f)

    seed_models.append(model)
    seed_histories.append(history)
    min_val_loss = min(history['val_loss'])
    seed_val_losses.append(min_val_loss)
    
    plt.plot(history['val_loss'], label=f"Seed {seed} Val (min={min_val_loss:.4f})")

plt.title("TCN + Multi-Head Attention Multi-Seed Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("output/plots/training_losses.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 6. ENSEMBLE AVERAGING & EVALUATION
# ==========================================
print("\n--- Generating Multi-Seed Ensemble Predictions ---")
individual_predictions = []

for idx, model in enumerate(seed_models):
    prob_pred = model.predict(X_test_final, verbose=0).ravel()
    individual_predictions.append(prob_pred)
    seed_metrics = ut.compute_classification_metrics(y_test_final, prob_pred)
    print(f"Seed {seed_list[idx]}: Acc={seed_metrics['accuracy']*100:.2f}%, AUC={seed_metrics['auc']:.4f}, LogLoss={seed_metrics['log_loss']:.4f}")

# Compute Ensemble Average Probability: (1/K) * sum(p_k)
ensemble_probabilities = np.mean(individual_predictions, axis=0)
ensemble_metrics = ut.compute_classification_metrics(y_test_final, ensemble_probabilities)

# Target prices and returns for financial evaluation
# Align test returns to sequences
test_eval_returns = test_returns[-len(y_test_final):]
test_eval_prices = test_prices[-len(y_test_final):]

backtest_res = ut.backtest_directional_strategy(test_eval_returns, ensemble_probabilities)

print("\n==========================================")
print("FINAL ENSEMBLE TEST PERFORMANCE")
print("==========================================")
print(f"Ensemble Accuracy       : {ensemble_metrics['accuracy']*100:.2f}%")
print(f"Ensemble ROC-AUC Score  : {ensemble_metrics['auc']:.4f}")
print(f"Ensemble Precision      : {ensemble_metrics['precision']*100:.2f}%")
print(f"Ensemble Recall         : {ensemble_metrics['recall']*100:.2f}%")
print(f"Ensemble F1-Score       : {ensemble_metrics['f1']:.4f}")
print(f"Brier Score (Calibration): {ensemble_metrics['brier_score']:.4f}")
print(f"Mean Prediction Certainty: {ensemble_metrics['confidence_mean']:.2f}%")
print(f"High-Confidence Accuracy: {ensemble_metrics['high_conf_acc']*100:.2f}% (Coverage: {ensemble_metrics['high_conf_coverage']:.1f}%)")
print(f"Strategy Cumulative Ret : {backtest_res['total_strategy_return']:.2f}% (vs Market Buy&Hold: {backtest_res['total_market_return']:.2f}%)")
print(f"Strategy Sharpe Ratio   : {backtest_res['sharpe_ratio']:.2f}")
print("==========================================\n")

# Save evaluation results
evaluation_summary = {
    "ensemble_metrics": ensemble_metrics,
    "backtest_results": backtest_res,
    "best_hyperparameters": best_hp,
    "seed_val_losses": seed_val_losses
}
dump(evaluation_summary, "output/model/ensemble_evaluation.joblib")

# ==========================================
# 7. GENERATE COMPREHENSIVE VISUALIZATIONS
# ==========================================
# A. ROC Curve
fpr, tpr, _ = roc_curve(y_test_final, ensemble_probabilities)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble ROC (AUC = {ensemble_metrics["auc"]:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title(f'ROC Curve - Directional Movement ({ticker})')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('output/plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# B. Confusion Matrix
cm = ensemble_metrics['confusion_matrix']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Directional Movement Confusion Matrix')
plt.savefig('output/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# C. Probability & Confidence Distribution
confidences = np.abs(ensemble_probabilities - 0.5) * 2.0 * 100.0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(ensemble_probabilities, bins=25, color='teal', edgecolor='black', alpha=0.7)
ax1.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
ax1.set_title('Predicted Probability Distribution P(Up)')
ax1.set_xlabel('Probability P(Up)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(confidences, bins=25, color='purple', edgecolor='black', alpha=0.7)
ax2.set_title('Prediction Confidence Distribution (%)')
ax2.set_xlabel('Confidence (%) [0% = Uncertain, 100% = Certain]')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plots/probability_confidence.png', dpi=300, bbox_inches='tight')
plt.close()

# D. Directional Signal Backtest vs Actual Prices
plt.figure(figsize=(14, 7))
ax_top = plt.subplot(2, 1, 1)
ax_top.plot(test_eval_prices, label=f'{ticker} Actual Close Price', color='black', alpha=0.8)
up_signals = np.where(ensemble_probabilities >= 0.5)[0]
down_signals = np.where(ensemble_probabilities < 0.5)[0]
ax_top.scatter(up_signals, test_eval_prices[up_signals], color='green', marker='^', s=25, label='Signal UP (P >= 0.5)', alpha=0.7)
ax_top.scatter(down_signals, test_eval_prices[down_signals], color='red', marker='v', s=25, label='Signal DOWN (P < 0.5)', alpha=0.7)
ax_top.set_title(f'Ensemble Model Directional Signals vs {ticker} Price')
ax_top.legend(loc='upper left')
ax_top.grid(True, alpha=0.3)

ax_bot = plt.subplot(2, 1, 2)
ax_bot.plot(backtest_res['cumulative_strategy'] * 100, label=f'PSO-TCN Strategy ({backtest_res["total_strategy_return"]:.1f}%)', color='green', lw=1.8)
ax_bot.plot(backtest_res['cumulative_market'] * 100, label=f'Buy & Hold Benchmark ({backtest_res["total_market_return"]:.1f}%)', color='gray', linestyle='--', lw=1.5)
ax_bot.set_title('Cumulative Return Backtest (%)')
ax_bot.set_xlabel('Test Sample Days')
ax_bot.set_ylabel('Cumulative Return (%)')
ax_bot.legend(loc='upper left')
ax_bot.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plots/trading_backtest.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved in output/plots/")