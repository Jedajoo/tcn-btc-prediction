import os
import gc
import keras
import tensorflow as tf
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tcn import TCN

try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configured with dynamic memory growth: {len(gpus)} physical device(s)")
    except RuntimeError as e:
        print(f"GPU configuration notice: {e}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
SEED_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "seeds")
os.makedirs(SEED_CHECKPOINT_DIR, exist_ok=True)

train = np.load("output/data/train_data.npz")
X_train, y_train = train["X"], train["y"]

test = np.load("output/data/test_data.npz")
X_test, y_test = test["X"], test["y"]

input_shape = (X_train.shape[1], X_train.shape[2])
print(f"Loaded dataset: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

seed_list = [42, 43, 44, 45, 46]
individual_predictions = []
seed_histories = {}
seed_metrics = []

for seed in seed_list:
    seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
    seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

    if os.path.exists(seed_model_path) and os.path.exists(seed_history_path):
        print(f"\n[Seed {seed}] Loading existing trained model and history...")
        model_tcn = tf.keras.models.load_model(
            seed_model_path,
            compile=False,
            safe_mode=False,
            custom_objects={'TCN': TCN}
        )
        with open(seed_history_path, "rb") as f:
            hist_data = pickle.load(f)
    else:
        print(f"\n--- Training TCN with Seed {seed} ---")
        set_seed(seed)
        tf.keras.backend.clear_session()

        model_tcn = ut.build_tcn_model(
            input_shape=input_shape,
            n_filters=120,
            k_size=2,
            dropout=0.01,
            learning_rate=10 ** (-2.5),
        )

        early_stopping = ut.get_early_stopping(patience=50)
        reduce_lr = ut.get_reduce_lr(monitor='val_loss', factor=0.5, patience=10, mode='min', verbose=0)

        history = model_tcn.fit(
            X_train, y_train,
            epochs=200,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        hist_data = history.history if hasattr(history, 'history') else history

        model_tcn.save(seed_model_path)
        with open(seed_history_path, "wb") as f:
            pickle.dump(hist_data, f)
        print(f"[Seed {seed}] Model and history saved to {SEED_CHECKPOINT_DIR}")

    seed_histories[seed] = hist_data

    # Predict on test set
    pred = model_tcn.predict(X_test, verbose=0).ravel()
    individual_predictions.append(pred)

    # Evaluate individual seed
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / (np.abs(y_test) + 1e-8))) * 100.0

    seed_metrics.append({
        'seed': seed,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    })

    print(f"[Seed {seed}] MAE: {mae:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

    del model_tcn
    tf.keras.backend.clear_session()
    gc.collect()

# ==========================================================
# 5-SEED ENSEMBLE EVALUATION
# ==========================================================
print("\n" + "=" * 78)
print("              5-SEED ENSEMBLE EVALUATION RESULTS")
print("=" * 78)

individual_predictions = np.array(individual_predictions)  # Shape: (5, len(y_test))
ensemble_pred = np.mean(individual_predictions, axis=0)
ensemble_std = np.std(individual_predictions, axis=0)

ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(ensemble_mse)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / (np.abs(y_test) + 1e-8))) * 100.0

# Print per-seed metrics table
print(f"{'Model / Seed':<16} | {'MAE':<10} | {'MSE':<12} | {'RMSE':<10} | {'R2 Score':<10} | {'MAPE (%)':<10}")
print("-" * 78)
for m in seed_metrics:
    print(f"Seed {m['seed']:<11} | {m['mae']:<10.6f} | {m['mse']:<12.6f} | {m['rmse']:<10.6f} | {m['r2']:<10.4f} | {m['mape']:<10.2f}%")

mean_mae = np.mean([m['mae'] for m in seed_metrics])
std_mae = np.std([m['mae'] for m in seed_metrics])
mean_mse = np.mean([m['mse'] for m in seed_metrics])
mean_rmse = np.mean([m['rmse'] for m in seed_metrics])
std_rmse = np.std([m['rmse'] for m in seed_metrics])
mean_r2 = np.mean([m['r2'] for m in seed_metrics])
std_r2 = np.std([m['r2'] for m in seed_metrics])
mean_mape = np.mean([m['mape'] for m in seed_metrics])
std_mape = np.std([m['mape'] for m in seed_metrics])

print("-" * 78)
print(f"{'Mean ± Std':<16} | {mean_mae:<10.6f} | {mean_mse:<12.6f} | {mean_rmse:<10.6f} | {mean_r2:<10.4f} | {mean_mape:<10.2f}%")
print("=" * 78)
print(f"{'ENSEMBLE (Mean)':<16} | {ensemble_mae:<10.6f} | {ensemble_mse:<12.6f} | {ensemble_rmse:<10.6f} | {ensemble_r2:<10.4f} | {ensemble_mape:<10.2f}%")
print("=" * 78)

# Performance improvement vs average single model
mae_improvement = ((mean_mae - ensemble_mae) / mean_mae) * 100.0
rmse_improvement = ((mean_rmse - ensemble_rmse) / mean_rmse) * 100.0
print(f"Ensemble improvement over Average Seed -> MAE: {mae_improvement:+.2f}%, RMSE: {rmse_improvement:+.2f}%\n")

# Save evaluation summary and predictions
ensemble_summary = {
    'seed_list': seed_list,
    'seed_metrics': seed_metrics,
    'mean_metrics': {
        'mae': mean_mae, 'std_mae': std_mae,
        'mse': mean_mse,
        'rmse': mean_rmse, 'std_rmse': std_rmse,
        'r2': mean_r2, 'std_r2': std_r2,
        'mape': mean_mape, 'std_mape': std_mape
    },
    'ensemble_metrics': {
        'mae': ensemble_mae,
        'mse': ensemble_mse,
        'rmse': ensemble_rmse,
        'r2': ensemble_r2,
        'mape': ensemble_mape
    }
}

with open("output/model/ensemble_evaluation.pkl", "wb") as f:
    pickle.dump(ensemble_summary, f)

np.savez(
    "output/model/ensemble_predictions.npz",
    y_test=y_test,
    ensemble_pred=ensemble_pred,
    ensemble_std=ensemble_std,
    individual_predictions=individual_predictions,
    seed_list=np.array(seed_list)
)
print("Saved evaluation summary to output/model/ensemble_evaluation.pkl")
print("Saved ensemble predictions to output/model/ensemble_predictions.npz")

# ==========================================================
# VISUALIZATION & PLOTS
# ==========================================================
# Plot 1: Actual vs Predictions (Individual seeds, Ensemble, and Uncertainty band)
plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Actual y_test', color='black', linewidth=1.8, alpha=0.85)

for i, seed in enumerate(seed_list):
    plt.plot(individual_predictions[i], label=f'Seed {seed} (R2={seed_metrics[i]["r2"]:.3f})', linestyle='--', alpha=0.45, linewidth=1.0)

plt.plot(ensemble_pred, label=f'Ensemble Mean (R2={ensemble_r2:.3f})', color='crimson', linewidth=2.0)
plt.fill_between(
    range(len(y_test)),
    ensemble_pred - ensemble_std,
    ensemble_pred + ensemble_std,
    color='crimson',
    alpha=0.15,
    label='Ensemble Uncertainty (±1 Std)'
)
plt.title("5-Seed TCN Ensemble: Actual vs Predicted", fontsize=13, fontweight='bold')
plt.xlabel("Sample Index (Test Set)")
plt.ylabel("Target Value")
plt.legend(loc="best", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/plots/ensemble_actual_vs_pred.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Residual Analysis
residuals = y_test - ensemble_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(residuals, color='royalblue', alpha=0.75, linewidth=1.0)
ax1.axhline(0, color='red', linestyle='--', linewidth=1.2)
ax1.set_title("Ensemble Residuals over Test Horizon", fontsize=11, fontweight='bold')
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Error (Actual - Predicted)")
ax1.grid(True, alpha=0.3)

ax2.hist(residuals, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax2.set_title(f"Residual Distribution (Mean: {np.mean(residuals):.5f}, Std: {np.std(residuals):.5f})", fontsize=11, fontweight='bold')
ax2.set_xlabel("Error")
ax2.set_ylabel("Frequency")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/plots/ensemble_residuals.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Validation Loss Across Seeds
plt.figure(figsize=(12, 5))
for seed in seed_list:
    hist = seed_histories[seed]
    val_loss = hist.get('val_loss', [])
    if len(val_loss) > 0:
        plt.plot(val_loss, label=f"Seed {seed} (min={min(val_loss):.5f})", alpha=0.8, linewidth=1.4)

plt.title("TCN Multi-Seed Validation Loss (MSE) Across Epochs", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend(loc="best", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/plots/seed_training_losses.png", dpi=300, bbox_inches='tight')
plt.close()

print("All evaluation plots saved to output/plots/ (ensemble_actual_vs_pred.png, ensemble_residuals.png, seed_training_losses.png)")



