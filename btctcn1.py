import os
import sys
import gc
import keras
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import argparse
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load

def init_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured with dynamic memory growth: {len(gpus)} physical device(s)")
        except RuntimeError as e:
            print(f"GPU configuration notice: {e}")


try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

FIXED_TIME_WINDOW = 120
FIXED_KERNEL_SIZE = 2
FIXED_BATCH_SIZE = 32
DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32]

boundaries = [
    (32, 256),     # [0] n_filters
    (0.05, 0.40),  # [2] dropout
    (-4.5, -2.5),  # [3] log_lr
    (-5.0, -2.0)   # [4] log_weight_decay
]

# GWO-WOA Metaheuristic Parameters
n_agents = 20
n_iterations = 30

# Checkpoint paths
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GWO_WOA_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gwo_woa_state.pkl")
SEED_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "seeds")
os.makedirs(SEED_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

def load_tabular_data():
    train_tab = np.load("output/data/train_tabular.npz")
    test_tab = np.load("output/data/test_tabular.npz")
    return train_tab, test_tab

def decode_hyperparameters(p):
    """
    Decodes 5-dimensional continuous GWO-WOA position into concrete hyperparameters.
    """
    # 0. n_filters (multiple of 16 in [32, 256])
    n_filters = int(16 * round(p[0] / 16.0))
    n_filters = max(32, min(256, n_filters))

    # 2. dropout in [0.05, 0.40]
    dropout = float(np.clip(p[1], 0.05, 0.40))

    # 3. learning_rate
    log_lr = float(np.clip(p[2], -4.5, -2.5))
    learning_rate = 10.0 ** log_lr

    # 4. weight_decay
    log_wd = float(np.clip(p[3], -5.0, -2.0))
    weight_decay = 10.0 ** log_wd

    return {
        "n_filters": n_filters,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "kernel_size": FIXED_KERNEL_SIZE,
        "time_window": FIXED_TIME_WINDOW,
        "dilations": DEFAULT_DILATIONS,
        "batch_size": FIXED_BATCH_SIZE,
        "weight_decay": weight_decay,
        "log_lr": log_lr,
        "log_wd": log_wd
    }

def save_gwo_woa_checkpoint(
    iter_count,
    agent_idx,
    agents,
    alpha_pos,
    alpha_score,
    beta_pos,
    beta_score,
    delta_pos,
    delta_score
):
    state = {
        "iter_count": iter_count,
        "agent_idx": agent_idx,
        "agents": agents,
        "alpha_pos": alpha_pos,
        "alpha_score": alpha_score,
        "beta_pos": beta_pos,
        "beta_score": beta_score,
        "delta_pos": delta_pos,
        "delta_score": delta_score,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state()
    }
    temp_file = GWO_WOA_CHECKPOINT + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(state, f)
    os.replace(temp_file, GWO_WOA_CHECKPOINT)

def load_gwo_woa_checkpoint():
    if not os.path.exists(GWO_WOA_CHECKPOINT):
        return None
    try:
        with open(GWO_WOA_CHECKPOINT, "rb") as f:
            state = pickle.load(f)
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        return state
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return None

def precache_sequence_data(train_features, train_target):
    import utils as ut
    X_all, y_all = ut.create_sequences(train_features, train_target, FIXED_TIME_WINDOW)
    val_size = max(1, int(len(X_all) * 0.20))
    return {
        FIXED_TIME_WINDOW: {
            "X_tr": X_all[:-val_size],
            "y_tr": y_all[:-val_size],
            "X_val": X_all[-val_size:],
            "y_val": y_all[-val_size:],
            "n_samples": len(X_all)
        }
    }

def objective_function(params, precomputed_data, n_features):
    import tensorflow as tf
    import utils as ut

    hp = decode_hyperparameters(params)
    data = precomputed_data.get(hp["time_window"])

    if data is None or data["n_samples"] < 100:
        return 999.0

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]

    set_seed(42)
    tf.keras.backend.clear_session()

    model = ut.build_tcn_model(
        input_shape=(hp["time_window"], n_features),
        n_filters=hp["n_filters"],
        k_size=hp['kernel_size'],
        dropout=hp["dropout"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        dilations=hp["dilations"]
    )

    early_stop = ut.get_early_stopping(patience=20, monitor='val_loss', mode='min', verbose=0)
    reduce_lr = ut.get_reduce_lr(monitor='val_loss', factor=0.5, patience=10, mode='min', verbose=0)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=hp["batch_size"],
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    best_val_loss = float(min(history.history.get('val_loss', [999.0])))
    fitness = best_val_loss

    # Strict memory cleanup
    del model
    del history
    tf.keras.backend.clear_session()
    gc.collect()

    print(
        f"  [Eval] Filters={hp['n_filters']}, "
        f"Window={hp['time_window']}, "
        f"Drop={hp['dropout']:.3f}, LR={hp['learning_rate']:.5f}, "
        f"Batch={hp['batch_size']}, WD={hp['weight_decay']:.5f} | "
        f"ValLoss={best_val_loss:.5f}| Fitness={fitness:.5f}"
    )

    return fitness

def run_gwo_woa_worker():
    """
    Executes exactly 1 GWO-WOA iteration (or remaining agents of an interrupted iteration),
    updates positions via hybrid GWO encircling + WOA spiral bubble-net, saves checkpoint, and exits.
    Returns:
        10: Iteration completed, more iterations left.
        20: All iterations completed.
    """
    init_gpu()
    train_tab, test_tab = load_tabular_data()
    train_features = train_tab["features"]
    train_target = train_tab["target"]  # Next-day directional binary target (0 or 1)
    n_features = train_features.shape[1]

    precomputed_data = precache_sequence_data(train_features, train_target)

    set_seed(42)
    checkpoint = load_gwo_woa_checkpoint()

    if checkpoint is None or len(checkpoint.get("agents", [{}])[0].get("position", [])) != len(boundaries):
        agents = []
        for _ in range(n_agents):
            pos = [random.uniform(b[0], b[1]) for b in boundaries]
            agents.append({
                'position': pos,
                'fitness': float('inf')
            })
        alpha_pos = None
        alpha_score = float('inf')
        beta_pos = None
        beta_score = float('inf')
        delta_pos = None
        delta_score = float('inf')
        start_iter = 0
        start_agent = 0
    else:
        agents = checkpoint["agents"]
        alpha_pos = checkpoint["alpha_pos"]
        alpha_score = checkpoint["alpha_score"]
        beta_pos = checkpoint["beta_pos"]
        beta_score = checkpoint["beta_score"]
        delta_pos = checkpoint["delta_pos"]
        delta_score = checkpoint["delta_score"]
        start_iter = checkpoint["iter_count"]
        start_agent = checkpoint["agent_idx"] + 1

    # Check if already complete
    if start_iter >= n_iterations and checkpoint is not None and checkpoint.get("agent_idx") == -1:
        print(f"GWO-WOA Optimization is already complete ({start_iter}/{n_iterations} iterations).")
        best_hp = decode_hyperparameters(alpha_pos)
        dump(best_hp, "output/model/best_hyperparameters.joblib")
        return 20

    iter_count = start_iter
    a = 2.0 - (2.0 * iter_count / n_iterations)  # Linearly decreases from 2 to 0
    print(f"\n--- [Process Worker] Hybrid GWO-WOA Iteration {iter_count + 1}/{n_iterations} (a={a:.3f}) ---")

    for i in range(start_agent, n_agents):
        agent = agents[i]
        print(f"Evaluating Search Agent {i + 1}/{n_agents}:")

        fitness = objective_function(
            agent['position'],
            precomputed_data,
            n_features
        )
        agent['fitness'] = fitness

        # Update Alpha, Beta, Delta wolves
        if fitness < alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy() if beta_pos is not None else None
            beta_score = alpha_score
            beta_pos = alpha_pos.copy() if alpha_pos is not None else None
            alpha_score = fitness
            alpha_pos = agent['position'].copy()
            print(f"  >>> New Alpha Leader Fitness (ValLoss): {alpha_score:.5f}")
        elif fitness < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy() if beta_pos is not None else None
            beta_score = fitness
            beta_pos = agent['position'].copy()
            print(f"  >>> New Beta Leader Fitness (ValLoss): {beta_score:.5f}")
        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = agent['position'].copy()
            print(f"  >>> New Delta Leader Fitness (ValLoss): {delta_score:.5f}")

        # Fallback initialization if leaders are not yet populated
        if beta_pos is None:
            beta_pos = alpha_pos.copy()
            beta_score = alpha_score
        if delta_pos is None:
            delta_pos = alpha_pos.copy()
            delta_score = alpha_score

        save_gwo_woa_checkpoint(
            iter_count, i, agents,
            alpha_pos, alpha_score,
            beta_pos, beta_score,
            delta_pos, delta_score
        )

    # Hybrid Position Updates (GWO Encircling + WOA Spiral Bubble-Net)
    b = 1.0  # Spiral constant for WOA
    for agent in agents:
        p = random.random()
        pos = agent['position']
        new_pos = [0.0] * len(boundaries)

        if p < 0.5:
            # GWO Encircling / Exploitation guided by Alpha, Beta, Delta or Random Exploration
            for j in range(len(boundaries)):
                r1_1, r2_1 = random.random(), random.random()
                r1_2, r2_2 = random.random(), random.random()
                r1_3, r2_3 = random.random(), random.random()

                A1 = 2.0 * a * r1_1 - a
                C1 = 2.0 * r2_1
                A2 = 2.0 * a * r1_2 - a
                C2 = 2.0 * r2_2
                A3 = 2.0 * a * r1_3 - a
                C3 = 2.0 * r2_3

                if abs(A1) < 1.0:
                    # GWO Encircling around Alpha, Beta, Delta
                    D_alpha = abs(C1 * alpha_pos[j] - pos[j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    D_beta = abs(C2 * beta_pos[j] - pos[j])
                    X2 = beta_pos[j] - A2 * D_beta

                    D_delta = abs(C3 * delta_pos[j] - pos[j])
                    X3 = delta_pos[j] - A3 * D_delta

                    new_val = (X1 + X2 + X3) / 3.0
                else:
                    # Exploration: Search for prey with random agent
                    rand_agent = random.choice(agents)
                    D_rand = abs(C1 * rand_agent['position'][j] - pos[j])
                    new_val = rand_agent['position'][j] - A1 * D_rand

                new_pos[j] = max(boundaries[j][0], min(boundaries[j][1], new_val))
        else:
            # WOA Bubble-net spiral update around Alpha Leader
            l = random.uniform(-1.0, 1.0)
            for j in range(len(boundaries)):
                dist_to_alpha = abs(alpha_pos[j] - pos[j])
                spiral_step = dist_to_alpha * np.exp(b * l) * np.cos(2.0 * np.pi * l) + alpha_pos[j]
                new_pos[j] = max(boundaries[j][0], min(boundaries[j][1], spiral_step))

        agent['position'] = new_pos

    next_iter = iter_count + 1
    save_gwo_woa_checkpoint(
        next_iter, -1, agents,
        alpha_pos, alpha_score,
        beta_pos, beta_score,
        delta_pos, delta_score
    )

    print(f"--- [Process Worker] Iteration {iter_count + 1} completed and saved to checkpoint ---")

    if next_iter >= n_iterations:
        print("\n==========================================")
        print("HYBRID GWO-WOA OPTIMIZATION COMPLETED!")
        best_hp = decode_hyperparameters(alpha_pos)
        print(f"Best Fitness Value (Val RMSE) : {alpha_score:.5f}")
        print("Optimal Hyperparameters:")
        for k, v in best_hp.items():
            if not k.startswith(('log_', 'k_', 'b_')):
                print(f"  - {k:15s}: {v}")
        print("==========================================\n")
        dump(best_hp, "output/model/best_hyperparameters.joblib")
        return 20

    return 10

def run_final_ensemble():
    import tensorflow as tf
    import utils as ut
    from tcn import TCN

    init_gpu()

    hp_path = "output/model/best_hyperparameters.joblib"
    if os.path.exists(hp_path):
        best_hp = load(hp_path)
    else:
        cp = load_gwo_woa_checkpoint()
        if cp and cp.get("alpha_pos") is not None:
            best_hp = decode_hyperparameters(cp["alpha_pos"])
            dump(best_hp, hp_path)
        else:
            raise FileNotFoundError("Could not find best_hyperparameters.joblib or valid GWO-WOA checkpoint.")

    train = np.load("output/data/train_data.npz")
    X_train, y_train = train["X"], train["y"]

    test = np.load("output/data/test_data.npz")
    X_test, y_test = test["X"], test["y"]

    # Temporal split: Last 20% of train set reserved for validation & seed ranking (Option B)
    val_size = max(1, int(len(X_train) * 0.20))
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Loaded dataset: X_tr={X_tr.shape}, y_tr={y_tr.shape}, X_val={X_val.shape}, y_val={y_val.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

    TARGET_SCALER_PATH = "output/scaler/target_scaler.joblib"
    # Inverse transform target test (y_test) ke skala harga asli (USD)
    y_test_inv = ut.inverse_transform_target(y_test, TARGET_SCALER_PATH)

    seed_list = [42, 43, 44, 45, 46]
    individual_predictions = []          # Inverse transformed (USD)
    individual_predictions_scaled = []   # Scaled
    seed_histories = {}
    seed_metrics = []

    for seed in seed_list:
        seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
        seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

        loaded_existing = False
        if os.path.exists(seed_model_path) and os.path.exists(seed_history_path):
            try:
                print(f"\n[Seed {seed}] Checking existing checkpoint...")
                model_tcn = tf.keras.models.load_model(
                    seed_model_path,
                    compile=False,
                    safe_mode=False,
                    custom_objects={'TCN': TCN}
                )
                if model_tcn.input_shape[1:] == input_shape:
                    with open(seed_history_path, "rb") as f:
                        hist_data = pickle.load(f)
                    loaded_existing = True
                    print(f"[Seed {seed}] Loaded existing trained model and history successfully.")
                else:
                    print(f"[Seed {seed}] Model input shape {model_tcn.input_shape[1:]} does not match current data shape {input_shape}. Retraining...")
                    del model_tcn
                    tf.keras.backend.clear_session()
            except Exception as e:
                print(f"[Seed {seed}] Could not load checkpoint ({e}). Retraining...")
                loaded_existing = False

        if not loaded_existing:
            print(f"\n--- Training TCN with Seed {seed} ---")
            set_seed(seed)
            tf.keras.backend.clear_session()

            model_tcn = ut.build_tcn_model(
                input_shape=input_shape,
                n_filters=best_hp["n_filters"],
                k_size=best_hp["kernel_size"],
                dropout=best_hp["dropout"],
                learning_rate=best_hp["learning_rate"],
                weight_decay=best_hp["weight_decay"]
            )

            early_stopping = ut.get_early_stopping(patience=50)
            reduce_lr = ut.get_reduce_lr(monitor='val_loss', factor=0.5, patience=10, mode='min', verbose=0)

            history = model_tcn.fit(
                X_tr, y_tr,
                epochs=200,
                batch_size=best_hp["batch_size"],
                validation_data=(X_val, y_val),
                shuffle=False,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            hist_data = history.history if hasattr(history, 'history') else history

            model_tcn.save(seed_model_path)
            with open(seed_history_path, "wb") as f:
                pickle.dump(hist_data, f)
            print(f"[Seed {seed}] Model and history saved to {SEED_CHECKPOINT_DIR}")

        seed_histories[seed] = hist_data

        # Predict on validation set to evaluate seed reliability without data snooping (Option B)
        val_pred_raw = model_tcn.predict(X_val, verbose=0)
        if val_pred_raw.ndim == 3:
            val_pred = val_pred_raw[:, -1, :].ravel()
        elif val_pred_raw.ndim == 2 and val_pred_raw.shape[1] > 1:
            val_pred = val_pred_raw[:, -1].ravel()
        else:
            val_pred = val_pred_raw.ravel()

        val_mae = mean_absolute_error(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, val_pred)

        # Predict on test set (handle both 3D sequence outputs and 2D/1D single-step outputs)
        pred_raw = model_tcn.predict(X_test, verbose=0)
        if pred_raw.ndim == 3:
            pred_scaled = pred_raw[:, -1, :].ravel()
        elif pred_raw.ndim == 2 and pred_raw.shape[1] > 1:
            pred_scaled = pred_raw[:, -1].ravel()
        else:
            pred_scaled = pred_raw.ravel()
        individual_predictions_scaled.append(pred_scaled)

        # Inverse transform prediction ke skala harga asli (USD) sebelum evaluasi
        pred_inv = ut.inverse_transform_target(pred_scaled, TARGET_SCALER_PATH)
        individual_predictions.append(pred_inv)

        # Evaluasi individual seed menggunakan harga asli (USD)
        mae = mean_absolute_error(y_test_inv, pred_inv)
        mse = mean_squared_error(y_test_inv, pred_inv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_inv, pred_inv)
        mape = np.mean(np.abs((y_test_inv - pred_inv) / (np.abs(y_test_inv) + 1e-8))) * 100.0
        da = ut.compute_directional_accuracy(y_test_inv, pred_inv)

        seed_metrics.append({
            'seed': seed,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'da': da,
            'pred_inv': pred_inv,
            'pred_scaled': pred_scaled
        })

        print(f"[Seed {seed}] Val MSE: {val_mse:.6f} (Val R2: {val_r2:.4f}) | Test MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}% | DA: {da:.2f}%")

        del model_tcn
        tf.keras.backend.clear_session()
        gc.collect()

    # ==========================================================
    # OPTION B: TOP-K SEED FILTERING (BASED ON VALIDATION PERFORMANCE)
    # ==========================================================
    TOP_K = 3
    # Rank seeds purely based on validation set performance (val_mse ascending)
    ranked_seeds = sorted(seed_metrics, key=lambda m: m['val_mse'])
    top_seeds = ranked_seeds[:TOP_K]
    top_seed_ids = [s['seed'] for s in top_seeds]
    excluded_seeds = ranked_seeds[TOP_K:]
    excluded_seed_ids = [s['seed'] for s in excluded_seeds]

    print("\n" + "=" * 82)
    print(f"       VALIDATION-BASED SEED RANKING & TOP-{TOP_K} SELECTION (OPTION B)")
    print("=" * 82)
    print(f"{'Rank':<6} | {'Seed':<10} | {'Val MSE':<12} | {'Val RMSE':<12} | {'Val R2':<10} | {'Selection Status':<20}")
    print("-" * 82)
    for rank, s in enumerate(ranked_seeds, 1):
        status = f"SELECTED (Top-{TOP_K})" if s['seed'] in top_seed_ids else "FILTERED OUT"
        print(f"{rank:<6} | Seed {s['seed']:<5} | {s['val_mse']:<12.6f} | {s['val_rmse']:<12.6f} | {s['val_r2']:<10.4f} | {status:<20}")
    print("=" * 82)
    print(f"-> Top-{TOP_K} Selected Seeds for Ensemble : {top_seed_ids}")
    print(f"-> Filtered Out (Unreliable Seeds)          : {excluded_seed_ids}")

    # ==========================================================
    # ENSEMBLE EVALUATION: FULL 5-SEED vs TOP-K FILTERED ENSEMBLE
    # ==========================================================
    # 1. Full 5-Seed Ensemble
    all_test_preds = np.array([m['pred_inv'] for m in seed_metrics])
    full_ensemble_pred = np.mean(all_test_preds, axis=0)
    full_ensemble_std = np.std(all_test_preds, axis=0)
    full_mae = mean_absolute_error(y_test_inv, full_ensemble_pred)
    full_mse = mean_squared_error(y_test_inv, full_ensemble_pred)
    full_rmse = np.sqrt(full_mse)
    full_r2 = r2_score(y_test_inv, full_ensemble_pred)
    full_mape = np.mean(np.abs((y_test_inv - full_ensemble_pred) / (np.abs(y_test_inv) + 1e-8))) * 100.0
    full_da = ut.compute_directional_accuracy(y_test_inv, full_ensemble_pred)

    # 2. Top-K Filtered Ensemble (Option B)
    selected_test_preds = np.array([s['pred_inv'] for s in top_seeds])
    ensemble_pred = np.mean(selected_test_preds, axis=0)
    ensemble_std = np.std(selected_test_preds, axis=0)
    ensemble_mae = mean_absolute_error(y_test_inv, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test_inv, ensemble_pred)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_r2 = r2_score(y_test_inv, ensemble_pred)
    ensemble_mape = np.mean(np.abs((y_test_inv - ensemble_pred) / (np.abs(y_test_inv) + 1e-8))) * 100.0
    ensemble_da = ut.compute_directional_accuracy(y_test_inv, ensemble_pred)

    # Print per-seed test metrics table
    print("\n" + "=" * 96)
    print("       TEST SET EVALUATION RESULTS (ORIGINAL PRICE - USD)")
    print("=" * 96)
    print(f"{'Model / Seed':<22} | {'MAE ($)':<12} | {'RMSE ($)':<12} | {'R2 Score':<10} | {'MAPE (%)':<10} | {'Dir Acc (%)':<11}")
    print("-" * 96)
    for m in seed_metrics:
        tag = f" [Top-{TOP_K}]" if m['seed'] in top_seed_ids else " [Filtered]"
        name = f"Seed {m['seed']}{tag}"
        print(f"{name:<22} | ${m['mae']:<11,.2f} | ${m['rmse']:<11,.2f} | {m['r2']:<10.4f} | {m['mape']:<9.2f}% | {m['da']:<10.2f}%")

    mean_mae = np.mean([m['mae'] for m in seed_metrics])
    std_mae = np.std([m['mae'] for m in seed_metrics])
    mean_mse = np.mean([m['mse'] for m in seed_metrics])
    mean_rmse = np.mean([m['rmse'] for m in seed_metrics])
    std_rmse = np.std([m['rmse'] for m in seed_metrics])
    mean_r2 = np.mean([m['r2'] for m in seed_metrics])
    std_r2 = np.std([m['r2'] for m in seed_metrics])
    mean_mape = np.mean([m['mape'] for m in seed_metrics])
    std_mape = np.std([m['mape'] for m in seed_metrics])
    mean_da = np.mean([m['da'] for m in seed_metrics])
    std_da = np.std([m['da'] for m in seed_metrics])

    print("-" * 96)
    print(f"{'Mean ± Std (All 5)':<22} | ${mean_mae:<11,.2f} | ${mean_rmse:<11,.2f} | {mean_r2:<10.4f} | {mean_mape:<9.2f}% | {mean_da:<10.2f}%")
    print(f"{'Full 5-Seed Ensemble':<22} | ${full_mae:<11,.2f} | ${full_rmse:<11,.2f} | {full_r2:<10.4f} | {full_mape:<9.2f}% | {full_da:<10.2f}%")
    print("=" * 96)
    print(f"{f'TOP-{TOP_K} FILTERED ENS.':<22} | ${ensemble_mae:<11,.2f} | ${ensemble_rmse:<11,.2f} | {ensemble_r2:<10.4f} | {ensemble_mape:<9.2f}% | {ensemble_da:<10.2f}%")
    print("=" * 96)

    mae_diff = ((full_mae - ensemble_mae) / full_mae) * 100.0
    rmse_diff = ((full_rmse - ensemble_rmse) / full_rmse) * 100.0
    r2_diff = ensemble_r2 - full_r2
    da_diff = ensemble_da - full_da
    print(f"Top-{TOP_K} Filtered vs Full 5-Seed -> MAE: {mae_diff:+.2f}%, RMSE: {rmse_diff:+.2f}%, R2 Gain: {r2_diff:+.4f}, DA Gain: {da_diff:+.2f}%\n")

    # Save evaluation summary and predictions
    ensemble_summary = {
        'seed_list': seed_list,
        'top_k': TOP_K,
        'top_seed_ids': top_seed_ids,
        'excluded_seed_ids': excluded_seed_ids,
        'seed_metrics': seed_metrics,
        'mean_metrics': {
            'mae': mean_mae, 'std_mae': std_mae,
            'mse': mean_mse,
            'rmse': mean_rmse, 'std_rmse': std_rmse,
            'r2': mean_r2, 'std_r2': std_r2,
            'mape': mean_mape, 'std_mape': std_mape,
            'da': mean_da, 'std_da': std_da
        },
        'full_ensemble_metrics': {
            'mae': full_mae,
            'mse': full_mse,
            'rmse': full_rmse,
            'r2': full_r2,
            'mape': full_mape,
            'da': full_da
        },
        'top_k_ensemble_metrics': {
            'mae': ensemble_mae,
            'mse': ensemble_mse,
            'rmse': ensemble_rmse,
            'r2': ensemble_r2,
            'mape': ensemble_mape,
            'da': ensemble_da
        }
    }

    with open("output/model/ensemble_evaluation.pkl", "wb") as f:
        pickle.dump(ensemble_summary, f)

    np.savez(
        "output/model/ensemble_predictions.npz",
        y_test=y_test_inv,
        y_test_scaled=y_test,
        ensemble_pred=ensemble_pred,
        ensemble_std=ensemble_std,
        full_ensemble_pred=full_ensemble_pred,
        full_ensemble_std=full_ensemble_std,
        individual_predictions=individual_predictions,
        individual_predictions_scaled=np.array(individual_predictions_scaled),
        seed_list=np.array(seed_list),
        top_seed_ids=np.array(top_seed_ids),
        excluded_seed_ids=np.array(excluded_seed_ids)
    )
    print("Saved evaluation summary to output/model/ensemble_evaluation.pkl")
    print("Saved ensemble predictions to output/model/ensemble_predictions.npz")

    # ==========================================================
    # VISUALIZATION & PLOTS (ORIGINAL SCALE - USD)
    # ==========================================================
    # Plot 1: Actual vs Predictions (Individual seeds, Top-K Ensemble, and Uncertainty band)
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv, label='Actual BTC Price (USD)', color='black', linewidth=1.8, alpha=0.85)

    for i, seed in enumerate(seed_list):
        if seed in top_seed_ids:
            plt.plot(individual_predictions[i], label=f'Seed {seed} [Top-{TOP_K}] (R2={seed_metrics[i]["r2"]:.3f}, DA={seed_metrics[i]["da"]:.1f}%)', linestyle='--', alpha=0.65, linewidth=1.1)
        else:
            plt.plot(individual_predictions[i], label=f'Seed {seed} [Filtered] (R2={seed_metrics[i]["r2"]:.3f}, DA={seed_metrics[i]["da"]:.1f}%)', linestyle=':', color='gray', alpha=0.35, linewidth=0.9)

    plt.plot(ensemble_pred, label=f'Top-{TOP_K} Ensemble Mean (R2={ensemble_r2:.3f}, DA={ensemble_da:.1f}%)', color='crimson', linewidth=2.0)
    plt.fill_between(
        range(len(y_test_inv)),
        ensemble_pred - ensemble_std,
        ensemble_pred + ensemble_std,
        color='crimson',
        alpha=0.15,
        label=f'Top-{TOP_K} Uncertainty (±1 Std)'
    )
    plt.title(f"Top-{TOP_K} Filtered TCN Ensemble (Seeds {top_seed_ids}): Actual vs Predicted BTC Price (USD)\n[Ensemble R2: {ensemble_r2:.4f} | Dir Acc: {ensemble_da:.2f}%]", fontsize=12, fontweight='bold')
    plt.xlabel("Sample Index (Test Set)")
    plt.ylabel("BTC Price (USD)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/plots/ensemble_actual_vs_pred.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Residual Analysis (USD) for Top-K Ensemble
    residuals = y_test_inv - ensemble_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(residuals, color='royalblue', alpha=0.75, linewidth=1.0)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax1.set_title(f"Top-{TOP_K} Ensemble Residuals over Test Horizon (USD)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Error in USD (Actual - Predicted)")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.2)
    ax2.set_title(f"Residual Distribution (Mean: ${np.mean(residuals):,.2f}, Std: ${np.std(residuals):,.2f})", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Error in USD")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/plots/ensemble_residuals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Validation Loss Across Seeds
    plt.figure(figsize=(12, 5))
    for seed in seed_list:
        hist = seed_histories.get(seed, {})
        val_loss = hist.get('val_loss', [])
        if len(val_loss) > 0:
            tag = f"[Top-{TOP_K}]" if seed in top_seed_ids else "[Filtered Out]"
            plt.plot(val_loss, label=f"Seed {seed} {tag} (min={min(val_loss):.5f})", alpha=0.8, linewidth=1.4)

    plt.title(f"TCN Multi-Seed Validation Loss (MSE) Across Epochs (Top-{TOP_K} Selected vs Filtered)", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/plots/seed_training_losses.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("All evaluation plots saved to output/plots/ (ensemble_actual_vs_pred.png, ensemble_residuals.png, seed_training_losses.png)")

def run_orchestrator(script_path):
    print("==========================================================")
    print(" HYBRID GWO-WOA ORCHESTRATOR: Process Isolation Enabled")
    print(f" Target: {n_iterations} Iterations, {n_agents} Agents")
    print(" Model : Hybrid TCN-GRU (Next-Day Directional Classification)")
    print(" Process is terminated after each iteration to free 100% RAM.")
    print("==========================================================")

    while True:
        checkpoint = load_gwo_woa_checkpoint()
        if checkpoint is not None:
            iter_count = checkpoint["iter_count"]
            agent_idx = checkpoint["agent_idx"]
            if iter_count >= n_iterations and agent_idx == -1:
                print(f"\n[Orchestrator] All {n_iterations} GWO-WOA iterations are complete!")
                break
            current_iter = iter_count + 1
            current_agent = agent_idx + 1
        else:
            current_iter = 1
            current_agent = 0

        print(f"\n[Orchestrator] Starting fresh Python process for Iteration {current_iter}/{n_iterations} (Agent {current_agent + 1}/{n_agents})...")
        cmd = [sys.executable, script_path, "--worker"]
        result = subprocess.run(cmd)

        if result.returncode == 20:
            print("\n[Orchestrator] GWO-WOA Optimization completed.")
            break
        elif result.returncode not in (0, 10):
            print(f"\n[Orchestrator] Worker exited with error code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\n[Orchestrator] Starting fresh process for Multi-Seed Ensemble training & evaluation...")
    cmd = [sys.executable, script_path, "--final-ensemble"]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n==========================================================")
        print(" ALL HYBRID TCN-GRU TRAINING AND EVALUATIONS FINISHED!")
        print("==========================================================")
    else:
        print(f"[Orchestrator] Final ensemble process failed with code {result.returncode}.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid TCN-GRU with GWO-WOA Optimization for BTC Next-Day Log Return")
    parser.add_argument("--worker", action="store_true", help="Run a single GWO-WOA iteration in a dedicated worker process")
    parser.add_argument("--single-iter", action="store_true", help="Alias for --worker: run 1 iteration and exit")
    parser.add_argument("--final-ensemble", action="store_true", help="Run 5-seed ensemble training and evaluation only")
    parser.add_argument("--no-orchestrator", action="store_true", help="Run all iterations continuously in a single process without recycling")

    args = parser.parse_args()
    script_path = os.path.abspath(__file__)

    if args.worker or args.single_iter:
        exit_code = run_gwo_woa_worker()
        sys.exit(exit_code)
    elif args.final_ensemble:
        exit_code = run_final_ensemble()
        sys.exit(exit_code)
    elif args.no_orchestrator:
        while True:
            code = run_gwo_woa_worker()
            if code == 20:
                break
        run_final_ensemble()
    else:
        run_orchestrator(script_path)

