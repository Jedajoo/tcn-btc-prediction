import os
import sys
import gc
import pickle
import random
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

try:
    import keras
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

# Global Constants & Hyperparameter Constraints
ticker = "BTC-USD"
FIXED_TIME_WINDOW = 90
FIXED_KERNEL_SIZE = 2
FIXED_BATCH_SIZE = 32
DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32]

# Compute theoretical receptive field
# Formula: 1 + 2 * (k - 1) * sum(dilations)
RECEPTIVE_FIELD = 1 + 2 * (FIXED_KERNEL_SIZE - 1) * 1 * sum(DEFAULT_DILATIONS)

print(f"--- Hybrid TCN-GRU & GWO-WOA Configuration ---")
print(f"  Target         : Next-Day Direction (Binary Classification: Up/Down)")
print(f"  Kernel Size    : {FIXED_KERNEL_SIZE}")
print(f"  Time Window    : {FIXED_TIME_WINDOW} days (3 Months)")
print(f"  Batch Size     : {FIXED_BATCH_SIZE}")
print(f"  Dilations      : {DEFAULT_DILATIONS}")
print(f"  Receptive Field: {RECEPTIVE_FIELD}")
print(f"----------------------------------------------")

# Boundaries for the 5 search parameters:
# [0] n_filters       : [32, 256]  (TCN Conv Filters)
# [1] gru_units       : [32, 256]  (GRU Hidden Units)
# [2] dropout         : [0.05, 0.40]
# [3] log_lr          : [-4.5, -2.5] -> 10^log_lr
# [4] log_weight_decay: [-5.0, -2.0] -> 10^log_wd
boundaries = [
    (32, 256),     # [0] n_filters
    (32, 256),     # [1] gru_units
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
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


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


def decode_hyperparameters(p):
    """
    Decodes 5-dimensional continuous GWO-WOA position into concrete hyperparameters.
    """
    # 0. n_filters (multiple of 16 in [32, 256])
    n_filters = int(16 * round(p[0] / 16.0))
    n_filters = max(32, min(256, n_filters))

    # 1. gru_units (multiple of 16 in [32, 256])
    gru_units = int(16 * round(p[1] / 16.0))
    gru_units = max(32, min(256, gru_units))

    # 2. dropout in [0.05, 0.40]
    dropout = float(np.clip(p[2], 0.05, 0.40))

    # 3. learning_rate
    log_lr = float(np.clip(p[3], -4.5, -2.5))
    learning_rate = 10.0 ** log_lr

    # 4. weight_decay
    log_wd = float(np.clip(p[4], -5.0, -2.0))
    weight_decay = 10.0 ** log_wd

    return {
        "n_filters": n_filters,
        "gru_units": gru_units,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "kernel_size": FIXED_KERNEL_SIZE,
        "time_window": FIXED_TIME_WINDOW,
        "receptive_field": RECEPTIVE_FIELD,
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


def load_tabular_data():
    train_tab = np.load("output/data/train_tabular.npz")
    test_tab = np.load("output/data/test_tabular.npz")
    return train_tab, test_tab


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

    model = ut.build_hybrid_tcn_gru_model(
        input_shape=(hp["time_window"], n_features),
        n_filters=hp["n_filters"],
        gru_units=hp["gru_units"],
        k_size=hp["kernel_size"],
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
    best_val_acc = float(max(history.history.get('val_accuracy', [0.0])))
    best_val_auc = float(max(history.history.get('val_auc', [0.0])))
    fitness = best_val_loss

    # Strict memory cleanup
    del model
    del history
    tf.keras.backend.clear_session()
    gc.collect()

    print(
        f"  [Eval] Filters={hp['n_filters']}, GRU={hp['gru_units']}, "
        f"Window={hp['time_window']} (RF={hp['receptive_field']}), "
        f"Drop={hp['dropout']:.3f}, LR={hp['learning_rate']:.5f}, "
        f"Batch={hp['batch_size']}, WD={hp['weight_decay']:.5f} | "
        f"ValLoss={best_val_loss:.5f}, ValAcc={best_val_acc*100:.2f}%, ValAUC={best_val_auc:.4f} | Fitness={fitness:.5f}"
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
    """
    Trains 5 multi-seed Hybrid TCN-GRU models, evaluates ensemble regression & financial backtest,
    and saves plots.
    """
    import tensorflow as tf
    import utils as ut

    init_gpu()
    train_tab, test_tab = load_tabular_data()
    train_features = train_tab["features"]
    train_target = train_tab["target"]  # Next-day directional binary target (0 or 1)
    train_prices = train_tab["prices"]
    test_features = test_tab["features"]
    test_target = test_tab["target"]
    test_prices = test_tab["prices"]
    n_features = train_features.shape[1]

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

    print("\n==========================================")
    print("5. MULTI-SEED HYBRID TCN-GRU ENSEMBLE TRAINING (5 SEEDS)")
    print("==========================================")
    print(
        f"Using Optimal Parameters (Window={best_hp['time_window']}, Filters={best_hp['n_filters']}, "
        f"GRU={best_hp['gru_units']}, K={best_hp['kernel_size']})..."
    )

    best_window = best_hp["time_window"]

    X_train_final, y_train_final = ut.create_sequences(train_features, train_target, best_window)
    val_size = max(1, int(len(X_train_final) * 0.20))
    X_tr_final, y_tr_final = X_train_final[:-val_size], y_train_final[:-val_size]
    X_val_final, y_val_final = X_train_final[-val_size:], y_train_final[-val_size:]

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

    seed_list = [42, 43, 44, 45, 46]
    individual_predictions = []
    val_individual_predictions = []
    seed_histories = []
    seed_val_rmses = []

    plt.figure(figsize=(10, 5))

    for seed in seed_list:
        seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
        seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

        if os.path.exists(seed_model_path) and os.path.exists(seed_history_path):
            print(f"Loading existing trained model for Seed {seed}...")
            model = tf.keras.models.load_model(seed_model_path, compile=False, safe_mode=False)
            with open(seed_history_path, "rb") as f:
                history = pickle.load(f)
        else:
            print(f"\n--- Training Hybrid TCN-GRU with Seed {seed} ---")
            set_seed(seed)
            tf.keras.backend.clear_session()

            model = ut.build_hybrid_tcn_gru_model(
                input_shape=(best_window, n_features),
                n_filters=best_hp["n_filters"],
                gru_units=best_hp["gru_units"],
                k_size=best_hp["kernel_size"],
                dropout=best_hp["dropout"],
                learning_rate=best_hp["learning_rate"],
                weight_decay=best_hp["weight_decay"],
                dilations=best_hp.get("dilations", DEFAULT_DILATIONS)
            )

            early_stopping = ut.get_early_stopping(patience=20, monitor='val_loss', mode='min', verbose=1)
            reduce_lr = ut.get_reduce_lr(monitor='val_loss', factor=0.5, patience=10, mode='min', verbose=0)

            history_obj = model.fit(
                X_tr_final,
                y_tr_final,
                validation_data=(X_val_final, y_val_final),
                epochs=200,
                batch_size=best_hp["batch_size"],
                shuffle=False,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            history = history_obj.history
            model.save(seed_model_path)
            with open(seed_history_path, "wb") as f:
                pickle.dump(history, f)

        # Predict on validation and test sets (probabilities in [0, 1])
        val_pred = model.predict(X_val_final, verbose=0).ravel()
        val_individual_predictions.append(val_pred)

        pred = model.predict(X_test_final, verbose=0).ravel()
        individual_predictions.append(pred)

        seed_metrics = ut.compute_classification_metrics(y_test_final, pred, threshold=0.50)
        print(f"Seed {seed}: Acc={seed_metrics['accuracy']*100:.2f}%, AUC={seed_metrics['auc']:.4f}, F1={seed_metrics['f1']:.4f}, Loss={seed_metrics['log_loss']:.5f}")

        seed_histories.append(history)
        min_val_loss = min(history.get('val_loss', [999.0]))
        seed_val_rmses.append(min_val_loss)

        plt.plot(history.get('val_loss', []), label=f"Seed {seed} Val Loss (min={min_val_loss:.5f})")

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    plt.title("Hybrid TCN-GRU Multi-Seed Validation Binary Crossentropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/plots/training_losses.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Ensemble Evaluation
    print("\n--- Generating Multi-Seed Ensemble Predictions ---")
    ensemble_predictions = np.mean(individual_predictions, axis=0)
    val_ensemble_probs = np.mean(val_individual_predictions, axis=0)
    optimal_thresh = ut.find_optimal_threshold(y_val_final, val_ensemble_probs)
    print(f"Calibrated Optimal Threshold (Youden's J): {optimal_thresh:.4f}")
    best_hp["optimal_threshold"] = optimal_thresh

    ensemble_metrics = ut.compute_classification_metrics(y_test_final, ensemble_predictions, threshold=optimal_thresh)

    test_eval_returns = test_tab["returns"][-len(y_test_final):]
    test_eval_prices = test_prices[-len(y_test_final):]
    backtest_res = ut.backtest_directional_strategy(test_eval_returns, ensemble_predictions, threshold=optimal_thresh)

    print("\n==========================================")
    print("FINAL HYBRID TCN-GRU ENSEMBLE DIRECTIONAL PERFORMANCE")
    print("==========================================")
    print(f"Calibrated Threshold       : {optimal_thresh:.4f}")
    print(f"Directional Accuracy (DA)  : {ensemble_metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC Score              : {ensemble_metrics['auc']:.4f}")
    print(f"Precision (Up Class)       : {ensemble_metrics['precision']:.4f}")
    print(f"Recall (Up Class)          : {ensemble_metrics['recall']:.4f}")
    print(f"F1 Score                   : {ensemble_metrics['f1']:.4f}")
    print(f"Brier Score Loss           : {ensemble_metrics['brier_score']:.5f}")
    print(f"Log Loss (BCE)             : {ensemble_metrics['log_loss']:.5f}")
    print(f"High Confidence Accuracy   : {ensemble_metrics['high_conf_acc']*100:.2f}% (Coverage: {ensemble_metrics['high_conf_coverage']:.1f}%)")
    print(f"Strategy Cumulative Return : {backtest_res['total_strategy_return']:.2f}% (vs Market Buy&Hold: {backtest_res['total_market_return']:.2f}%)")
    print(f"Strategy Sharpe Ratio      : {backtest_res['sharpe_ratio']:.2f}")
    print(f"Confusion Matrix: \n{ensemble_metrics['confusion_matrix']}")
    print("==========================================\n")

    evaluation_summary = {
        "ensemble_metrics": ensemble_metrics,
        "backtest_results": backtest_res,
        "best_hyperparameters": best_hp,
        "seed_val_losses": seed_val_rmses,
        "optimal_threshold": optimal_thresh
    }
    dump(evaluation_summary, "output/model/ensemble_evaluation.joblib")
    dump(best_hp, "output/model/best_hyperparameters.joblib")

    # 7. Generate Classification & Financial Plots
    # A. Actual Direction vs Predicted Up Probability
    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(y_test_final)), y_test_final, label='Actual Direction (1=Up, 0=Down)', color='black', alpha=0.3, s=15)
    plt.plot(ensemble_predictions, label=f'Hybrid TCN-GRU P(Up) (AUC={ensemble_metrics["auc"]:.4f})', color='royalblue', lw=1.5)
    plt.axhline(optimal_thresh, color='red', linestyle='--', alpha=0.7, label=f'Calibrated Threshold ({optimal_thresh:.3f})')
    plt.title(f'{ticker} Next-Day Direction: Actual vs Hybrid TCN-GRU Ensemble Probability')
    plt.xlabel('Test Sample Days')
    plt.ylabel('P(Up)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/plots/actual_vs_predicted_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # B. Probability Distribution & Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(ensemble_predictions[y_test_final == 1], bins=25, alpha=0.6, color='forestgreen', label='Actual Up (1)')
    ax1.hist(ensemble_predictions[y_test_final == 0], bins=25, alpha=0.6, color='crimson', label='Actual Down (0)')
    ax1.axvline(optimal_thresh, color='blue', linestyle='--', label=f'Threshold ({optimal_thresh:.3f})')
    ax1.set_title('Predicted Probability Distribution by Class')
    ax1.set_xlabel('Predicted Probability P(Up)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cm = ensemble_metrics['confusion_matrix']
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Directional Confusion Matrix')
    plt.colorbar(im, ax=ax2)
    classes = ['Down (0)', 'Up (1)']
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(classes)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")
    ax2.set_ylabel('Actual Label')
    ax2.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('output/plots/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # C. Backtest vs Price
    plt.figure(figsize=(14, 7))
    ax_top = plt.subplot(2, 1, 1)
    ax_top.plot(test_eval_prices, label=f'{ticker} Actual Close Price', color='black', alpha=0.8)
    pred_binary = (ensemble_predictions >= optimal_thresh).astype(int)
    long_signals = np.where(pred_binary == 1)[0]
    cash_signals = np.where(pred_binary == 0)[0]
    ax_top.scatter(long_signals, test_eval_prices[long_signals], color='green', marker='^', s=25, label=f'Signal Long (P >= {optimal_thresh:.2f})', alpha=0.7)
    ax_top.scatter(cash_signals, test_eval_prices[cash_signals], color='red', marker='v', s=25, label=f'Signal Cash (P < {optimal_thresh:.2f})', alpha=0.7)
    ax_top.set_title(f'Hybrid TCN-GRU Predicted Direction Signals vs {ticker} Price')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, alpha=0.3)

    ax_bot = plt.subplot(2, 1, 2)
    ax_bot.plot(backtest_res['cumulative_strategy'] * 100, label=f'Hybrid TCN-GRU Strategy ({backtest_res["total_strategy_return"]:.1f}%)', color='green', lw=1.8)
    ax_bot.plot(backtest_res['cumulative_market'] * 100, label=f'Buy & Hold Benchmark ({backtest_res["total_market_return"]:.1f}%)', color='gray', linestyle='--', lw=1.5)
    ax_bot.set_title('Cumulative Return Backtest (%)')
    ax_bot.set_xlabel('Test Sample Days')
    ax_bot.set_ylabel('Cumulative Return (%)')
    ax_bot.legend(loc='upper left')
    ax_bot.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/plots/trading_backtest.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All evaluation plots successfully saved in output/plots/")
    return 0


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