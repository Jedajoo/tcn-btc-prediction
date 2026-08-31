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

# Global Constants & Hyperparameter Constraints
ticker = "BTC-USD"
ALLOWED_WINDOWS = [64, 128, 256]
ALLOWED_KERNELS = [2, 3, 5]
FIXED_BATCH_SIZE = 32
DEFAULT_DILATIONS = [1, 2, 4, 8, 16]

# Build all valid pairs (k, w) where w >= receptive_field(k)
VALID_KW_PAIRS = []
for w in ALLOWED_WINDOWS:
    for k in ALLOWED_KERNELS:
        # Receptive field formula: 1 + 2 * (k - 1) * sum(dilations)
        rf = 1 + 2 * (int(k) - 1) * 1 * sum(DEFAULT_DILATIONS)
        if w >= rf:
            VALID_KW_PAIRS.append({
                "kernel_size": k,
                "time_window": w,
                "receptive_field": rf,
                "dilations": DEFAULT_DILATIONS
            })

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

# PSO Hyperparameters
n_particles = 20
n_iterations = 30
c1 = 1.8
c2 = 2.2

# Checkpoint paths
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
PSO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pso_state.pkl")
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
    precomputed = {}
    for w in ALLOWED_WINDOWS:
        X_all, y_all = ut.create_sequences(train_features, train_target, w)
        val_size = max(1, int(len(X_all) * 0.20))
        precomputed[w] = {
            "X_tr": X_all[:-val_size],
            "y_tr": y_all[:-val_size],
            "X_val": X_all[-val_size:],
            "y_val": y_all[-val_size:],
            "n_samples": len(X_all)
        }
    return precomputed


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

    model = ut.build_tcn_attention_model(
        input_shape=(hp["time_window"], n_features),
        n_filters=hp["n_filters"],
        k_size=hp["kernel_size"],
        dropout=hp["dropout"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        dilations=hp["dilations"]
    )

    early_stop = ut.get_early_stopping(patience=100, monitor='val_auc', mode='max', verbose=0)
    reduce_lr = ut.get_reduce_lr(monitor='val_auc', factor=0.5, patience=10, mode='max', verbose=0)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=hp["batch_size"],
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    best_val_auc = float(max(history.history.get('val_auc', [0.5])))
    best_val_loss = float(min(history.history.get('val_loss', [0.693])))
    fitness = float(1.0 - best_val_auc)

    # Strict memory cleanup to avoid C++ heap accumulation
    del model
    del history
    tf.keras.backend.clear_session()
    gc.collect()

    print(
        f"  [Eval] Filters={hp['n_filters']}, K={hp['kernel_size']}, "
        f"Window={hp['time_window']} (RF={hp['receptive_field']}), "
        f"Drop={hp['dropout']:.3f}, LR={hp['learning_rate']:.5f}, "
        f"Batch={hp['batch_size']}, WD={hp['weight_decay']:.5f} | "
        f"ValAUC={best_val_auc:.4f}, ValLoss={best_val_loss:.4f} | Fitness={fitness:.4f}"
    )

    return fitness


def run_pso_worker():
    """
    Executes exactly 1 PSO iteration (or remaining particles of an interrupted iteration),
    updates velocities & positions, saves checkpoint, and exits.
    Returns:
        10: Iteration completed, more iterations left.
        20: All iterations completed.
    """
    init_gpu()
    train_tab, test_tab = load_tabular_data()
    train_features = train_tab["features"]
    train_target = train_tab["target"]
    n_features = train_features.shape[1]

    precomputed_data = precache_sequence_data(train_features, train_target)

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

    # Check if already complete
    if start_iter >= n_iterations and checkpoint is not None and checkpoint.get("particle_idx") == -1:
        print(f"PSO Optimization is already complete ({start_iter}/{n_iterations} iterations).")
        best_hp = decode_hyperparameters(gbest_position)
        dump(best_hp, "output/model/best_hyperparameters.joblib")
        return 20

    iter_count = start_iter
    w = 0.9 - (0.5 * iter_count / n_iterations)
    print(f"\n--- [Process Worker] PSO Iteration {iter_count + 1}/{n_iterations} (Inertia w={w:.3f}) ---")

    for i in range(start_particle, n_particles):
        particle = particles[i]
        print(f"Evaluating Particle {i + 1}/{n_particles}:")

        current_val = objective_function(
            particle['position'],
            precomputed_data,
            n_features
        )

        if current_val < particle['pbest_value']:
            particle['pbest_value'] = current_val
            particle['pbest_position'] = particle['position'].copy()

        if current_val < gbest_value:
            gbest_value = current_val
            gbest_position = particle['position'].copy()
            print(f"  >>> New Global Best Fitness: {gbest_value:.5f} (Val AUC: {1.0 - gbest_value:.4f})")

        save_pso_checkpoint(iter_count, i, particles, gbest_position, gbest_value)

    # Velocity and position updates for all particles
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

    next_iter = iter_count + 1
    save_pso_checkpoint(next_iter, -1, particles, gbest_position, gbest_value)

    print(f"--- [Process Worker] Iteration {iter_count + 1} completed and saved to checkpoint ---")

    if next_iter >= n_iterations:
        print("\n==========================================")
        print("PSO OPTIMIZATION COMPLETED!")
        best_hp = decode_hyperparameters(gbest_position)
        print(f"Best Fitness Value (1 - AUC) : {gbest_value:.5f} (AUC: {1.0 - gbest_value:.4f})")
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
    Trains 5 multi-seed models, evaluates the ensemble, generates backtest metrics, and saves plots.
    """
    import tensorflow as tf
    import utils as ut
    from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

    init_gpu()
    train_tab, test_tab = load_tabular_data()
    train_features = train_tab["features"]
    train_target = train_tab["target"]
    train_prices = train_tab["prices"]
    train_returns = train_tab["returns"]
    test_features = test_tab["features"]
    test_target = test_tab["target"]
    test_prices = test_tab["prices"]
    test_returns = test_tab["returns"]
    n_features = train_features.shape[1]

    hp_path = "output/model/best_hyperparameters.joblib"
    if os.path.exists(hp_path):
        best_hp = load(hp_path)
    else:
        cp = load_pso_checkpoint()
        if cp and cp.get("gbest_position") is not None:
            best_hp = decode_hyperparameters(cp["gbest_position"])
            dump(best_hp, hp_path)
        else:
            raise FileNotFoundError("Could not find best_hyperparameters.joblib or valid PSO checkpoint.")

    print("\n==========================================")
    print("5. MULTI-SEED ENSEMBLE TRAINING (5 SEEDS)")
    print("==========================================")
    print(f"Using Optimal Parameters (Window={best_hp['time_window']}, Filters={best_hp['n_filters']}, K={best_hp['kernel_size']})...")

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

    seed_list = [42, 43, 44, 45, 46]
    individual_predictions = []
    seed_histories = []
    seed_val_aucs = []

    plt.figure(figsize=(10, 5))

    for seed in seed_list:
        seed_model_path = os.path.join(SEED_CHECKPOINT_DIR, f"model_seed_{seed}.keras")
        seed_history_path = os.path.join(SEED_CHECKPOINT_DIR, f"history_seed_{seed}.pkl")

        if os.path.exists(seed_model_path) and os.path.exists(seed_history_path):
            print(f"Loading existing trained model for Seed {seed}...")
            model = tf.keras.models.load_model(seed_model_path, compile=False)
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

            early_stopping = ut.get_early_stopping(patience=200, monitor='val_auc', mode='max', verbose=1)
            reduce_lr = ut.get_reduce_lr(monitor='val_auc', factor=0.5, patience=15, mode='max', verbose=0)

            history_obj = model.fit(
                X_train_final,
                y_train_final,
                epochs=600,
                batch_size=best_hp["batch_size"],
                validation_split=0.2,
                shuffle=False,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            history = history_obj.history
            model.save(seed_model_path)
            with open(seed_history_path, "wb") as f:
                pickle.dump(history, f)

        # Immediately predict on test set and free model memory
        prob_pred = model.predict(X_test_final, verbose=0).ravel()
        individual_predictions.append(prob_pred)
        seed_metrics = ut.compute_classification_metrics(y_test_final, prob_pred)
        print(f"Seed {seed}: Acc={seed_metrics['accuracy']*100:.2f}%, AUC={seed_metrics['auc']:.4f}, LogLoss={seed_metrics['log_loss']:.4f}")

        seed_histories.append(history)
        max_val_auc = max(history.get('val_auc', [0.5]))
        seed_val_aucs.append(max_val_auc)

        plt.plot(history.get('val_auc', []), label=f"Seed {seed} Val AUC (max={max_val_auc:.4f})")

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    plt.title("TCN + Multi-Head Attention Multi-Seed Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC-AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/plots/training_losses.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Ensemble Evaluation
    print("\n--- Generating Multi-Seed Ensemble Predictions ---")
    ensemble_probabilities = np.mean(individual_predictions, axis=0)
    ensemble_metrics = ut.compute_classification_metrics(y_test_final, ensemble_probabilities)

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

    evaluation_summary = {
        "ensemble_metrics": ensemble_metrics,
        "backtest_results": backtest_res,
        "best_hyperparameters": best_hp,
        "seed_val_aucs": seed_val_aucs
    }
    dump(evaluation_summary, "output/model/ensemble_evaluation.joblib")

    # 7. Generate Plots
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

    # D. Backtest vs Price
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

    print("All evaluation plots successfully saved in output/plots/")
    return 0


def run_orchestrator(script_path):
    print("==========================================================")
    print(" PSO-TCN ORCHESTRATOR: Process Isolation Enabled")
    print(f" Target: {n_iterations} Iterations, {n_particles} Particles")
    print(" Process is terminated after each iteration to free 100% RAM.")
    print("==========================================================")

    while True:
        checkpoint = load_pso_checkpoint()
        if checkpoint is not None:
            iter_count = checkpoint["iter_count"]
            particle_idx = checkpoint["particle_idx"]
            if iter_count >= n_iterations and particle_idx == -1:
                print(f"\n[Orchestrator] All {n_iterations} PSO iterations are complete!")
                break
            current_iter = iter_count + 1
            current_particle = particle_idx + 1
        else:
            current_iter = 1
            current_particle = 0

        print(f"\n[Orchestrator] Starting fresh Python process for Iteration {current_iter}/{n_iterations} (Particle {current_particle + 1}/{n_particles})...")
        cmd = [sys.executable, script_path, "--worker"]
        result = subprocess.run(cmd)

        if result.returncode == 20:
            print("\n[Orchestrator] PSO Optimization completed.")
            break
        elif result.returncode not in (0, 10):
            print(f"\n[Orchestrator] Worker exited with error code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\n[Orchestrator] Starting fresh process for Multi-Seed Ensemble training & evaluation...")
    cmd = [sys.executable, script_path, "--final-ensemble"]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n==========================================================")
        print(" ALL PSO-TCN TRAINING AND EVALUATIONS FINISHED!")
        print("==========================================================")
    else:
        print(f"[Orchestrator] Final ensemble process failed with code {result.returncode}.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO-TCN Directional Prediction for Bitcoin")
    parser.add_argument("--worker", action="store_true", help="Run a single PSO iteration in a dedicated worker process")
    parser.add_argument("--single-iter", action="store_true", help="Alias for --worker: run 1 iteration and exit")
    parser.add_argument("--final-ensemble", action="store_true", help="Run 5-seed ensemble training and evaluation only")
    parser.add_argument("--no-orchestrator", action="store_true", help="Run all iterations continuously in a single process without recycling")

    args = parser.parse_args()
    script_path = os.path.abspath(__file__)

    if args.worker or args.single_iter:
        exit_code = run_pso_worker()
        sys.exit(exit_code)
    elif args.final_ensemble:
        exit_code = run_final_ensemble()
        sys.exit(exit_code)
    elif args.no_orchestrator:
        while True:
            code = run_pso_worker()
            if code == 20:
                break
        run_final_ensemble()
    else:
        run_orchestrator(script_path)