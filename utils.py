import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


def build_tcn_model(input_shape, n_filters, k_size, dropout, learning_rate):
    # Helper untuk membangun model berdasarkan parameter dari PSO
    tcn_layer = TCN(
        nb_filters=int(n_filters),
        kernel_size=int(k_size),
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=dropout,
        return_sequences=False,
        input_shape=input_shape
    )

    inputs = layers.Input(shape=input_shape)
    x = tcn_layer(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError(name='MSE'))
    return model

def get_early_stopping(patience=20):
    return EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=patience,         # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

def get_reduce_lr(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, mode='min', verbose=0):
    """
    Returns ReduceLROnPlateau callback to reduce learning rate when monitored metric plateaus.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes comprehensive binary classification metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    loss = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    cm = confusion_matrix(y_true, y_pred)

    confidence = np.abs(y_prob - threshold) * 2.0
    high_conf_mask = np.abs(y_prob - threshold) >= 0.04
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
        high_conf_coverage = np.mean(high_conf_mask) * 100.0
    else:
        high_conf_acc = acc
        high_conf_coverage = 0.0

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_loss": loss,
        "confusion_matrix": cm,
        "confidence_mean": np.mean(confidence) * 100.0,
        "high_conf_acc": high_conf_acc,
        "high_conf_coverage": high_conf_coverage
    }

def find_optimal_threshold(y_true, y_prob):
    """
    Finds the optimal classification threshold using Youden's J-statistic on the ROC curve.
    """
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.50
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thresh = float(thresholds[optimal_idx])
    return float(np.clip(optimal_thresh, 0.35, 0.65))

def backtest_directional_strategy(actual_returns, y_prob, threshold=0.5, fee=0.0005):
    """
    Simulates a long-or-cash trading strategy based on directional probability.
    """
    actual_returns = np.asarray(actual_returns).ravel()
    y_prob = np.asarray(y_prob).ravel()

    signals = (y_prob >= threshold).astype(int)
    actual_simple_returns = np.expm1(actual_returns) if np.all(np.abs(actual_returns) < 1.0) else actual_returns
    strategy_returns = signals * actual_simple_returns

    position_changes = np.abs(np.diff(signals, prepend=signals[0]))
    strategy_returns -= position_changes * fee

    cum_market = np.cumprod(1.0 + actual_simple_returns) - 1.0
    cum_strategy = np.cumprod(1.0 + strategy_returns) - 1.0

    mean_strat = np.mean(strategy_returns)
    std_strat = np.std(strategy_returns) + 1e-9
    sharpe = (mean_strat / std_strat) * np.sqrt(365)

    return {
        "cumulative_market": cum_market,
        "cumulative_strategy": cum_strategy,
        "total_strategy_return": cum_strategy[-1] * 100.0 if len(cum_strategy) > 0 else 0.0,
        "total_market_return": cum_market[-1] * 100.0 if len(cum_market) > 0 else 0.0,
        "sharpe_ratio": sharpe,
        "signals": signals
    }
