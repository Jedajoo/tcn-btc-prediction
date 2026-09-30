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


def create_sequences(features, target, window):
    """
    Creates (samples, window, features) and corresponding 1-step-ahead target.
    """
    features = np.asarray(features)
    target = np.asarray(target)
    
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i : (i + window)])
        y.append(target[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_tcn_model(
    input_shape,
    n_filters,
    k_size=2,
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    dilations=None,
    **kwargs
):
    # Helper untuk membangun model berdasarkan parameter dari GWO-WOA / PSO
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    tcn_layer = TCN(
        nb_filters=int(n_filters),
        kernel_size=int(k_size),
        nb_stacks=1,
        dilations=dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=float(dropout),
        return_sequences=False,
        input_shape=input_shape
    )

    inputs = layers.Input(shape=input_shape)
    x = tcn_layer(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay)
        )
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='MSE'),
        metrics=['mse']
    )
    return model

def get_early_stopping(patience=20, monitor='val_los', mode='min', verbose=0):
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=verbose,
        restore_best_weights=True
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

def inverse_transform_target(data, scaler_or_path="output/scaler/target_scaler.joblib"):
    """
    Inverse transforms scaled target or prediction values back to their original scale (e.g. BTC-USD price).

    Parameters:
    -----------
    data : array-like
        Scaled target or prediction values. Supports 1D array (N,) or 2D array (N, 1) or (M, N).
    scaler_or_path : str or sklearn Scaler object
        Path to the saved scaler joblib file or an already loaded scaler instance.

    Returns:
    --------
    np.ndarray
        Inverse-transformed values in the original unscaled unit.
    """
    from joblib import load
    if isinstance(scaler_or_path, str):
        scaler = load(scaler_or_path)
    else:
        scaler = scaler_or_path

    arr = np.asarray(data)
    if arr.ndim == 1:
        return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            return scaler.inverse_transform(arr).ravel()
        else:
            return np.array([scaler.inverse_transform(row.reshape(-1, 1)).ravel() for row in arr])
    else:
        orig_shape = arr.shape
        flat = arr.reshape(-1, 1)
        inv = scaler.inverse_transform(flat)
        return inv.reshape(orig_shape)

inverse_transform = inverse_transform_target

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

def compute_directional_accuracy(y_true, y_pred, mode='mda'):
    """
    Computes Directional Accuracy (DA) in percentage (0 - 100%).

    Parameters:
    -----------
    y_true : array-like
        Actual target prices (USD).
    y_pred : array-like
        Predicted target prices (USD).
    mode : str, 'mda' or 'trend'
        - 'mda': Mean Directional Accuracy relative to previous known actual close:
                 Sign(y_true[t] - y_true[t-1]) == Sign(y_pred[t] - y_true[t-1])
        - 'trend': Predicted trajectory slope vs actual slope:
                   Sign(y_true[t] - y_true[t-1]) == Sign(y_pred[t] - y_pred[t-1])

    Returns:
    --------
    float: Percentage of correct direction predictions.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) < 2:
        return 0.0

    actual_diff = y_true[1:] - y_true[:-1]
    if mode.lower() == 'trend':
        pred_diff = y_pred[1:] - y_pred[:-1]
    else:  # default 'mda'
        pred_diff = y_pred[1:] - y_true[:-1]

    # Matching sign indicates correct direction
    correct = (np.sign(actual_diff) == np.sign(pred_diff))
    return float(np.mean(correct) * 100.0)
