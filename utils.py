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
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

def create_sequences(features, target, window):
    """
    Creates (samples, window, features) and corresponding 1-step-ahead target.
    """
    features = np.asarray(features)
    target = np.asarray(target)
    
    X, y = [], []
    for i in range(len(features) - window + 1):
        X.append(features[i : (i + window)])
        y.append(target[i + window - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def compute_receptive_field(k_size, dilations=(1, 2, 4, 8, 16, 32), nb_stacks=1):
    """
    Computes theoretical receptive field for a 1D causal TCN:
    RF = 1 + 2 * (k - 1) * nb_stacks * sum(dilations)
    """
    return 1 + 2 * (int(k_size) - 1) * int(nb_stacks) * sum(dilations)

def get_focal_loss(gamma=2.0, alpha=0.50, label_smoothing=0.05):
    """
    Returns Symmetric Binary Focal Loss with Label Smoothing.
    """
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - 2.0 * label_smoothing) + label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.math.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

    return focal_loss

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

def build_hybrid_tcn_gru_model(
    input_shape,
    n_filters,
    gru_units,
    k_size=2,
    dropout=0.20,
    learning_rate=1e-3,
    weight_decay=1e-4,
    dilations=None
):
    """
    Builds a Hybrid TCN-GRU Regression Model for continuous next-day log return prediction.
    Architecture:
    1. Input Sequence (window, n_features)
    2. Causal Dilated TCN layer for multi-scale temporal receptive field
    3. GRU layer for recurrent sequential temporal dynamics
    4. Multi-Head Temporal Self-Attention + Residual Connection + LayerNorm
    5. Dual Temporal Pooling (last causal timestep + global average pooling)
    6. Deep Regression Head with GELU, BatchNorm, and Dropout
    7. Linear output predicting 1-step ahead continuous log return
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]

    inputs = layers.Input(shape=input_shape, name="input_sequence")

    # 1. Temporal Convolutional Network (TCN)
    tcn_out = TCN(
        nb_filters=int(n_filters),
        kernel_size=int(k_size),
        nb_stacks=1,
        dilations=dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=float(dropout),
        return_sequences=True,
        name="tcn_layer"
    )(inputs)

    # 2. Gated Recurrent Unit (GRU)
    gru_out = layers.GRU(
        units=int(gru_units),
        return_sequences=True,
        dropout=float(dropout),
        name="gru_layer"
    )(tcn_out)

    # 3. Multi-Head Temporal Self-Attention over sequential representations
    num_heads = 2
    key_dim = max(16, int(gru_units // num_heads))
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="temporal_mha"
    )(gru_out, gru_out)

    # Residual Connection + Layer Normalization
    x = layers.Add(name="attn_residual")([gru_out, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # 4. Dual Temporal Pooling: Last causal timestep + Global Average Pooling
    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_step_extraction")(x)
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    pooled = layers.Concatenate(name="dual_temporal_pooled")([last_step, avg_pool])

    # 5. Deep Regression Head
    h = layers.Dense(128, activation='gelu', name="dense_head_1")(pooled)
    h = layers.BatchNormalization(name="bn_1")(h)
    h = layers.Dropout(float(dropout), name="dropout_1")(h)

    h = layers.Dense(64, activation='gelu', name="dense_head_2")(h)
    h = layers.BatchNormalization(name="bn_2")(h)
    h = layers.Dropout(float(dropout), name="dropout_2")(h)

    h = layers.Dense(32, activation='gelu', name="dense_head_3")(h)
    h = layers.Dropout(float(dropout), name="dropout_3")(h)

    # 6. Scaled Tanh Output Layer (Bounded to +/- 4% daily return with zero-centric initialization)
    h_out = layers.Dense(
        1,
        activation='tanh',
        kernel_initializer='zeros',
        bias_initializer='zeros',
        name="tanh_core"
    )(h)
    outputs = layers.Lambda(lambda t: t * 0.04, name="predicted_log_return")(h_out)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_TCN_GRU_Regressor")

    # Optimizer with decoupled weight decay (AdamW)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay)
    )

    # Huber loss tuned for crypto return scale (delta=0.01 protects against flash crashes)
    loss_fn = tf.keras.losses.MeanSquaredError(name='mse')
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'mae',
            'mse',
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )

    return model

def build_tcn_attention_model(
    input_shape,
    n_filters,
    k_size,
    dropout,
    learning_rate,
    weight_decay=1e-4,
    dilations=None
):
    """
    Builds a TCN model with Multi-Head Attention for classification (backward compatibility).
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    
    tcn_out = TCN(
        nb_filters=int(n_filters),
        kernel_size=int(k_size),
        nb_stacks=1,
        dilations=dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=float(dropout),
        return_sequences=True,
        name="tcn_layer"
    )(inputs)

    num_heads = 2
    key_dim = max(16, int(n_filters // num_heads))
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="temporal_mha"
    )(tcn_out, tcn_out)

    x = layers.Add(name="attn_residual")([tcn_out, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_step_extraction")(x)
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    pooled = layers.Concatenate(name="dual_temporal_pooled")([last_step, avg_pool])

    h = layers.Dense(128, activation='gelu', name="dense_head_1")(pooled)
    h = layers.BatchNormalization(name="bn_1")(h)
    h = layers.Dropout(float(dropout), name="dropout_1")(h)

    h = layers.Dense(64, activation='gelu', name="dense_head_2")(h)
    h = layers.BatchNormalization(name="bn_2")(h)
    h = layers.Dropout(float(dropout), name="dropout_2")(h)

    h = layers.Dense(32, activation='gelu', name="dense_head_3")(h)
    h = layers.Dropout(float(dropout), name="dropout_3")(h)

    outputs = layers.Dense(1, activation='sigmoid', name="direction_probability")(h)

    model = Model(inputs=inputs, outputs=outputs, name="PSO_TCN_Attention_Classifier")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay)
    )

    loss_fn = get_focal_loss(gamma=2.0, alpha=0.50, label_smoothing=0.05)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def get_early_stopping(patience=20, monitor='val_loss', mode='min', verbose=0):
    """
    Returns EarlyStopping callback with weight restoration.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=True,
        verbose=verbose
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

def compute_regression_metrics(y_true, y_pred):
    """
    Computes comprehensive continuous return regression and directional metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # R2 Score (coefficient of determination)
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    # Directional Accuracy: Percentage where sign(predicted) matches sign(actual)
    actual_direction = (y_true > 0).astype(int)
    predicted_direction = (y_pred > 0).astype(int)
    da = float(accuracy_score(actual_direction, predicted_direction)) * 100.0

    # Pearson Correlation between actual and predicted returns
    if np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "explained_variance": evs,
        "directional_accuracy": da,
        "pearson_corr": corr
    }

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
    brier = brier_score_loss(y_true, y_prob)
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
        "brier_score": brier,
        "log_loss": loss,
        "confusion_matrix": cm,
        "confidence_mean": np.mean(confidence) * 100.0,
        "high_conf_acc": high_conf_acc,
        "high_conf_coverage": high_conf_coverage
    }

def backtest_return_strategy(actual_returns, predicted_returns, threshold=0.0, fee=0.0005):
    """
    Simulates a long-or-cash quantitative trading strategy based on continuous predicted log returns.
    - Long when predicted return > threshold (positive expected gain), Cash (0) otherwise.
    - Applies transaction fee on position transitions.
    """
    actual_returns = np.asarray(actual_returns, dtype=np.float64).ravel()
    predicted_returns = np.asarray(predicted_returns, dtype=np.float64).ravel()

    # Position signal: 1 for Long, 0 for Cash
    signals = (predicted_returns > threshold).astype(int)

    # Strategy simple returns: convert actual log return to simple return or multiply directly
    actual_simple_returns = np.expm1(actual_returns) if np.all(np.abs(actual_returns) < 1.0) else actual_returns
    strategy_returns = signals * actual_simple_returns

    # Deduct transaction fee on trade switches
    position_changes = np.abs(np.diff(signals, prepend=signals[0]))
    strategy_returns -= position_changes * fee

    # Cumulative wealth
    cum_market = np.cumprod(1.0 + actual_simple_returns) - 1.0
    cum_strategy = np.cumprod(1.0 + strategy_returns) - 1.0

    # Annualized Sharpe Ratio (assuming 365 crypto days)
    mean_strat = np.mean(strategy_returns)
    std_strat = np.std(strategy_returns) + 1e-9
    sharpe = float((mean_strat / std_strat) * np.sqrt(365))

    # Maximum Drawdown
    wealth_index = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

    return {
        "cumulative_market": cum_market,
        "cumulative_strategy": cum_strategy,
        "total_strategy_return": float(cum_strategy[-1] * 100.0) if len(cum_strategy) > 0 else 0.0,
        "total_market_return": float(cum_market[-1] * 100.0) if len(cum_market) > 0 else 0.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "signals": signals
    }

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
