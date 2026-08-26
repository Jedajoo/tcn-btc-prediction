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
    confusion_matrix
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

def compute_receptive_field(k_size, dilations=(1, 2, 4, 8, 16), nb_stacks=1):
    """
    Computes theoretical receptive field for a 1D causal TCN:
    RF = 1 + 2 * (k - 1) * nb_stacks * sum(dilations)
    """
    return 1 + 2 * (int(k_size) - 1) * int(nb_stacks) * sum(dilations)

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
    Builds a TCN model with Multi-Head Attention, Layer Normalization,
    and GlobalAveragePooling1D for binary directional classification.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    
    # 1. Temporal Convolutional Network with sequence output
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

    # 2. Multi-Head Temporal Self-Attention
    num_heads = 2
    key_dim = max(16, int(n_filters // num_heads))
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="temporal_mha"
    )(tcn_out, tcn_out)

    # Residual Connection + Layer Normalization
    x = layers.Add(name="attn_residual")([tcn_out, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # 3. Global Average Pooling (replaces deep dense stack to reduce overfitting)
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # 4. Classification Head
    x = layers.Dense(32, activation='gelu', name="dense_projection")(x)
    x = layers.Dropout(float(dropout), name="head_dropout")(x)

    # 5. Output Layer: Probability of UP (1) vs DOWN (0)
    outputs = layers.Dense(1, activation='sigmoid', name="direction_probability")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PSO_TCN_Attention_Classifier")

    # Optimizer with decoupled weight decay (AdamW)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay)
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def get_early_stopping(patience=30, monitor='val_loss'):
    """
    Returns EarlyStopping callback with weight restoration.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes comprehensive binary classification metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    
    # Handle single class edge case in ROC-AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.5

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    loss = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    cm = confusion_matrix(y_true, y_pred)

    # Confidence calculation: distance from decision boundary (0.5)
    confidence = np.abs(y_prob - 0.5) * 2.0  # Range [0.0, 1.0]
    
    # High confidence subset metrics (> 0.60 certainty)
    high_conf_mask = confidence >= 0.20  # prob >= 0.60 or <= 0.40
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

def backtest_directional_strategy(actual_returns, y_prob, threshold=0.5, fee=0.0005):
    """
    Simulates a long-or-cash trading strategy based on model directional probability.
    - Long when y_prob >= threshold, Cash (0 return) otherwise.
    - Applies transaction fee on position switches.
    """
    actual_returns = np.asarray(actual_returns).ravel()
    y_prob = np.asarray(y_prob).ravel()

    signals = (y_prob >= threshold).astype(int)
    
    # Strategy returns: when signal is 1, earn actual_returns; otherwise 0
    strategy_returns = signals * actual_returns

    # Subtract transaction fees on position changes
    position_changes = np.abs(np.diff(signals, prepend=signals[0]))
    strategy_returns -= position_changes * fee

    # Cumulative returns
    cum_market = np.cumprod(1.0 + actual_returns) - 1.0
    cum_strategy = np.cumprod(1.0 + strategy_returns) - 1.0

    # Sharpe ratio (assuming 365 trading days for crypto, 0% risk-free rate)
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
