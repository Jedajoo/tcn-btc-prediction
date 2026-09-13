"""
Cross-Asset Generalizability & Out-of-Domain Evaluation
Evaluates the Bitcoin-trained Hybrid TCN-GRU ensemble model on Ethereum (ETH-USD)
or any other crypto asset without any retraining or fine-tuning.
"""

import os
import sys
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from joblib import load, dump
import tensorflow as tf

import utils as ut

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Asset Evaluation on ETH-USD or another asset")
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker to test on (default: ETH-USD)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (default: 2020-01-01)")
    parser.add_argument("--end-date", type=str, default="2026-01-01", help="End date (default: 2026-01-01)")
    parser.add_argument("--eval-mode", type=str, choices=["matching", "full"], default="matching",
                        help="'matching' uses the 30 percent test slice matching BTC test dates; 'full' tests across all data.")
    return parser.parse_args()


def compute_cross_asset_features(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Computes identical stationary, scale-invariant technical indicators as dataset.py.
    """
    df = df_raw.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.levels[1]:
            df = df.xs(ticker, axis=1, level='Ticker')
        else:
            df.columns = df.columns.get_level_values(-1)

    # Clean Close vs Adj Close
    if 'Close' in df.columns and 'Adj Close' in df.columns:
        df = df.drop(columns=['Close'])
    elif 'Close' in df.columns and 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        df = df.drop(columns=['Close'])

    # Filter invalid volume rows
    df = df[df['Volume'] > 0].dropna()

    close_s = df['Adj Close'].squeeze()
    high_s = df['High'].squeeze()
    low_s = df['Low'].squeeze()
    open_s = df['Open'].squeeze()
    vol_s = df['Volume'].squeeze()

    # A. Stationary Price & Return Ratios
    df['Log_Return_1'] = np.log(close_s / (close_s.shift(1) + 1e-9)).fillna(0.0)
    df['Log_Return_3'] = np.log(close_s / (close_s.shift(3) + 1e-9)).fillna(0.0)
    df['Log_Return_5'] = np.log(close_s / (close_s.shift(5) + 1e-9)).fillna(0.0)
    df['Log_Return_10'] = np.log(close_s / (close_s.shift(10) + 1e-9)).fillna(0.0)
    df['High_Low_Ratio'] = ((high_s - low_s) / (close_s + 1e-9)).fillna(0.0)
    df['Close_Open_Ratio'] = ((close_s - open_s) / (open_s + 1e-9)).fillna(0.0)

    # B. Moving Average Distance
    df['SMA10'] = close_s.rolling(10, min_periods=1).mean()
    df['SMA25'] = close_s.rolling(25, min_periods=1).mean()
    df['SMA50'] = close_s.rolling(50, min_periods=1).mean()
    df['Dist_SMA10'] = ((close_s - df['SMA10']) / (df['SMA10'] + 1e-9)).fillna(0.0)
    df['Dist_SMA25'] = ((close_s - df['SMA25']) / (df['SMA25'] + 1e-9)).fillna(0.0)
    df['Dist_SMA50'] = ((close_s - df['SMA50']) / (df['SMA50'] + 1e-9)).fillna(0.0)

    # C. Normalized MACD
    ema_12 = close_s.ewm(span=12, min_periods=1).mean()
    ema_26 = close_s.ewm(span=26, min_periods=1).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
    macd_hist = macd_line - macd_signal
    df['MACD_Line_Norm'] = (macd_line / (close_s + 1e-9)).fillna(0.0)
    df['MACD_Signal_Norm'] = (macd_signal / (close_s + 1e-9)).fillna(0.0)
    df['MACD_Hist_Norm'] = (macd_hist / (close_s + 1e-9)).fillna(0.0)

    # D. Normalized RSI (14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df['RSI_Norm'] = (rsi - 50.0) / 50.0

    # E. Bollinger Bands Normalized
    std_dev = close_s.rolling(20, min_periods=1).std(ddof=0).fillna(0)
    m_band = close_s.rolling(20, min_periods=1).mean()
    upper_bband = m_band + 2 * std_dev
    lower_bband = m_band - 2 * std_dev
    band_width = upper_bband - lower_bband
    df['Band_Pos'] = ((close_s - lower_bband) / (band_width + 1e-9)).clip(-1.0, 2.0).fillna(0.5)
    df['Band_Width_Norm'] = (band_width / (close_s + 1e-9)).fillna(0.0)

    # F. Garman-Klass Volatility
    log_hl = np.log(np.maximum(high_s / (low_s + 1e-9), 1e-9))
    log_co = np.log(np.maximum(close_s / (open_s + 1e-9), 1e-9))
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df['GK_Vol'] = np.sqrt(np.maximum(gk_var, 0.0))
    df['GK_Vol_14'] = df['GK_Vol'].rolling(14, min_periods=1).mean()

    # G. Chaikin Money Flow (CMF 20)
    hl_diff = (high_s - low_s).replace(0, np.nan)
    mf_multiplier = ((close_s - low_s) - (high_s - close_s)) / (hl_diff + 1e-9)
    mf_multiplier = mf_multiplier.fillna(0.0)
    mf_volume = mf_multiplier * vol_s
    df['CMF'] = mf_volume.rolling(20, min_periods=1).sum() / (vol_s.rolling(20, min_periods=1).sum() + 1e-9)
    df['CMF'] = df['CMF'].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # H. Stochastic Oscillator Normalized
    lowest_low_14 = low_s.rolling(14, min_periods=1).min()
    highest_high_14 = high_s.rolling(14, min_periods=1).max()
    stoch_range = (highest_high_14 - lowest_low_14).replace(0, np.nan)
    stoch_k = ((close_s - lowest_low_14) / (stoch_range + 1e-9)) * 100.0
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    stoch_d = stoch_k.rolling(3, min_periods=1).mean().ffill().bfill()
    df['Stoch_K_Norm'] = (stoch_k - 50.0) / 50.0
    df['Stoch_D_Norm'] = (stoch_d - 50.0) / 50.0

    # I. Volume Relative Metrics
    vol_sma20 = vol_s.rolling(20, min_periods=1).mean()
    df['Volume_Pct_Change'] = (vol_s.pct_change()).clip(-2.0, 5.0).fillna(0.0)
    df['Volume_SMA_Ratio'] = ((vol_s / (vol_sma20 + 1e-9)) - 1.0).clip(-2.0, 5.0).fillna(0.0)

    # Next-Day Continuous Log Return Target
    df['Next_Adj_Close'] = close_s.shift(-1)
    df['Next_Log_Return'] = np.log(df['Next_Adj_Close'] / (close_s + 1e-9)).fillna(0.0)
    df['Next_Direction'] = (df['Next_Log_Return'] > 0).astype(int)

    df = df.dropna()
    return df


def main():
    args = parse_args()
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date
    eval_mode = args.eval_mode

    print("\n" + "=" * 65)
    print(f"CROSS-ASSET GENERALIZATION EVALUATION: BTC-USD MODEL ON {ticker}")
    print("=" * 65)

    # 1. Load Bitcoin Pipeline Artifacts (Scaler, Selected Features, Hyperparameters)
    scaler_path = "output/scaler/feature_scaler.joblib"
    cols_path = "output/scaler/feature_columns.joblib"
    hp_path = "output/model/best_hyperparameters.joblib"
    btc_eval_path = "output/model/ensemble_evaluation.joblib"

    if not os.path.exists(scaler_path) or not os.path.exists(cols_path):
        raise FileNotFoundError("Missing BTC scaler or feature columns. Run dataset.py first.")
    if not os.path.exists(hp_path):
        raise FileNotFoundError("Missing BTC hyperparameters. Run psotcn.py first.")

    btc_scaler = load(scaler_path)
    selected_features = load(cols_path)
    best_hp = load(hp_path)
    time_window = best_hp["time_window"]

    # Retrieve BTC training drift
    train_mean_drift = 0.00151174
    btc_summary = None
    if os.path.exists(btc_eval_path):
        btc_summary = load(btc_eval_path)
        train_mean_drift = btc_summary.get("train_mean_drift", train_mean_drift)

    print(f"Loaded BTC Scaler with {len(selected_features)} mRMR features.")
    print(f"Selected features: {selected_features}")
    print(f"Optimal Window = {time_window}, Filters = {best_hp['n_filters']}, GRU = {best_hp['gru_units']}")
    print(f"BTC Historical Drift: {train_mean_drift:.6f} ({train_mean_drift*100:.3f}%/day)")

    # 2. Check for Trained Seed Models
    seed_list = [42, 43, 44, 45, 46]
    seed_paths = [f"checkpoints/seeds/model_seed_{s}.keras" for s in seed_list]
    missing = [p for p in seed_paths if not os.path.exists(p)]
    if missing:
        print(f"\n[ERROR] Trained BTC model checkpoints not found in checkpoints/seeds/:")
        for m in missing:
            print(f"  - Missing: {m}")
        print("\nPlease run the BTC training command first:")
        print("  .venv\\Scripts\\python psotcn.py --final-ensemble\n")
        sys.exit(1)

    # 3. Download & Process Target Asset (e.g. ETH-USD)
    print(f"\nDownloading {ticker} from {start_date} to {end_date}...")
    df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if df_raw.empty:
        raise ValueError(f"Failed to download data for {ticker}.")

    print(f"Engineering technical indicators for {ticker}...")
    df_processed = compute_cross_asset_features(df_raw, ticker)
    print(f"Total processed days for {ticker}: {len(df_processed)}")

    # 4. Determine Evaluation Split
    if eval_mode == "matching":
        # Match BTC test split (last 30% of timeline)
        train_len = int(len(df_processed) * 0.70)
        eval_df = df_processed.iloc[train_len:].copy()
        print(f"Evaluation Mode: MATCHING 30% Test Window ({len(eval_df)} days: {eval_df.index[0].date()} to {eval_df.index[-1].date()})")
        # Include lookback buffer from preceding data for seamless sequences
        lookback_buffer = df_processed.iloc[train_len - time_window + 1 : train_len]
        full_eval_df = pd.concat([lookback_buffer, eval_df])
    else:
        eval_df = df_processed.copy()
        full_eval_df = df_processed.copy()
        print(f"Evaluation Mode: FULL Historical Timeline ({len(eval_df)} days: {eval_df.index[0].date()} to {eval_df.index[-1].date()})")

    # 5. Scale Features using the BTC RobustScaler (Zero Leakage)
    print(f"Transforming {ticker} features using the BTC RobustScaler...")
    eth_features_scaled = btc_scaler.transform(full_eval_df[selected_features])
    eth_returns = full_eval_df['Next_Log_Return'].values
    eth_prices = full_eval_df['Adj Close'].values

    # 6. Create Sequences
    X_eth, y_eth = ut.create_sequences(eth_features_scaled, eth_returns, time_window)
    eval_prices = eth_prices[-len(y_eth):]
    print(f"Created {len(X_eth)} sequence samples of shape {X_eth.shape}")

    # 7. Multi-Seed Ensemble Inference
    print("\n--- Running BTC-Trained Multi-Seed Ensemble on ETH Data ---")
    individual_predictions = []
    seed_metrics_list = []

    for seed in seed_list:
        model_path = f"checkpoints/seeds/model_seed_{seed}.keras"
        print(f"Loading Seed {seed} from {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

        pred_demeaned = model.predict(X_eth, verbose=0).ravel()
        # Add back drift to evaluate real return scale
        pred = pred_demeaned + train_mean_drift
        individual_predictions.append(pred)

        sm = ut.compute_regression_metrics(y_eth, pred)
        seed_metrics_list.append(sm)
        neg_count = np.sum(pred < 0)
        print(f"  Seed {seed} -> RMSE: {sm['rmse']:.5f} | MAE: {sm['mae']:.5f} | R2: {sm['r2']:.4f} | DA: {sm['directional_accuracy']:.2f}% | Negative Days: {neg_count}/{len(pred)} ({neg_count/len(pred)*100:.1f}%)")

        del model
        tf.keras.backend.clear_session()

    ensemble_preds = np.mean(individual_predictions, axis=0)
    ensemble_metrics = ut.compute_regression_metrics(y_eth, ensemble_preds)

    # 8. Financial Backtest on ETH
    backtest_res = ut.backtest_return_strategy(y_eth, ensemble_preds, threshold=0.0)

    print("\n" + "=" * 65)
    print(f"CROSS-ASSET PERFORMANCE: BTC-TRAINED MODEL ON {ticker}")
    print("=" * 65)
    print(f"Ensemble RMSE (Log Return) : {ensemble_metrics['rmse']:.6f}")
    print(f"Ensemble MAE  (Log Return) : {ensemble_metrics['mae']:.6f}")
    print(f"Ensemble MSE               : {ensemble_metrics['mse']:.8f}")
    print(f"Ensemble R2 Score          : {ensemble_metrics['r2']:.4f}")
    print(f"Directional Accuracy (DA)  : {ensemble_metrics['directional_accuracy']:.2f}%")
    print(f"Pearson Correlation (r)    : {ensemble_metrics['pearson_corr']:.4f}")
    print(f"Strategy Cumulative Return : {backtest_res['total_strategy_return']:.2f}% (vs {ticker} Buy&Hold: {backtest_res['total_market_return']:.2f}%)")
    print(f"Strategy Sharpe Ratio      : {backtest_res['sharpe_ratio']:.2f}")
    print(f"Strategy Max Drawdown      : {backtest_res['max_drawdown']:.2f}%")
    print(f"Total Long Days            : {backtest_res['total_trades_long']} ({backtest_res['long_ratio']*100:.1f}%)")
    print(f"Total Cash Days            : {backtest_res['total_trades_cash']} ({backtest_res['cash_ratio']*100:.1f}%)")
    print("=" * 65 + "\n")

    # 9. Comparison against BTC In-Domain Performance
    if btc_summary:
        btc_m = btc_summary.get("ensemble_metrics", {})
        btc_bt = btc_summary.get("backtest_results", {})
        print("--- CROSS-ASSET COMPARISON SUMMARY ---")
        print(f"{'Metric':<25} | {'In-Domain (BTC-USD)':<20} | {'Cross-Asset (' + ticker + ')':<20}")
        print("-" * 71)
        print(f"{'Directional Accuracy':<25} | {btc_m.get('directional_accuracy', 0):.2f}%{'':<14} | {ensemble_metrics['directional_accuracy']:.2f}%")
        print(f"{'Strategy Return':<25} | {btc_bt.get('total_strategy_return', 0):.2f}% (BH: {btc_bt.get('total_market_return', 0):.2f}%){'':<2} | {backtest_res['total_strategy_return']:.2f}% (BH: {backtest_res['total_market_return']:.2f}%)")
        print(f"{'Strategy Sharpe':<25} | {btc_bt.get('sharpe_ratio', 0):.2f}{'':<16} | {backtest_res['sharpe_ratio']:.2f}")
        print(f"{'Strategy Max Drawdown':<25} | {btc_bt.get('max_drawdown', 0):.2f}%{'':<13} | {backtest_res['max_drawdown']:.2f}%")
        print(f"{'Pearson Correlation':<25} | {btc_m.get('pearson_corr', 0):.4f}{'':<14} | {ensemble_metrics['pearson_corr']:.4f}")
        print(f"{'RMSE':<25} | {btc_m.get('rmse', 0):.5f}{'':<13} | {ensemble_metrics['rmse']:.5f}")
        print(f"{'R2 Score':<25} | {btc_m.get('r2', 0):.4f}{'':<14} | {ensemble_metrics['r2']:.4f}")
        print("-" * 71 + "\n")

    # 10. Generate Visual Plots
    plot_dir = f"output/plots/cross_asset_{ticker.split('-')[0].lower()}"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot A: Actual vs Predicted Return
    plt.figure(figsize=(14, 6))
    plt.plot(y_eth, label=f'Actual {ticker} Log Return', color='black', alpha=0.5, lw=1.2)
    plt.plot(ensemble_preds, label=f'BTC-Trained Hybrid TCN-GRU Predicted (RMSE={ensemble_metrics["rmse"]:.4f})', color='darkorange', lw=1.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'{ticker} Next-Day Return: Actual vs BTC-Trained Cross-Asset Prediction')
    plt.xlabel('Evaluation Sample Days')
    plt.ylabel('Log Return')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, f"{ticker}_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot B: Residual Scatter and Histogram
    residuals = y_eth - ensemble_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_eth, ensemble_preds, alpha=0.5, color='coral', s=20)
    min_v = min(np.min(y_eth), np.min(ensemble_preds))
    max_v = max(np.max(y_eth), np.max(ensemble_preds))
    ax1.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', label='Identity (Ideal)')
    ax1.set_title(f'{ticker} Actual vs Predicted Scatter')
    ax1.set_xlabel(f'Actual {ticker} Return')
    ax1.set_ylabel(f'Predicted Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, color='sandybrown', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', label=f'Mean Error ({np.mean(residuals):.5f})')
    ax2.set_title(f'{ticker} Prediction Residuals Distribution')
    ax2.set_xlabel('Residual (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{ticker}_residual_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot C: Trading Strategy vs Buy & Hold
    plt.figure(figsize=(14, 8))
    ax_top = plt.subplot(2, 1, 1)
    ax_top.plot(eval_prices, label=f'{ticker} Close Price', color='black', lw=1.2)
    long_mask = ensemble_preds > 0.0
    cash_mask = ~long_mask
    ax_top.scatter(np.where(long_mask)[0], eval_prices[long_mask], marker='^', color='green', s=25, alpha=0.7, label='Signal: Long (Pred > 0)')
    ax_top.scatter(np.where(cash_mask)[0], eval_prices[cash_mask], marker='v', color='red', s=25, alpha=0.7, label='Signal: Cash (Pred <= 0)')
    ax_top.set_title(f'Cross-Asset Signals: BTC-Trained Model on {ticker} Price')
    ax_top.set_ylabel('Price (USD)')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, alpha=0.3)

    ax_bot = plt.subplot(2, 1, 2, sharex=ax_top)
    ax_bot.plot(backtest_res['strategy_curve'] * 100.0, label=f'BTC-Trained Strategy on {ticker} ({backtest_res["total_strategy_return"]:.1f}%)', color='forestgreen', lw=1.8)
    ax_bot.plot(backtest_res['market_curve'] * 100.0, label=f'{ticker} Buy & Hold Benchmark ({backtest_res["total_market_return"]:.1f}%)', color='gray', linestyle='--', lw=1.4)
    ax_bot.set_title('Cumulative Return Backtest (%)')
    ax_bot.set_xlabel('Evaluation Days')
    ax_bot.set_ylabel('Cumulative Return (%)')
    ax_bot.legend(loc='upper left')
    ax_bot.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{ticker}_trading_backtest.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 11. Save Evaluation Results
    cross_eval_summary = {
        "ticker": ticker,
        "eval_mode": eval_mode,
        "ensemble_metrics": ensemble_metrics,
        "backtest_results": backtest_res,
        "seed_metrics": seed_metrics_list,
        "best_hyperparameters": best_hp
    }
    dump_path = f"output/model/{ticker.split('-')[0].lower()}_cross_asset_evaluation.joblib"
    dump(cross_eval_summary, dump_path)
    print(f"Saved cross-asset evaluation artifact to {dump_path}")
    print(f"Generated plots saved in {plot_dir}/")
    print("\nCross-asset evaluation completed successfully.")


if __name__ == "__main__":
    main()
