import yfinance as yf
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from datetime import datetime, timedelta
from pathlib import Path

def get_dolthub_iv_history(tickers, lookback_days=252):
    print(f"Querying DoltHub for {len(tickers)} assets...")
    
    all_dfs = []
    api_url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
    
    # Add a buffer to ensure we have enough data for lookback and potential missing days
    buffer_days = lookback_days + 30 
    
    # Loop through each ticker and fetch its historical implied volatility data
    for ticker in tickers:
        query = f"""
            SELECT date, act_symbol, iv_current
            FROM volatility_history
            WHERE act_symbol = '{ticker}'
        """
        response = requests.get(api_url, params={'q': query})
        response.raise_for_status()
        
        data = response.json()
        if 'rows' in data and data['rows']:
            df_ticker = pd.DataFrame(data['rows'])
            all_dfs.append(df_ticker)
        else:
            print(f"  -> [WARNING] No historical data found for {ticker}")
            
    if not all_dfs:
        raise KeyError("DoltHub returned an empty data structure for all assets.")
        
    df = pd.concat(all_dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df['iv_current'] = df['iv_current'].astype(float)
    
    pivot_df = df.pivot(index='date', columns='act_symbol', values='iv_current').sort_index()
    
    cleaned_df = pivot_df.ffill().tail(lookback_days).dropna(axis=1)
    
    return cleaned_df

def build_vol_network(asset_tickers, k_neighbors, rho_global=0.05):
    # Get historical IV from DoltHub
    iv_history = get_dolthub_iv_history(asset_tickers)
    available_tickers = iv_history.columns.tolist()
    
    print("Getting spot prices and risk-free rate from yfinance...")
    raw_data = yf.download(["^IRX"] + available_tickers, period="5d")['Close']
    live_r = raw_data["^IRX"].iloc[-1] / 100.0
    
    # iv_profiles: rows are assets, columns are historical IV time-steps
    iv_profiles = iv_history.T.values
    
    # 1. Build the Weight Matrix (W)
    k = min(k_neighbors, len(available_tickers) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(iv_profiles)
    distances, indices = nn.kneighbors(iv_profiles)
    
    n_assets = len(available_tickers)
    W_mat = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for neighbor_idx in indices[i, 1:]:
            W_mat[i, neighbor_idx] = 1.0 / k

    # --- PATHWAY A: 2SLS/GMM NETWORK CALIBRATION ---
    print("Calibrating structural network parameters via 2SLS (Otto et al. 2023)...")
    
    # Pre-calculate matrix objects for the instruments
    ln_Y_squared = np.log(iv_history**2)
    Y_mat = ln_Y_squared.values      # T x N matrix of log-squared returns
    W_mat_sq = np.dot(W_mat, W_mat)  # W^2 (Neighbors of neighbors)
    
    omega_baseline = []
    gamma_memory = []
    mu_smearing = []
    initial_v0 = []
    spot_prices = []
    
    for i, ticker in enumerate(available_tickers):
        # Time-series vectors for the target stock
        y_curr = Y_mat[1:, i]       # X_{t}
        y_lag = Y_mat[:-1, i]       # X_{t-1}
        
        # Endogenous regressor: Contemporaneous spatial spillover
        spatial_curr = np.dot(Y_mat[1:, :], W_mat[i, :])
        
        # The Instruments (Exogenous predictors)
        spatial_lag1 = np.dot(Y_mat[:-1, :], W_mat[i, :])     # W * X_{t-1}
        spatial_lag2 = np.dot(Y_mat[:-1, :], W_mat_sq[i, :])  # W^2 * X_{t-1}
        
        # STAGE 1: Instrument the endogenous spatial term
        # Regress spatial_curr onto the safe, lagged instruments
        Z_instruments = np.column_stack([np.ones(len(y_lag)), y_lag, spatial_lag1, spatial_lag2])
        stage1_reg = np.linalg.lstsq(Z_instruments, spatial_curr, rcond=None)[0]
        spatial_curr_hat = np.dot(Z_instruments, stage1_reg)
        
        # STAGE 2: Structural Regression
        # Regress the target return onto its past and the *purified* spatial prediction
        X_structural = np.column_stack([np.ones(len(y_curr)), y_lag, spatial_curr_hat])
        structural_reg = np.linalg.lstsq(X_structural, y_curr, rcond=None)[0]
        
        intercept = structural_reg[0]
        gamma = structural_reg[1]
        
        # NON-PARAMETRIC SMEARING FACTOR
        # We calculate the residual using the fixed rho_global that your C++ engine uses
        # This guarantees perfect alignment between the Python estimator and C++ simulator
        u_hat = y_curr - (intercept + gamma * y_lag + rho_global * spatial_curr)
        smearing_factor = -np.log(np.mean(np.exp(u_hat)))
        
        # Enforce realistic stability bounds (Notice the ceiling is relaxed to 0.85)
        gamma_bounded = max(0.05, min(0.85, gamma))
        
        initial_v0.append(iv_history[ticker].iloc[-1]**2)
        gamma_memory.append(gamma_bounded)
        omega_baseline.append(intercept)
        mu_smearing.append(smearing_factor)
        spot_prices.append(raw_data[ticker].iloc[-1])

    base_path = Path(__file__).resolve().parent.parent / "OptionPricingApp"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Export weight matrix to CSV
    pd.DataFrame(W_mat).to_csv(base_path / "weight_matrix.csv", header=False, index=False)
    
    # Export system config
    pd.DataFrame({
        'Ticker': available_tickers,
        'S0': spot_prices,
        'v0': initial_v0,
        'omega_baseline': omega_baseline,
        'gamma_memory': gamma_memory,
        'mu_smearing': mu_smearing,
        'r': [live_r] * len(available_tickers)
    }).to_csv(base_path / "network_config.csv", index=False)

def generate_calibration_options(asset_tickers):
    print(f"Generating C++ calibration options for {len(asset_tickers)} assets...")
    calibration_rows = []
    
    today = datetime.now()
    
    for ticker in asset_tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            expirations = yf_ticker.options
            if not expirations:
                print(f"  -> [WARNING] No options chains found for {ticker}")
                continue
                
            target_exp_str = expirations[min(2, len(expirations) - 1)] 
            for exp in expirations:
                days_to_mat = (datetime.strptime(exp, "%Y-%m-%d") - today).days
                if 90 <= days_to_mat <= 180:
                    target_exp_str = exp
                    break
                    
            target_date = datetime.strptime(target_exp_str, "%Y-%m-%d")
            ttm_years = (target_date - today).days / 365.25
            
            opt_chain = yf_ticker.option_chain(target_exp_str)
            calls_df = opt_chain.calls
            
            calls_df = calls_df[(calls_df['volume'] > 10) & (calls_df['bid'] > 0.0)].copy()
            
            if calls_df.empty:
                print(f"  -> [WARNING] No liquid contracts found for {ticker} at {target_exp_str}")
                continue
            
            spot_price = yf_ticker.history(period="1d")["Close"].iloc[-1]
            
            calls_df['strike_diff'] = (calls_df['strike'] - spot_price).abs()
            best_option = calls_df.sort_values(by='strike_diff').iloc[0]
            
            market_price = (best_option['bid'] + best_option['ask']) / 2.0
            
            if np.isnan(market_price) or market_price <= 0.0:
                market_price = best_option['lastPrice']
                
            calibration_rows.append({
                'Ticker': ticker,
                'Strike': best_option['strike'],
                'Maturity': round(ttm_years, 6),
                'MarketPrice': round(market_price, 4),
                'IsCall': 1
            })
            print(f"  -> Calibrated {ticker}: Spot={spot_price:.2f}, Strike={best_option['strike']:.2f}, TTM={ttm_years:.4f}, MidPrice={market_price:.2f}")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed to compile contract parameters for {ticker}: {e}")
            
    df_calib = pd.DataFrame(calibration_rows)
    base_path = Path(__file__).resolve().parent.parent / "OptionPricingApp"
    base_path.mkdir(parents=True, exist_ok=True)
    df_calib.to_csv(base_path / "calibration_options.csv", index=False)

if __name__ == "__main__":
    chosen_tickers = [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN",
        "AMAT", "LRCX", "MSFT", "GOOG", "AMZN", "AAPL"
    ]
    
    print(f"Initiating full data pipeline for {len(chosen_tickers)} assets...")
    
    # Run the network generation (Notice rho_global=0.05 is passed to match C++)
    build_vol_network(chosen_tickers, k_neighbors=5, rho_global=0.05)
    
    # Run the contract generator 
    generate_calibration_options(chosen_tickers)
    
    # POST-EXECUTION DIAGNOSTICS
    print("\n" + "="*45)
    print("PIPELINE DIAGNOSTICS & VERIFICATION")
    print("="*45)
    
    base_path = Path(__file__).resolve().parent.parent / "OptionPricingApp"
    config_path = base_path / "network_config.csv"
    matrix_path = base_path / "weight_matrix.csv"
    calib_path = base_path / "calibration_options.csv"
    
    all_passed = True
    
    if config_path.exists() and matrix_path.exists():
        cfg = pd.read_csv(config_path)
        mat = pd.read_csv(matrix_path, header=None)
        
        print(f"[OK] Network Config: Extracted {len(cfg)} asset profiles.")
        print(f"[OK] Weight Matrix: Built spatial grid of shape {mat.shape}.")
        
        if len(cfg) != mat.shape[0] or mat.shape[0] != mat.shape[1]:
            print("  -> [CRITICAL WARNING] Dimension mismatch! Matrix size does not match config layout.")
            all_passed = False
    else:
        print("[ERROR] Missing core network artifacts (network_config.csv or weight_matrix.csv).")
        all_passed = False
        
    if calib_path.exists():
        calib = pd.read_csv(calib_path)
        print(f"[OK] Calibration Exporter: Tracked {len(calib)} vanilla contracts.")
        
        if len(calib) != len(chosen_tickers):
            print(f"  -> [WARNING] Expected {len(chosen_tickers)} contracts, but only generated {len(calib)}.")
            missing = set(chosen_tickers) - set(calib['Ticker'].tolist())
            if missing:
                print(f"  -> Missing tickers: {missing}")
            all_passed = False
    else:
        print("[ERROR] Missing calibration artifact (calibration_options.csv).")
        all_passed = False
        
    print("-" * 45)
    if all_passed:
        print("[SUCCESS] All artifacts generated. Dimensions align perfectly.")
        print("Data environment is locked and ready for C++ ingestion.")
    else:
        print("[FAILED] Pipeline execution encountered structural discrepancies.")
    print("="*45)