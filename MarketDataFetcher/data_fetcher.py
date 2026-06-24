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

def build_vol_network(asset_tickers, k_neighbors):
    # Get historical IV from DoltHub
    iv_history = get_dolthub_iv_history(asset_tickers)
    available_tickers = iv_history.columns.tolist()
    
    print("Getting spot prices and risk-free rate from yfinance...")
    raw_data = yf.download(["^IRX"] + available_tickers, period="5d")['Close']
    live_r = raw_data["^IRX"].iloc[-1] / 100.0
    
    # iv_profiles: rows are assets, columns are historical IV time-steps
    iv_profiles = iv_history.T.values
    
    k = min(k_neighbors, len(available_tickers) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(iv_profiles)
    distances, indices = nn.kneighbors(iv_profiles)
    
    n_assets = len(available_tickers)
    W = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for neighbor_idx in indices[i, 1:]:
            W[i, neighbor_idx] = 1.0 / k

    # Log-ARCH parameters
    omega_baseline = []
    gamma_memory = []
    initial_v0 = []
    spot_prices = []
    
    for ticker in available_tickers:
        iv_series = iv_history[ticker]
        ivar_series = iv_series**2
        ln_ivar = np.log(ivar_series)
        
        df_reg = pd.DataFrame({
            'current': ln_ivar, 
            'lagged': ln_ivar.shift(1)
        }).dropna()
        
        reg = np.polyfit(df_reg['lagged'], df_reg['current'], deg=1)
        
        # Bound gamma in [0.05, 0.95] to avoid extreme results
        gamma = max(0.05, min(0.95, reg[0]))
        omega = reg[1]
        
        initial_v0.append(ivar_series.iloc[-1])
        gamma_memory.append(gamma)
        omega_baseline.append(omega)
        spot_prices.append(raw_data[ticker].iloc[-1])

    base_path = Path(__file__).resolve().parent.parent / "OptionPricingApp"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Export weight matrix and parameters to CSV
    pd.DataFrame(W).to_csv(base_path / "weight_matrix.csv", header=False, index=False)
    
    pd.DataFrame({
        'Ticker': available_tickers,
        'S0': spot_prices,
        'v0': initial_v0,
        'omega_baseline': omega_baseline,
        'gamma_memory': gamma_memory,
        'r': [live_r] * len(available_tickers)
    }).to_csv(base_path / "network_config.csv", index=False)

def generate_calibration_options(asset_tickers):
    print(f"Generating C++ calibration options for {len(asset_tickers)} assets...")
    calibration_rows = []
    
    # Anchor date for exact TTM calculation
    today = datetime.now()
    
    for ticker in asset_tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Fetch available expiration dates
            expirations = yf_ticker.options
            if not expirations:
                print(f"  -> [WARNING] No options chains found for {ticker}")
                continue
                
            # Pick an option expiration between 90 and 180 days from today, or fallback to the third available expiration
            target_exp_str = expirations[min(2, len(expirations) - 1)] # Fallback
            for exp in expirations:
                days_to_mat = (datetime.strptime(exp, "%Y-%m-%d") - today).days
                if 90 <= days_to_mat <= 180:
                    target_exp_str = exp
                    break
                    
            target_date = datetime.strptime(target_exp_str, "%Y-%m-%d")
            
            # Calculate time to maturity in years
            ttm_years = (target_date - today).days / 365.25
            
            # Extract the options chain dataframe
            opt_chain = yf_ticker.option_chain(target_exp_str)
            calls_df = opt_chain.calls
            
            # Require at least 10 contracts of volume and a non-zero bid to ensure fresh pricing
            calls_df = calls_df[(calls_df['volume'] > 10) & (calls_df['bid'] > 0.0)].copy()
            
            if calls_df.empty:
                print(f"  -> [WARNING] No liquid contracts found for {ticker} at {target_exp_str}")
                continue
            
            # Get current spot price to find an At-The-Money contract
            spot_price = yf_ticker.history(period="1d")["Close"].iloc[-1]
            
            # Find the contract where the strike is closest to the spot price
            calls_df['strike_diff'] = (calls_df['strike'] - spot_price).abs()
            best_option = calls_df.sort_values(by='strike_diff').iloc[0]
            
            # Calculate the mid-price of the bid-ask spread
            market_price = (best_option['bid'] + best_option['ask']) / 2.0
            
            # Fallback if bid-ask spread data is missing or stale
            if np.isnan(market_price) or market_price <= 0.0:
                market_price = best_option['lastPrice']
                
            calibration_rows.append({
                'Ticker': ticker,
                'Strike': best_option['strike'],
                'Maturity': round(ttm_years, 6),
                'MarketPrice': round(market_price, 4),
                'IsCall': 1 # Enforce 1 for Call, 0 for Put
            })
            print(f"  -> Calibrated {ticker}: Spot={spot_price:.2f}, Strike={best_option['strike']:.2f}, TTM={ttm_years:.4f}, MidPrice={market_price:.2f}")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed to compile contract parameters for {ticker}: {e}")
            
    # Write directly into a CSV
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
    
    # Run the network generation (builds weight_matrix.csv and network_config.csv)
    build_vol_network(chosen_tickers, k_neighbors=5)
    
    # Run the contract generator (builds calibration_options.csv)
    generate_calibration_options(chosen_tickers)
    

    # POST-EXECUTION DIAGNOSTICS
    print("\n" + "="*45)
    print("PIPELINE DIAGNOSTICS & VERIFICATION")
    print("="*45)
    
    # Define output paths mirroring export logic
    base_path = Path(__file__).resolve().parent.parent / "OptionPricingApp"
    config_path = base_path / "network_config.csv"
    matrix_path = base_path / "weight_matrix.csv"
    calib_path = base_path / "calibration_options.csv"
    
    all_passed = True
    
    # Diagnostic 1: Verify the Core Network Ecosystem
    if config_path.exists() and matrix_path.exists():
        cfg = pd.read_csv(config_path)
        mat = pd.read_csv(matrix_path, header=None)
        
        print(f"[OK] Network Config: Extracted {len(cfg)} asset profiles.")
        print(f"[OK] Weight Matrix: Built spatial grid of shape {mat.shape}.")
        
        # Dimension Guardrail: Matrix grid must be a perfect N x N square matching the config rows
        if len(cfg) != mat.shape[0] or mat.shape[0] != mat.shape[1]:
            print("  -> [CRITICAL WARNING] Dimension mismatch! Matrix size does not match config layout.")
            all_passed = False
    else:
        print("[ERROR] Missing core network artifacts (network_config.csv or weight_matrix.csv).")
        all_passed = False
        
    # Diagnostic 2: Verify the Calibration Contracts
    if calib_path.exists():
        calib = pd.read_csv(calib_path)
        print(f"[OK] Calibration Exporter: Tracked {len(calib)} vanilla contracts.")
        
        # Guardrail: Ensure every ticker successfully pulled an option contract
        if len(calib) != len(chosen_tickers):
            print(f"  -> [WARNING] Expected {len(chosen_tickers)} contracts, but only generated {len(calib)}.")
            missing = set(chosen_tickers) - set(calib['Ticker'].tolist())
            if missing:
                print(f"  -> Missing tickers: {missing}")
            all_passed = False
    else:
        print("[ERROR] Missing calibration artifact (calibration_options.csv).")
        all_passed = False
        
    # Final Pipeline Status
    print("-" * 45)
    if all_passed:
        print("[SUCCESS] All artifacts generated. Dimensions align perfectly.")
        print("Data environment is locked and ready for C++ ingestion.")
    else:
        print("[FAILED] Pipeline execution encountered structural discrepancies.")
    print("="*45)