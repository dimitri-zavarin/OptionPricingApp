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
    
    print("Getting spot prices and risk-free rate from yfinance")
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

if __name__ == "__main__":
    chosen_tickers = [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN",
        "AMAT", "LRCX", "MSFT", "GOOG", "AMZN", "AAPL"
    ]
    build_vol_network(chosen_tickers, k_neighbors=5)