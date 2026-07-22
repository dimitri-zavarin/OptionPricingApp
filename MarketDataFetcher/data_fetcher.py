import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def get_dolthub_iv_history(tickers, lookback_days=252):
    """
    Queries DoltHub for historical implied volatility data using ticker-by-ticker requests.
    """
    print(f"Querying DoltHub for {len(tickers)} assets...")

    all_dfs = []
    api_url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

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


def calculate_kpps_generalized_fevd(var_results, forecast_horizon=10):
    """
    Implements the Generalized Forecast Error Variance Decomposition (KPPS)
    from Diebold & Yilmaz (2012) to ensure results are invariant to variable ordering.
    """
    sigma_u = var_results.sigma_u.values
    N = sigma_u.shape[0]
    A = var_results.ma_rep(maxn=forecast_horizon)
    
    theta = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            numerator = 0.0
            denominator = 0.0
            sigma_jj = sigma_u[j, j]
            
            e_i = np.zeros(N)
            e_i[i] = 1.0
            e_j = np.zeros(N)
            e_j[j] = 1.0
            
            for h in range(forecast_horizon):
                A_h = A[h]
                term_num = np.dot(np.dot(e_i, A_h), np.dot(sigma_u, e_j))
                numerator += (term_num ** 2) / sigma_jj
                
                term_den_matrix = np.dot(np.dot(A_h, sigma_u), A_h.T)
                term_den = np.dot(np.dot(e_i, term_den_matrix), e_i)
                denominator += term_den
                
            theta[i, j] = numerator / denominator

    # Normalize by row sum to enforce row-stochastic boundaries (Eq. 2 in paper)
    theta_tilde = theta / theta.sum(axis=1, keepdims=True)
    return theta_tilde


def generate_sparse_dynamic_W_matrix(iv_dataframe, self_reliance=0.20, horizon=10, max_neighbors=3):
    """
    Fits a VAR model, calculates KPPS generalized spillovers, and enforces 
    top-K sparsity to optimize C++ engine computation.
    """
    tickers = iv_dataframe.columns
    N = len(tickers)
    
    model = VAR(iv_dataframe)
    results = model.fit(maxlags=1)
    spillover_matrix = calculate_kpps_generalized_fevd(results, forecast_horizon=horizon)
    
    W_raw = spillover_matrix.copy()
    np.fill_diagonal(W_raw, 0.0)
    
    W_sparse = np.zeros((N, N))
    exogenous_budget = 1.0 - self_reliance
    
    for i in range(N):
        W_sparse[i, i] = self_reliance
        
        row_values = W_raw[i, :].copy()
        top_neighbor_indices = np.argsort(row_values)[::-1][:max_neighbors]
        
        filtered_row = np.zeros(N)
        filtered_row[top_neighbor_indices] = row_values[top_neighbor_indices]
        
        row_sum = np.sum(filtered_row)
        if row_sum > 1e-9:
            W_sparse[i, np.arange(N) != i] = (filtered_row[np.arange(N) != i] / row_sum) * exogenous_budget
        else:
            W_sparse[i, top_neighbor_indices] = exogenous_budget / max_neighbors
            
    return pd.DataFrame(W_sparse, index=tickers, columns=tickers)


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    target_tickers = ["NVDA", "AMD", "AAPL", "MSFT", "GOOG", "AMZN", 
                      "JPM", "GS", "XOM", "META", "TSLA", "NFLX"]
    
    try:
        # 1. Fetch data using your working fetcher function
        real_market_ivs = get_dolthub_iv_history(target_tickers, lookback_days=252)
        print("\nSuccessfully pulled real-world IV data. Shape:", real_market_ivs.shape)
        
        # 2. Build sparse directional weight matrix W
        W_matrix = generate_sparse_dynamic_W_matrix(real_market_ivs, self_reliance=0.20, horizon=10, max_neighbors=3)
        
        print("\n=== FINAL ROW-STOCHASTIC W MATRIX ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(W_matrix.round(4))
        
        # 3. Export matrix for C++ engine
        W_matrix.to_csv("calibrated_W_matrix.csv")
        print("\nSaved network matrix to 'calibrated_W_matrix.csv'. Ready for C++ engine!")
        
    except Exception as e:
        print(f"\nPipeline Exception: {e}")