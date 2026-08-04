import requests
import numpy as np
import pandas as pd
import yfinance as yf  # Add this import
from statsmodels.tsa.api import VAR


def get_equity_price_history(tickers, lookback_days=252):
    """
    Fetches daily adjusted closing prices for equities from Yahoo Finance.
    This data is used to calculate historical return correlations.
    
    Args:
        tickers: List of equity ticker symbols
        lookback_days: Number of trading days to retrieve
        
    Returns:
        DataFrame with dates as index and tickers as columns (adjusted close prices)
    """
    print(f"Fetching equity price history for {len(tickers)} assets...")
    
    try:
        # Download OHLCV data from Yahoo Finance
        price_data = yf.download(
            tickers=tickers,
            period=f'{lookback_days + 30}d',  # Extra buffer for dropped NaNs
            interval='1d',
            progress=False
        )['Adj Close']
        
        # Handle single ticker case (returns Series instead of DataFrame)
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()
            price_data.columns = [tickers[0]]
        
        # Ensure correct column ordering matches input tickers
        price_data = price_data[tickers]
        
        # Drop NaN rows and tail to exact lookback window
        cleaned_df = price_data.dropna().tail(lookback_days)
        
        print(f"  ✓ Retrieved {len(cleaned_df)} trading days for {len(tickers)} equities")
        return cleaned_df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch equity prices: {e}")
        raise


def get_dolthub_iv_history(tickers, lookback_days=252):
    """
    Queries DoltHub for historical implied volatility data using ticker-by-ticker requests.
    This data is used for the Diebold-Yilmaz VAR weight matrix W.
    """
    print(f"Querying DoltHub for {len(tickers)} assets (IV data for network matrix)...")

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
            print(f"  -> [WARNING] No historical IV data found for {ticker}")

    if not all_dfs:
        raise KeyError("DoltHub returned an empty data structure for all assets.")

    df = pd.concat(all_dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df['iv_current'] = df['iv_current'].astype(float)

    pivot_df = df.pivot(index='date', columns='act_symbol', values='iv_current').sort_index()

    cleaned_df = pivot_df.ffill().tail(lookback_days).dropna(axis=1)

    return cleaned_df


def calculate_return_correlation_matrix(price_dataframe):
    """
    Calculates the N x N historical log-return correlation matrix from price data
    and computes its lower triangular Cholesky decomposition.
    
    Args:
        price_dataframe: DataFrame with prices indexed by date, columns are asset tickers
        
    Returns:
        Tuple of (correlation_matrix, cholesky_L_matrix)
    """
    print("\nCalculating equity return correlation matrix from price history...")
    
    # Compute log-returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(price_dataframe / price_dataframe.shift(1)).dropna()
    print(f"  ✓ Computed log-returns over {len(log_returns)} trading days")
    
    # Calculate correlation matrix (Pearson correlation of returns)
    correlation_matrix = log_returns.corr().values
    
    # Ensure positive semi-definite via eigenvalue decomposition
    # (handles numerical instabilities in real data)
    print("  → Ensuring positive semi-definiteness via eigenvalue correction...")
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10  # Clamp negative/near-zero eigenvalues
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Compute lower triangular Cholesky decomposition: Sigma = L * L^T
    try:
        L = np.linalg.cholesky(correlation_matrix)
        print("  ✓ Cholesky decomposition successful")
    except np.linalg.LinAlgError as e:
        print(f"  [WARNING] Correlation matrix is not positive definite after adjustment: {e}")
        raise
    
    return correlation_matrix, L


def export_cholesky_matrix_to_csv(cholesky_L, tickers, output_filename="cholesky_L_matrix.csv"):
    """
    Exports the lower triangular Cholesky decomposition matrix L to CSV format.
    
    Args:
        cholesky_L: N x N lower triangular matrix
        tickers: List of asset tickers for row/column labels
        output_filename: Output CSV filename
    """
    L_df = pd.DataFrame(cholesky_L, index=tickers, columns=tickers)
    L_df.to_csv(output_filename)
    print(f"\n✓ Exported Cholesky decomposition matrix L to '{output_filename}'")
    print(f"  Shape: {L_df.shape}")
    print("\n=== Cholesky L Matrix (first 5x5) ===")
    print(L_df.iloc[:5, :5].round(6))


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
    Fits a VAR model on IV data, calculates KPPS generalized spillovers, and enforces 
    top-K sparsity to optimize C++ engine computation.
    """
    print("\nBuilding network matrix W from IV spillovers...")
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
    
    print("  ✓ Network matrix W constructed and sparsified")
    return pd.DataFrame(W_sparse, index=tickers, columns=tickers)


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    target_tickers = ["NVDA", "AMD", "AAPL", "MSFT", "GOOG", "AMZN", 
                      "JPM", "GS", "XOM", "META", "TSLA", "NFLX"]
    
    try:
        # 1. Fetch equity price data for return correlations
        equity_prices = get_equity_price_history(target_tickers, lookback_days=252)
        
        # 2. Calculate and export Cholesky decomposition for equity return correlations
        correlation_matrix, cholesky_L = calculate_return_correlation_matrix(equity_prices)
        export_cholesky_matrix_to_csv(cholesky_L, target_tickers, "cholesky_L_matrix.csv")
        
        # 3. Fetch IV data for network matrix construction
        real_market_ivs = get_dolthub_iv_history(target_tickers, lookback_days=252)
        print("\nSuccessfully pulled IV data. Shape:", real_market_ivs.shape)
        
        # 4. Build sparse directional weight matrix W from IV spillovers
        W_matrix = generate_sparse_dynamic_W_matrix(real_market_ivs, self_reliance=0.20, horizon=10, max_neighbors=3)
        
        print("\n=== FINAL ROW-STOCHASTIC W MATRIX ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(W_matrix.round(4))
        
        # 5. Export network matrix for C++ engine
        W_matrix.to_csv("calibrated_W_matrix.csv")
        print("\n✓ Saved network matrix to 'calibrated_W_matrix.csv'.")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE: Both cholesky_L_matrix.csv and")
        print("calibrated_W_matrix.csv ready for C++ engine!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FATAL] Pipeline Exception: {e}")
        import traceback
        traceback.print_exc()