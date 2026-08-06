import requests
import numpy as np
import pandas as pd
import yfinance as yf
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
        # Download OHLCV data with explicit auto_adjust=True
        df = yf.download(
            tickers=tickers,
            period=f'{lookback_days + 30}d',  # Extra buffer for dropped NaNs
            interval='1d',
            auto_adjust=True,                # Automatically adjusts OHLC for splits/dividends
            progress=False
        )
        
        # If multi-level columns exist (e.g. ('Close', 'NVDA')), extract 'Close' or 'Adj Close'
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                price_data = df['Close']
            elif 'Adj Close' in df.columns.levels[0]:
                price_data = df['Adj Close']
            else:
                price_data = df.xs(df.columns.levels[0][0], axis=1, level=0)
        else:
            # Single level column fallback
            price_data = df['Close'] if 'Close' in df.columns else df
        
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


def generate_hybrid_W_matrix(price_dataframe, tickers, skeleton_csv="skeleton_W_matrix.csv", fevd_steps=10):
    """
    Builds the W matrix by combining a structural binary mask (skeleton) 
    with dynamic edge weights from a Diebold-Yilmaz VAR FEVD.
    """
    print(f"\nBuilding Hybrid W matrix using structural prior: '{skeleton_csv}'...")
    N = len(tickers)
    
    # 1. Load Skeleton Mask
    try:
        skeleton_df = pd.read_csv(skeleton_csv, index_col=0)
        # Ensure the mask columns/index exactly match the order of our tickers
        skeleton_df = skeleton_df.loc[tickers, tickers]
        mask = skeleton_df.values
    except FileNotFoundError:
        print(f"[ERROR] {skeleton_csv} not found! Please create it in the project folder.")
        raise
    except Exception as e:
        print(f"[ERROR] Could not parse {skeleton_csv}: {e}")
        raise
        
    # CRITICAL: Ensure self-loops are explicitly allowed (1.0) for every asset
    np.fill_diagonal(mask, 1.0)
    
    # 2. Fit VAR and extract Diebold-Yilmaz Variance Decomposition (FEVD)
    print("  → Fitting VAR(1) model to log-returns...")
    log_returns = np.log(price_dataframe / price_dataframe.shift(1)).dropna()
    model = VAR(log_returns)
    results = model.fit(maxlags=1)
    
    print(f"  → Computing Forecast Error Variance Decomposition (Steps={fevd_steps})...")
    fevd = results.fevd(fevd_steps)
    
    # statsmodels fevd.decomp shape is (num_assets, fevd_steps, num_assets)
    # fevd.decomp[i, -1, j] is the proportion of asset i's variance explained by asset j at the final horizon
    dy_matrix = fevd.decomp[:, -1, :]
    
    # 3. Apply Structural Mask (Hadamard Product)
    print("  → Applying structural prior mask to DY spillovers...")
    W_filtered = dy_matrix * mask

    # 3b. Apply Minimum Threshold Floor (Zero out sub-1% noise)
    min_threshold = 0.01  # 1% minimum spillover
    W_filtered[W_filtered < min_threshold] = 0.0
    
    # 4. Row-Stochastic Normalization
    W_final = np.zeros((N, N))
    for i in range(N):
        row_sum = np.sum(W_filtered[i, :])
        if row_sum > 1e-9:
            W_final[i, :] = W_filtered[i, :] / row_sum
        else:
            # Fallback for completely isolated nodes
            W_final[i, i] = 1.0
            
    return pd.DataFrame(W_final, index=tickers, columns=tickers)


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    target_tickers = ["NVDA", "AMD", "AAPL", "MSFT", "GOOG", "AMZN", 
                      "JPM", "GS", "XOM", "META", "TSLA", "NFLX"]
    
    try:
        equity_prices = get_equity_price_history(target_tickers, lookback_days=252)
        
        # 1. Generate and save Cholesky Matrix
        correlation_matrix, cholesky_L = calculate_return_correlation_matrix(equity_prices)
        export_cholesky_matrix_to_csv(cholesky_L, target_tickers, "cholesky_L_matrix.csv")
        
        # 2. Generate and save Hybrid W Matrix
        W_matrix = generate_hybrid_W_matrix(
            price_dataframe=equity_prices, 
            tickers=target_tickers,
            skeleton_csv="skeleton_W_matrix.csv",
            fevd_steps=10
        )
        
        print("\n=== FINAL HYBRID ROW-STOCHASTIC W MATRIX ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(W_matrix.round(4))
        
        W_matrix.to_csv("calibrated_W_matrix.csv")
        print("\n✓ Saved network matrix to 'calibrated_W_matrix.csv'.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FATAL] Pipeline Exception: {e}")
        import traceback
        traceback.print_exc()