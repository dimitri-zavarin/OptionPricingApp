import requests
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def visualize_network_matrix(W_matrix, tickers, output_filename="W_matrix_network.png", weight_threshold=0.01):
    """
    Visualizes the network matrix W as a clean directed graph where line thickness 
    and opacity dramatically communicate spillover magnitude without text clutter.
    """
    print(f"\nGenerating network visualization: {output_filename}...")
    
    W_array = W_matrix.values if hasattr(W_matrix, 'values') else W_matrix
    G = nx.DiGraph()
    G.add_nodes_from(tickers)
    
    for i, recipient in enumerate(tickers):
        for j, transmitter in enumerate(tickers):
            if i != j:  
                weight = W_array[i, j]  # Weight of spillover flowing FROM j TO i
                if weight >= weight_threshold:  
                    G.add_edge(transmitter, recipient, weight=weight) # Arrow points j -> i
    
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    # 1. Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color='#1E3A8A', 
                           edgecolors='black', linewidths=1.5,
                           node_size=2400, ax=ax)
    
    # Node Ticker Labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='white', ax=ax)
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w = min(weights) if weights else 0.01
    max_w = max(weights) if weights else 1.0
    
    rad = 0.18  # Uniform curvature
    
    # 2. Draw Clean Curved Edges with Scaled Width & Opacity
    for u, v, data in G.edges(data=True):
        w = data['weight']
        p1 = pos[u]
        p2 = pos[v]
        
        # Linear scaling across min_w -> max_w range
        # Thickness ranges from 0.8px (1% spillover) to 7.5px (dominant spillover)
        norm_w = (w - min_w) / (max_w - min_w) if max_w > min_w else 0.5
        linewidth = 0.8 + 6.7 * (norm_w ** 1.2)  # Slight exponent to make high weights pop more
        
        # Alpha ranges from 0.35 (subtle) to 0.95 (crisp)
        alpha = 0.35 + 0.60 * norm_w
        
        # Arrowhead size scales slightly with width
        mutation_scale = 12 + 12 * norm_w
        
        arrow = mpatches.FancyArrowPatch(
            p1, p2,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color="#2563EB" if norm_w > 0.5 else "#4B5563",  # Highlight top-tier edges in blue
            alpha=alpha,
            shrinkA=22,
            shrinkB=22
        )
        ax.add_patch(arrow)
    
    ax.set_title("Network Volatility Spillover Topology (W)", fontsize=18, fontweight='bold', pad=25)
    ax.axis('off')
    
    # Clean Legend Box
    legend_text = (
        f"Edge Threshold: {weight_threshold:.2f}\n"
        f"Total Active Edges: {len(G.edges())}\n"
        f"Line Thickness ∝ Spillover Weight\n"
        f"Blue Lines = Primary Spillovers (>50% max)"
    )
    ax.text(0.01, 0.99, legend_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='#F8F9FA', edgecolor='#CBD5E1'))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved clean network visualization to '{output_filename}'")
    plt.close()


def print_network_statistics(W_matrix, tickers):
    """
    Prints statistics about the network structure.
    """
    W_array = W_matrix.values if hasattr(W_matrix, 'values') else W_matrix
    
    print("\n[NETWORK STATISTICS]")
    print(f"  Total assets: {len(tickers)}")
    
    # Count non-zero edges (excluding self-loops and diagonal)
    off_diagonal = W_array.copy()
    np.fill_diagonal(off_diagonal, 0)
    num_edges = np.sum(off_diagonal > 1e-6)
    
    print(f"  Total directed edges (weight > 1e-6): {num_edges}")
    print(f"  Network density: {num_edges / (len(tickers) * (len(tickers) - 1)):.2%}")
    
    # In-degree and out-degree
    print(f"\n  In-degree (incoming spillovers) and Out-degree (outgoing spillovers):")
    print(f"  {'Ticker':<10} {'In-Degree':<15} {'Out-Degree':<15}")
    print(f"  {'-'*40}")
    
    for i, ticker in enumerate(tickers):
        in_degree = np.sum(off_diagonal[:, i] > 1e-6)
        out_degree = np.sum(off_diagonal[i, :] > 1e-6)
        print(f"  {ticker:<10} {in_degree:<15} {out_degree:<15}")


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    target_tickers = ["NVDA", "AMD", "AAPL", "MSFT", "GOOG", "AMZN", 
                      "JPM", "GS", "XOM", "META", "TSLA", "NFLX"]
    
    try:
        equity_prices = get_equity_price_history(target_tickers, lookback_days=252)
        
        # Cholesky from returns
        correlation_matrix, cholesky_L = calculate_return_correlation_matrix(equity_prices)
        export_cholesky_matrix_to_csv(cholesky_L, target_tickers, "cholesky_L_matrix.csv")
        
        # Network matrix from hybrid approach
        W_matrix = generate_hybrid_W_matrix(equity_prices, target_tickers, 
                                            skeleton_csv="skeleton_W_matrix.csv", 
                                            fevd_steps=10)
        
        print("\n=== FINAL ROW-STOCHASTIC W MATRIX ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(W_matrix.round(4))
        
        W_matrix.to_csv("calibrated_W_matrix.csv")
        print("\n✓ Saved network matrix to 'calibrated_W_matrix.csv'.")
        
        # Visualize the network
        visualize_network_matrix(W_matrix, target_tickers, 
                                output_filename="W_matrix_network.png",
                                weight_threshold=0.01)
        
        # Print network statistics
        print_network_statistics(W_matrix, target_tickers)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FATAL] Pipeline Exception: {e}")
        import traceback
        traceback.print_exc()