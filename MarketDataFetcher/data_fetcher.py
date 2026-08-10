import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_skeleton_and_tickers(skeleton_csv="skeleton_W_matrix.csv"):
    """
    Reads the structural prior CSV and extracts the list of unique tickers and 
    the N x N binary mask dataframe. Automatically detects if the CSV is 
    formatted as an Edge List (From, To) or a full Adjacency Matrix.
    """
    print(f"\nParsing structural skeleton from '{skeleton_csv}'...")
    
    if not os.path.exists(skeleton_csv):
        raise FileNotFoundError(f"[ERROR] {skeleton_csv} not found in current directory!")
        
    df_raw = pd.read_csv(skeleton_csv)
    cols = [str(c).strip().lower() for c in df_raw.columns]
    
    if 'from' in cols and 'to' in cols:
        print("  ✓ Detected format: Directed Edge List (From, To)")
        from_col = df_raw.columns[cols.index('from')]
        to_col = df_raw.columns[cols.index('to')]
        
        raw_tickers = []
        for row in df_raw[[from_col, to_col]].values:
            for t in row:
                t_clean = str(t).strip().upper()
                if t_clean not in raw_tickers and t_clean != 'NAN':
                    raw_tickers.append(t_clean)
                    
        tickers = raw_tickers
        N = len(tickers)
        
        mask_df = pd.DataFrame(0.0, index=tickers, columns=tickers)
        for _, row in df_raw.iterrows():
            t = str(row[from_col]).strip().upper()
            f = str(row[to_col]).strip().upper()
            if f != 'NAN' and t != 'NAN':
                mask_df.loc[f, t] = 1.0
                
    else:
        print("  ✓ Detected format: N x N Adjacency Matrix")
        mask_df = pd.read_csv(skeleton_csv, index_col=0)
        tickers = [str(t).strip().upper() for t in mask_df.index]
        mask_df.index = tickers
        mask_df.columns = tickers

    np.fill_diagonal(mask_df.values, 1.0)
    print(f"  ✓ Extracted {len(tickers)} unique assets: {tickers}")
    return tickers, mask_df


def get_dolthub_iv_history(tickers, lookback_days=252):
    """
    Queries DoltHub for historical implied volatility data using ticker-by-ticker requests.
    """
    print(f"\nQuerying DoltHub for Implied Volatility history of {len(tickers)} assets...")

    all_dfs = []
    api_url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

    for ticker in tickers:
        query = f"""
            SELECT date, act_symbol, iv_current
            FROM volatility_history
            WHERE act_symbol = '{ticker}'
            ORDER BY date DESC, act_symbol ASC
            LIMIT 1000
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

    print(f"  ✓ Retrieved {len(cleaned_df)} trading days of IV data for {len(cleaned_df.columns)} equities")
    return cleaned_df


def get_equity_price_history(tickers, lookback_days=252):
    """
    Fetches daily adjusted closing prices for equities from Yahoo Finance.
    """
    print(f"\nFetching equity price history for {len(tickers)} assets (For Return Correlations)...")
    
    try:
        df = yf.download(
            tickers=tickers,
            period=f'{lookback_days + 30}d',
            interval='1d',
            auto_adjust=True,
            progress=False
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                price_data = df['Close']
            elif 'Adj Close' in df.columns.levels[0]:
                price_data = df['Adj Close']
            else:
                price_data = df.xs(df.columns.levels[0][0], axis=1, level=0)
        else:
            price_data = df['Close'] if 'Close' in df.columns else df
        
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()
            price_data.columns = [tickers[0]]
        
        price_data = price_data[tickers]
        cleaned_df = price_data.dropna().tail(lookback_days)
        
        print(f"  ✓ Retrieved {len(cleaned_df)} trading days of price data for {len(tickers)} equities")
        return cleaned_df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch equity prices: {e}")
        raise


def calculate_return_correlation_matrix(price_dataframe):
    """
    Calculates the N x N historical log-return correlation matrix from price data
    and computes its lower triangular Cholesky decomposition.
    """
    print("\nCalculating equity return correlation matrix from price history...")
    log_returns = np.log(price_dataframe / price_dataframe.shift(1)).dropna()
    correlation_matrix = log_returns.corr().values
    
    print("  → Ensuring positive semi-definiteness via eigenvalue correction...")
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10 
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    try:
        L = np.linalg.cholesky(correlation_matrix)
        print("  ✓ Cholesky decomposition successful")
    except np.linalg.LinAlgError as e:
        print(f"  [WARNING] Correlation matrix is not positive definite after adjustment: {e}")
        raise
    
    return correlation_matrix, L


def export_cholesky_matrix_to_csv(cholesky_L, tickers, output_filename="cholesky_L_matrix.csv"):
    """Exports the lower triangular Cholesky decomposition matrix L to CSV format."""
    L_df = pd.DataFrame(cholesky_L, index=tickers, columns=tickers)
    L_df.to_csv(output_filename)
    print(f"\n✓ Exported Cholesky decomposition matrix L to '{output_filename}'")


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

    # Normalize by row sum to enforce row-stochastic boundaries
    theta_tilde = theta / theta.sum(axis=1, keepdims=True)
    return theta_tilde


def generate_hybrid_W_matrix(iv_dataframe, tickers, mask_df, fevd_steps=10):
    """
    Builds the W matrix by combining a structural binary mask (skeleton) 
    with dynamic edge weights from a Diebold-Yilmaz VAR KPPS FEVD on IV data.
    """
    print(f"\nBuilding Hybrid W matrix using structural prior and DoltHub IV data...")
    
    # Ensure IV dataframe columns exactly match our target tickers
    iv_dataframe = iv_dataframe[tickers]
    N = len(tickers)
    mask = mask_df.values
    
    print("  → Fitting VAR(1) model to Implied Volatilities...")
    model = VAR(iv_dataframe)
    results = model.fit(maxlags=1)
    
    print(f"  → Computing KPPS Generalized Forecast Error Variance Decomposition (Steps={fevd_steps})...")
    dy_matrix = calculate_kpps_generalized_fevd(results, forecast_horizon=fevd_steps)
    
    print("  → Applying structural prior mask to DY spillovers...")
    W_filtered = dy_matrix * mask

    min_threshold = 0.01  # 1% minimum spillover
    W_filtered[W_filtered < min_threshold] = 0.0
    
    W_final = np.zeros((N, N))
    for i in range(N):
        row_sum = np.sum(W_filtered[i, :])
        if row_sum > 1e-9:
            W_final[i, :] = W_filtered[i, :] / row_sum
        else:
            W_final[i, i] = 1.0
            
    return pd.DataFrame(W_final, index=tickers, columns=tickers)


def visualize_network_matrix(W_matrix, tickers, output_filename="W_matrix_network.png", weight_threshold=0.01):
    """Visualizes the network matrix W as a clean directed graph."""
    print(f"\nGenerating network visualization: {output_filename}...")
    
    W_array = W_matrix.values if hasattr(W_matrix, 'values') else W_matrix
    G = nx.DiGraph()
    G.add_nodes_from(tickers)
    
    for i, recipient in enumerate(tickers):
        for j, transmitter in enumerate(tickers):
            if i != j:  
                weight = W_array[i, j] 
                if weight >= weight_threshold:  
                    G.add_edge(transmitter, recipient, weight=weight)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='#1E3A8A', edgecolors='black', linewidths=1.5, node_size=2400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='white', ax=ax)
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w = min(weights) if weights else 0.01
    max_w = max(weights) if weights else 1.0
    
    rad = 0.18
    
    for u, v, data in G.edges(data=True):
        w = data['weight']
        norm_w = (w - min_w) / (max_w - min_w) if max_w > min_w else 0.5
        linewidth = 0.8 + 6.7 * (norm_w ** 1.2)
        alpha = 0.35 + 0.60 * norm_w
        mutation_scale = 12 + 12 * norm_w
        
        arrow = mpatches.FancyArrowPatch(
            pos[u], pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color="#2563EB" if norm_w > 0.5 else "#4B5563",
            alpha=alpha,
            shrinkA=22,
            shrinkB=22
        )
        ax.add_patch(arrow)
    
    ax.set_title("Network Volatility Spillover Topology (W)", fontsize=18, fontweight='bold', pad=25)
    ax.axis('off')
    
    legend_text = (
        f"Edge Threshold: {weight_threshold:.2f}\n"
        f"Total Active Edges: {len(G.edges())}\n"
        f"Line Thickness ∝ Spillover Weight\n"
        f"Blue Lines = Primary Spillovers (>50% max)"
    )
    ax.text(0.01, 0.99, legend_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='#F8F9FA', edgecolor='#CBD5E1'))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved clean network visualization to '{output_filename}'")
    plt.close()


def print_network_statistics(W_matrix, tickers):
    """Prints statistics about the network structure."""
    W_array = W_matrix.values if hasattr(W_matrix, 'values') else W_matrix
    print("\n[NETWORK STATISTICS]")
    print(f"  Total assets: {len(tickers)}")
    
    off_diagonal = W_array.copy()
    np.fill_diagonal(off_diagonal, 0)
    num_edges = np.sum(off_diagonal > 1e-6)
    
    print(f"  Total directed edges (weight > 1e-6): {num_edges}")
    print(f"  Network density: {num_edges / (len(tickers) * (len(tickers) - 1)):.2%}")
    print(f"\n  In-degree (incoming spillovers) and Out-degree (outgoing spillovers):")
    print(f"  {'Ticker':<10} {'In-Degree':<15} {'Out-Degree':<15}")
    print(f"  {'-'*40}")
    
    for i, ticker in enumerate(tickers):
        # Row [i, :] = In-Degree (Receptions)
        in_degree = np.sum(off_diagonal[i, :] > 1e-6)
        
        # Column [:, i] = Out-Degree (Transmissions)
        out_degree = np.sum(off_diagonal[:, i] > 1e-6)
        
        print(f"  {ticker:<10} {in_degree:<15} {out_degree:<15}")


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    skeleton_csv_path = "skeleton_W_matrix.csv"
    
    try:
        # 1. Parse tickers and mask dynamically from skeleton CSV
        target_tickers, structural_mask_df = load_skeleton_and_tickers(skeleton_csv_path)
        
        # 2. Fetch price history (for Cholesky return correlations)
        equity_prices = get_equity_price_history(target_tickers, lookback_days=252)
        correlation_matrix, cholesky_L = calculate_return_correlation_matrix(equity_prices)
        export_cholesky_matrix_to_csv(cholesky_L, target_tickers, "cholesky_L_matrix.csv")
        
        # 3. Fetch DoltHub IV history (for Network W matrix)
        real_market_ivs = get_dolthub_iv_history(target_tickers, lookback_days=252)
        
        # 4. Generate hybrid W matrix using parsed mask & DoltHub IVs
        W_matrix = generate_hybrid_W_matrix(real_market_ivs, target_tickers, 
                                            mask_df=structural_mask_df, 
                                            fevd_steps=10)
        
        print("\n=== FINAL ROW-STOCHASTIC W MATRIX ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(W_matrix.round(4))
        
        W_matrix.to_csv("calibrated_W_matrix.csv")
        print("\n✓ Saved network matrix to 'calibrated_W_matrix.csv'.")
        
        # 5. Visualize the network
        visualize_network_matrix(W_matrix, target_tickers, 
                                output_filename="W_matrix_network.png",
                                weight_threshold=0.01)
        
        # 6. Print network statistics
        print_network_statistics(W_matrix, target_tickers)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FATAL] Pipeline Exception: {e}")
        import traceback
        traceback.print_exc()