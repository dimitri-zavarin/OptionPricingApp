import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

def build_cboe_vol_network(vol_mapping, k_neighbors=2, lookback_days=252):
    print("--- Step 1: Downloading High-Quality Baseline Market Signals ---")
    underlying_tickers = list(vol_mapping.values())
    
    # Download data for assets that have flawless histories: VIX, IRX, and the equities themselves
    download_tickers = ["^VIX", "^IRX"] + underlying_tickers
    raw_data = yf.download(download_tickers, period="2y")['Close']
    
    # Clean up macro benchmarks
    vix_series = raw_data["^VIX"].dropna().tail(lookback_days) / 100.0
    irx_clean = raw_data["^IRX"].dropna()
    live_r = irx_clean.iloc[-1] / 100.0
    print(f"Current risk-free rate extracted from ^IRX: {live_r * 100.0:.3f}%")
    
    print("\n--- Step 2: Extracting Live Spot Stock Prices & Realized Vol Betas ---")
    spot_prices = {}
    vol_betas = {}
    
    for ticker in underlying_tickers:
        prices = raw_data[ticker].dropna()
        spot_prices[ticker] = prices.iloc[-1]
        
        # Calculate log returns
        log_rets = np.log(prices / prices.shift(1))
        # Annualized 20-day rolling realized volatility
        realized_vol = log_rets.rolling(20).std() * np.sqrt(252)
        
        # Line up the stock's realized vol with the market's implied vol (VIX)
        combined = pd.DataFrame({'stock_vol': realized_vol, 'mkt_vol': raw_data["^VIX"]/100.0}).dropna().tail(lookback_days)
        
        # Calculate Volatility Beta via OLS covariance slope
        covariance = np.cov(combined['stock_vol'], combined['mkt_vol'])[0, 1]
        market_variance = np.var(combined['mkt_vol'])
        
        # Safety clamp to keep the proxy relationship realistic
        vol_betas[ticker] = max(0.5, min(2.5, covariance / market_variance))
        print(f"  * {ticker}: Spot = ${spot_prices[ticker]:.2f}, Implied Vol Beta = {vol_betas[ticker]:.3f}")

    print("\n--- Step 2.5: Scraping Live Market Option Chains for C++ Template Solver ---")
    opt_strikes = []
    opt_maturities = []
    opt_target_prices = []
    
    for ticker in underlying_tickers:
        print(f"  -> Fetching exchange-traded option chains for {ticker}...")
        t_obj = yf.Ticker(ticker)
        
        try:
            # Grab available expiration dates and select the front-month expiration (~30 days out)
            expirations = t_obj.options
            chosen_exp = expirations[1] if len(expirations) > 1 else expirations[0]
            
            # Calculate maturity time in fractions of a year (T)
            days_to_mat = (datetime.strptime(chosen_exp, "%Y-%m-%d") - datetime.now()).days
            T_years = max(0.01, days_to_mat / 365.0)
            
            # Fetch the call option chain matrix
            opt_chain = t_obj.option_chain(chosen_exp).calls
            spot = spot_prices[ticker]
            
            # Locate the closest at-the-money (ATM) call option
            atm_option = opt_chain.iloc[(opt_chain['strike'] - spot).abs().argsort()[:1]].iloc[0]
            
            # Calculate the target mid-market trading price (Average of Bid and Ask)
            bid = atm_option['bid'] if 'bid' in opt_chain.columns else 0.0
            ask = atm_option['ask'] if 'ask' in opt_chain.columns else 0.0
            mid_price = (bid + ask) / 2.0
            
            if mid_price <= 0 or np.isnan(mid_price):
                mid_price = atm_option['lastPrice']
                
            # CRITICAL GUARD: If the price is a tiny fraction of the stock value (less than 1%),
            # it's stale/illiquid old data. Force an exception to trigger the fallback proxy.
            if mid_price < (spot * 0.01) or np.isnan(mid_price):
                raise ValueError("Market option data is stale or illiquid outside trading hours.")
                
            opt_strikes.append(atm_option['strike'])
            opt_maturities.append(T_years)
            opt_target_prices.append(mid_price)
            print(f"     Found ATM Call: Strike=${atm_option['strike']}, Expiry={chosen_exp} (T={T_years:.3f}), Mid-Price=${mid_price:.2f}")
            
        except Exception as e:
            # Enhanced fallback: Assign a realistic 10% premium for an ATM Call option
            print(f"     [WARNING] Option data stale/failed for {ticker}. Using robust ATM pricing proxy.")
            spot = spot_prices[ticker]
            opt_strikes.append(spot)               # Strike equals Spot (ATM)
            opt_maturities.append(0.0833)          # 1-Month Expiry expiration horizon
            opt_target_prices.append(spot * 0.10)  # 10% of stock spot price

    print("\n--- Step 3: Computing Euclidean Space Clusters (k-NN Graph) ---")
    # Build proxy implied volatility history matrix by scaling the VIX across each asset's beta
    proxy_vol_dict = {t: vix_series.values * vol_betas[t] for t in underlying_tickers}
    vol_data_filled = pd.DataFrame(proxy_vol_dict, index=vix_series.index)
    iv_profiles = vol_data_filled.T.values 
    
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    nn.fit(iv_profiles)
    distances, indices = nn.kneighbors(iv_profiles)
    
    n_assets = len(underlying_tickers)
    W = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for neighbor_idx in indices[i, 1:]:
            W[i, neighbor_idx] = 1.0 / k_neighbors

    print("\n--- Step 4: Estimating Log-ARCH Parameters via Market Anchor ---")
    # Fit our AR(1) log-ARCH regression strictly to the high-quality ^VIX data stream
    vix_var = vix_series.values ** 2
    ln_h_vix = np.log(vix_var)
    df_lag_vix = pd.DataFrame({'current': ln_h_vix[1:], 'lagged': ln_h_vix[:-1]})
    
    reg_vix = np.polyfit(df_lag_vix['lagged'], df_lag_vix['current'], deg=1)
    gamma_vix = max(0.05, min(0.85, reg_vix[0]))
    omega_vix = reg_vix[1]
    
    omega_baseline = []
    gamma_memory = []
    initial_v0 = []
    
    for ticker in underlying_tickers:
        current_v0 = (vix_series.iloc[-1] * vol_betas[ticker]) ** 2
        initial_v0.append(current_v0)
        gamma_memory.append(gamma_vix)
        
        # Scale baseline floor using the asset's specific volatility beta profile
        scaled_omega = omega_vix * vol_betas[ticker]
        omega_baseline.append(min(scaled_omega, np.log(current_v0) * 0.5))

    print("\n--- Step 5: Exporting Upgraded CSV Channels for C++ ---")
    pd.DataFrame(W).to_csv("C:/Users/Dimitri/source/repos/OptionPricingApp/OptionPricingApp/weight_matrix.csv", header=False, index=False)
    
    # Export all parameters, including the 3 new option target columns
    pd.DataFrame({
        'Ticker': underlying_tickers,
        'S0': [spot_prices[t] for t in underlying_tickers],
        'v0': initial_v0,
        'omega_baseline': omega_baseline,
        'gamma_memory': gamma_memory,
        'r': [live_r] * n_assets,
        'opt_strike': opt_strikes,
        'opt_maturity': opt_maturities,
        'opt_target_price': opt_target_prices
    }).to_csv("C:/Users/Dimitri/source/repos/OptionPricingApp/OptionPricingApp/network_config.csv", index=False)
    
    print("[SUCCESS] Pipeline fully populated with valid proxy trajectories. Files written.")

if __name__ == "__main__":
    cboe_universe = {
        "^VIX":   "SPY",   
        "^VXAPL": "AAPL",  
        "^VXAZN": "AMZN",  
        "^VXGOG": "GOOG",  
        "^VXGS":  "GS",    
        "^VXIBM": "IBM"    
    }
    build_cboe_vol_network(cboe_universe, k_neighbors=2)