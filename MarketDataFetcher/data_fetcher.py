import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_save(t):
    ticker = yf.Ticker(t)

    # spot price (S0)
    data = ticker.history(period="1d")
    s0 = data['Close'].iloc[-1]

    # risk-free rate (r) from 13-week T-Bill
    r = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100.0

    # volatility (sigma), sd of daily log-returns over the last year
    hist = ticker.history(period="1y")['Close']
    log_returns = np.log(hist / hist.shift(1))
    sigma = log_returns.std() * np.sqrt(252)

    # dividend yield (q), TTM
    divs = ticker.dividends
    ttm_divs = divs[divs.index > (pd.Timestamp.now() - pd.DateOffset(years=1))].sum()
    q = ttm_divs / s0

    # save to CSV
    df = pd.DataFrame([[s0, r, sigma, q]], columns=['S0', 'r', 'sigma', 'q'])
    df.to_csv("market_data.csv", index=False)
    
    print(f"--- Data Fetched for {ticker} ---")
    print(f"S0: {s0:.2f} | r: {r:.4f} | sigma: {sigma:.4f} | q: {q:.4f}")