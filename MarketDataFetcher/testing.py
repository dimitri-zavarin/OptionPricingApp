import pandas as pd
from pathlib import Path
import data_fetcher as df

if __name__ == "__main__":
    print("Testing data pipeline using semiconductor universe...")
    
    chosen_tickers = [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN",
        "AMAT", "LRCX", "MSFT", "GOOG", "AMZN", "AAPL"
    ]
    df.build_vol_network(chosen_tickers, k_neighbors=5)
    print("[SUCCESS] Test execution finished.")

    # Keep validation tools securely inside the runtime block
    config_path = Path(__file__).resolve().parent.parent / "OptionPricingApp" / "network_config.csv"
    matrix_path = Path(__file__).resolve().parent.parent / "OptionPricingApp" / "weight_matrix.csv"

    if config_path.exists() and matrix_path.exists():
        cfg = pd.read_csv(config_path)
        mat = pd.read_csv(matrix_path, header=None)
        print("\n" + "="*40 + "\nDIAGNOSTIC VERIFICATION SUCCESSFUL\n" + "="*40)
        print(f"Successfully tracked {len(cfg)} semiconductor assets.")
        print(f"Weight Matrix configuration shape generated: {mat.shape}")
    else:
        print("\n[ERROR] Output files were not successfully generated.")