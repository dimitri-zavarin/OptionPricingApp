import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import norm
import subprocess
import pandas as pd
import io
import os

class VolatilitySurfaceCalibrator:
    """
    Calibrates SDE parameters (kappa, theta, xi, gamma, rho) by matching 
    simulated vol surface to market vol surface from Bloomberg.
    """
    
    def __init__(self, config, market_vol_surface, market_strikes, market_maturities, assets):
        """
        @param config: SimulationConfig (will be modified during calibration)
        @param market_vol_surface: 3D array of implied vols (strikes x maturities x assets)
        @param market_strikes: array of strike prices
        @param market_maturities: array of maturities in years
        @param assets: list of asset tickers
        """
        self.config = config
        self.market_vol_surface = market_vol_surface  # Shape: (num_strikes, num_maturities, num_assets)
        self.market_strikes = market_strikes
        self.market_maturities = market_maturities
        self.assets = assets
        self.call_count = 0
    
    def black_scholes_price_to_iv(self, option_price, S, K, T, r, option_type='call'):
        """
        Converts option price to implied volatility using Newton-Raphson inversion.
        Needed for simulated side: simulator outputs prices, need to convert to IVs.
        Market side: Bloomberg gives IVs directly, no conversion needed.
        
        @param option_price: European option price
        @param S: Spot price
        @param K: Strike price
        @param T: Time to maturity (years)
        @param r: Risk-free rate
        @param option_type: 'call' or 'put'
        @return: Implied volatility
        """
        sigma = 0.2  # Initial guess
        for _ in range(10):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-10)
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                vega = S*norm.pdf(d1)*np.sqrt(T)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                vega = S*norm.pdf(d1)*np.sqrt(T)
            
            sigma = sigma - (price - option_price) / (vega + 1e-10)
        
        return max(sigma, 0.001)  # Ensure positive
    
    def get_simulated_vol_surface(self, params):
        """
        Run simulator with given parameters and extract vol surface.
        
        @param params: [kappa, theta, xi, gamma, rho]
        @return: 2D array of implied vols (strikes x maturities)
        """
        # Update config with candidate parameters (apply uniformly to all assets)
        self.config.kappa = np.full(self.config.num_assets, params[0])
        self.config.theta = np.full(self.config.num_assets, params[1])
        self.config.xi = np.full(self.config.num_assets, params[2])
        self.config.gamma = np.full(self.config.num_assets, params[3])
        self.config.rho = np.full(self.config.num_assets, params[4])
        
        sim_vol_surface = np.zeros((len(self.market_strikes), len(self.market_maturities), len(self.assets)))
        
        for asset_idx, asset in enumerate(self.assets):
            S0 = self.config.S0[asset_idx]
            
            for j, maturity in enumerate(self.market_maturities):
                for i, strike in enumerate(self.market_strikes):  # <-- NEW LOOP
                    
                    results = SimulationRunnerCPP.run_simulation(
                        self.config,
                        strike_=strike,                           # <-- DYNAMIC
                        maturity_=maturity,
                        seed_=42,
                        verbose_=False
                    )
                    
                    eu_call_price = results.european_call_prices[asset_idx]
                    
                    iv = self.black_scholes_price_to_iv(
                        eu_call_price,
                        S=S0,
                        K=strike,                                 # <-- DYNAMIC
                        T=maturity,
                        r=self.config.risk_free_rate,
                        option_type='call'
                    )
                    
                    sim_vol_surface[i, j, asset_idx] = iv         # <-- INDIVIDUAL ASSIGNMENT
    
        return sim_vol_surface
    
    def objective_function(self, params):
        """
        Loss function to minimize: MSE between simulated and market vol surfaces.
        
        @param params: [kappa, theta, xi, gamma, rho]
        @return: scalar error (lower is better)
        """
        self.call_count += 1
        
        try:
            sim_vol_surface = self.get_simulated_vol_surface(params)
            
            # Compute mean squared error
            error = np.mean((sim_vol_surface - self.market_vol_surface) ** 2)  # MSE across all assets
            
            print(f"[Call {self.call_count}] kappa={params[0]:.3f}, theta={params[1]:.3f}, "
                  f"xi={params[2]:.3f}, gamma={params[3]:.3f}, rho={params[4]:.3f} → Error: {error:.8f}")
            
            return error
        except Exception as e:
            print(f"[Call {self.call_count}] Simulation failed: {e}")
            return 1e10
    
    def calibrate(self, initial_guess, param_bounds):
        """
        Run Bayesian Optimization (Gaussian Process) to find best-fit parameters.
        
        @param initial_guess: [kappa_0, theta_0, xi_0, gamma_0, rho_0]
        @param param_bounds: [(kappa_min, kappa_max), (theta_min, theta_max), ...]
        @return: optimized parameters
        """
        print("\n" + "="*80)
        print("STARTING BAYESIAN VOLATILITY SURFACE CALIBRATION")
        print("="*80)
        print(f"Market vol surface shape: {self.market_vol_surface.shape}")
        print(f"Strikes: {self.market_strikes}")
        print(f"Maturities: {self.market_maturities}")
        print(f"Initial guess: kappa={initial_guess[0]}, theta={initial_guess[1]}, "
              f"xi={initial_guess[2]}, gamma={initial_guess[3]}, rho={initial_guess[4]}")
        print("="*80 + "\n")
        
        # Convert bounds to skopt Real dimensions for clarity
        dimensions = [Real(b[0], b[1]) for b in param_bounds]
        
        # Execute Gaussian Process minimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            x0=initial_guess,        # Feed it our starting heuristic
            n_calls=75,              # Total number of C++ evaluations allowed
            n_random_starts=15,      # Initial random exploration before GP takes over
            acq_func='EI',           # Expected Improvement (standard for this)
            noise='gaussian',        # CRITICAL: Tells the GP the MC simulator has noise
            verbose=True,            # Prints generation progress
            random_state=42          # For reproducibility
        )
        
        print("\n" + "="*80)
        print("CALIBRATION COMPLETE")
        print("="*80)
        print(f"Total function evaluations: {self.call_count}")
        print(f"\nOptimal parameters:")
        print(f"  kappa   = {result.x[0]:.6f}")
        print(f"  theta   = {result.x[1]:.6f}")
        print(f"  xi      = {result.x[2]:.6f}")
        print(f"  gamma   = {result.x[3]:.6f}")
        print(f"  rho     = {result.x[4]:.6f}")
        print(f"\nFinal best error: {result.fun:.8f}")
        print("="*80)
        
        return result.x


class SimulationConfig:
    """Python wrapper for C++ SimulationConfig - holds parameters to pass to exe"""
    
    def __init__(self):
        self.num_assets = 11
        self.num_paths = 2000
        self.num_steps = 252
        self.dt = 1.0 / 252.0
        self.risk_free_rate = 0.04
        self.S0 = [100.0] * 11
        self.kappa = [2.0] * 11
        self.theta = [-3.2] * 11
        self.xi = [0.6] * 11
        self.gamma = [0.8] * 11
        self.rho = [-0.70] * 11


class SimulationRunnerCPP:
    """Wrapper that calls the compiled C++ OptionPricingApp"""
    
    EXE_PATH = r"C:\Users\Dimitri\source\repos\OptionPricingApp\x64\Release\OptionPricingApp.exe"
    
    @staticmethod
    def run_simulation(config_, strike_, maturity_, seed_=42, verbose_=False):
        """
        Calls C++ executable with parameters, parses output
        
        @param config_: SimulationConfig (used for num_assets, paths, etc)
        @param strike_: Strike price
        @param maturity_: Time to maturity
        @param seed_: Random seed
        @param verbose_: Verbosity flag
        @return: SimulationResults-like object with price arrays
        """
        
        # Check if exe exists
        if not os.path.exists(SimulationRunnerCPP.EXE_PATH):
            raise FileNotFoundError(
                f"C++ executable not found: {SimulationRunnerCPP.EXE_PATH}\n"
                f"Please compile: cd ../OptionPricingApp && cmake -B build && cmake --build build --config Release"
            )
        
        # Get parameters from config
        kappa = config_.kappa[0]
        theta = config_.theta[0]
        xi = config_.xi[0]
        gamma = config_.gamma[0]
        rho = config_.rho[0]
        
        # Call C++ exe with parameters
        cmd = [
            SimulationRunnerCPP.EXE_PATH,
            str(kappa),
            str(theta),
            str(xi),
            str(gamma),
            str(rho),
            str(strike_),
            str(maturity_)
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # DEBUG: Print what the exe actually returned
            print(f"[DEBUG] Return code: {result.returncode}")
            print(f"[DEBUG] Stdout:\n{result.stdout}")
            print(f"[DEBUG] Stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                raise RuntimeError(f"C++ exe failed:\n{result.stderr}")
            
            # Parse CSV output
            df = pd.read_csv(io.StringIO(result.stdout))
            
            # Create results object
            class CppResults:
                pass
            
            results = CppResults()
            results.european_call_prices = df['EU_CALL'].values
            results.european_put_prices = df['EU_PUT'].values
            results.american_call_prices = df['AM_CALL'].values
            results.american_put_prices = df['AM_PUT'].values
            
            return results
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("C++ simulation timed out (>60s)")
        except Exception as e:
            raise RuntimeError(f"Failed to run C++ simulator: {e}\n{result.stderr if result else ''}")
            

# --- USAGE EXAMPLE with SYNTHETIC DATA ---
if __name__ == "__main__":
    
    # 1. Set up config (uses the local Python SimulationConfig class)
    config = SimulationConfig()
    config.num_assets = 11
    config.num_paths = 2000   # Small for testing
    config.num_steps = 252
    config.dt = 1.0 / 252.0
    config.risk_free_rate = 0.04
    config.S0 = np.full(11, 100.0).tolist()  # Convert to list
    config.kappa = np.full(11, 2.0).tolist()
    config.theta = np.full(11, -3.2).tolist()
    config.xi = np.full(11, 0.6).tolist()
    config.gamma = np.full(11, 0.8).tolist()
    config.rho = np.full(11, -0.70).tolist()
    
    # Skip the matrix loading for now (C++ exe has them hardcoded)
    # We'll just use the default config
    
    # 2. Define assets and strikes/maturities
    assets = ['AAPL', 'CRUS', 'SWKS', 'BBY', 'MU', 'QRVO', 'NVDA', 'SMCI', 'MPWR', 'AVT', 'AMAT']
    market_strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    market_maturities = np.array([0.25, 0.5, 1.0])
    
    # 3. Create synthetic vol surface (7 strikes x 3 maturities x 11 assets)
    # Each asset has slightly different vol levels to make it realistic
    num_strikes = len(market_strikes)
    num_maturities = len(market_maturities)
    num_assets = len(assets)
    
    # Base vol surface (volatility smile pattern)
    base_vol = np.array([
        [0.32, 0.30, 0.28],  # Strike 85 (OTM put, highest vol)
        [0.28, 0.26, 0.24],  # Strike 90
        [0.24, 0.22, 0.21],  # Strike 95
        [0.20, 0.19, 0.18],  # Strike 100 (ATM, lowest vol)
        [0.22, 0.21, 0.20],  # Strike 105
        [0.26, 0.25, 0.24],  # Strike 110
        [0.31, 0.29, 0.27]   # Strike 115 (OTM call, high vol)
    ])
    
    # Add asset-specific variation (some assets more volatile than others)
    asset_vol_multipliers = np.array([1.0, 1.15, 1.25, 0.95, 1.35, 1.20, 1.40, 1.50, 1.25, 0.90, 1.30])
    
    market_vol_surface = np.zeros((num_strikes, num_maturities, num_assets))
    for asset_idx in range(num_assets):
        market_vol_surface[:, :, asset_idx] = base_vol * asset_vol_multipliers[asset_idx]
    
    print(f"Synthetic vol surface shape: {market_vol_surface.shape}")
    print(f"Strikes: {market_strikes}")
    print(f"Maturities: {market_maturities}")
    print(f"Assets: {assets}\n")
    
    # 4. Create calibrator
    calibrator = VolatilitySurfaceCalibrator(config, market_vol_surface, market_strikes, market_maturities, assets)
    
    # 5. Calibrate
    initial_guess = [2.0, -3.2, 0.6, 0.8, -0.70]
    bounds = [
        (0.1, 5.0),       # kappa
        (-4.0, -0.5),     # theta
        (0.1, 2.0),       # xi
        (0.0, 2.0),       # gamma
        (-1.0, 0.0)       # rho
    ]
    
    optimal_params = calibrator.calibrate(initial_guess, bounds)
    
    # 6. Save results
    np.save("calibrated_params.npy", optimal_params)
    print("\nCalibrated parameters saved to calibrated_params.npy")