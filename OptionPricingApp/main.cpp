#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <Eigen/Dense>

#include "simulation_config.h"
#include "option.h"
#include "network_simulator.h"
#include "network_pricers.h"
#include "config_loader.h"

/**
 * @brief Executes a single simulation regime, computes option prices, and prints diagnostics.
 */
void run_regime(const std::string& regime_name, const SimulationConfig& config) {
    std::cout << "\n==================================================================================================" << std::endl;
    std::cout << " REGIME CONFIGURATION: " << regime_name << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    // --- Print Key Parameters ---
    std::cout << "--- CRUCIAL STRUCTURAL PARAMETERS ---\n";
    std::cout << "Ticker    S0        Div (q)   Kappa     Theta     X0        Gamma (Sensitivity)\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < config.num_assets; ++i) {
        printf("%-9s %-9.1f %-9.2f %-9.1f %-9.2f %-9.2f %-9.2f\n",
            config.tickers[i].c_str(), config.S0(i), config.q(i), config.kappa(i),
            config.theta(i), config.X0(i), config.gamma(i));
    }
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    // --- 1. Simulate Paths ---
    std::cout << "\n[System] Spinning up Heston Network Simulator (" << config.num_paths << " paths)...\n";
    NetworkSimulator sim(config, 42); // Fixed seed for reproducible diagnostics
    sim.simulate();

    const auto& asset_paths = sim.get_asset_paths();
    const auto& variance_paths = sim.get_log_variance_paths();

    // --- 2. Initialize Pricers ---
    EuropeanPricer<EuropeanCall> eu_call_pricer(config, asset_paths);
    EuropeanPricer<EuropeanPut>  eu_put_pricer(config, asset_paths);

    AmericanPricer<AmericanCall> am_call_pricer(config, asset_paths, variance_paths, config.W);
    AmericanPricer<AmericanPut>  am_put_pricer(config, asset_paths, variance_paths, config.W);

    // Using ATM strike for all assets (Assuming S0 = 100 for all)
    double strike = 100.0;
    double maturity = 1.0;

    // Calculate European baseline prices to extract the early-exercise premium
    EuropeanCall eu_call(strike, maturity);
    EuropeanPut  eu_put(strike, maturity);
    Eigen::VectorXd euro_calls = eu_call_pricer.price(eu_call);
    Eigen::VectorXd euro_puts = eu_put_pricer.price(eu_put);

    // --- 3. Run Longstaff-Schwartz American Regressions ---
    std::cout << "\n[System] Executing Spatiotemporal Longstaff-Schwartz Regressions...\n";

    std::vector<double> amer_calls(config.num_assets);
    std::vector<double> amer_puts(config.num_assets);

    for (int i = 0; i < config.num_assets; ++i) {
        AmericanCall am_call(strike, maturity);
        AmericanPut  am_put(strike, maturity);

        amer_calls[i] = am_call_pricer.price_asset_option(i, am_call);
        amer_puts[i] = am_put_pricer.price_asset_option(i, am_put);
    }

    // --- 4. Print Final Volatility Surface ---
    std::cout << "\n--- DERIVATIVES SURFACE VALUATIONS (T = 1.0000 Year) ---\n";
    std::cout << "Ticker  Euro Call      Raw Amer Call  Call Prem      Euro Put       Raw Amer Put   Put Prem\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < config.num_assets; ++i) {
        double c_prem = amer_calls[i] - euro_calls(i);
        double p_prem = amer_puts[i] - euro_puts(i);

        printf("%-7s %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f\n",
            config.tickers[i].c_str(), euro_calls(i), amer_calls[i], c_prem,
            euro_puts(i), amer_puts[i], p_prem);
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "          SPATIOTEMPORAL GRAPH LAPLACIAN ANALYSIS\n";
    std::cout << "          (12-NODE BASKET W/ CSV MATRIX LOADING)\n";
    std::cout << "=================================================================\n";

    int num_assets = 12;
    std::vector<std::string> tickers = {
        "NVDA", "AMD", "AAPL", "MSFT", "GOOG", "AMZN",
        "JPM", "GS", "XOM", "META", "TSLA", "NFLX"
    };

    // --- Base Configuration ---
    SimulationConfig config;
    config.num_assets = num_assets;
    config.num_paths = 50000; // Lowered from 150k to 50k for speed during 12-asset testing
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.risk_free_rate = 0.04;
    config.tickers = tickers;

    // Standardize initial Heston parameters across all 12 assets
    config.S0 = Eigen::VectorXd::Constant(num_assets, 100.0);
    config.kappa = Eigen::VectorXd::Constant(num_assets, 2.0);
    config.theta = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.X0 = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
    config.xi = Eigen::VectorXd::Constant(num_assets, 0.60);    // Vol of vol
    config.rho = Eigen::VectorXd::Constant(num_assets, -0.70);  // Leverage effect

    // Set Dividends (q) - Assigning basic yields to specific tickers
    config.q = Eigen::VectorXd::Zero(num_assets);
    config.q(2) = 0.05; // AAPL
    config.q(3) = 0.08; // MSFT
    config.q(6) = 0.20; // JPM
    config.q(8) = 0.35; // XOM

    // ========================================================================
    // FILE I/O: Load Python-Generated Matrices
    // ========================================================================
    std::string cholesky_file = "cholesky_L_matrix.csv";
    std::string w_matrix_file = "calibrated_W_matrix.csv"; // Change to calibrated_W_matrix.csv later

    if (!ConfigLoader::load_matrices_into_config(config, cholesky_file, w_matrix_file)) {
        std::cerr << "\n[FATAL] Exiting simulation due to matrix loading failure.\n";
        return 1;
    }

    // ========================================================================
    // REGIME 1: Baseline Market
    // Uses standard volatility starts and lets the imported network run neutrally.
    // ========================================================================
    run_regime("1. BASELINE MARKET (LOADED TOPOLOGY)", config);

    // ========================================================================
    // REGIME 2: Organic Network Contagion Shock
    // Instead of hardcoding a cascade, we shock ONE asset (NVDA).
    // The imported W matrix will automatically route this shock to connected assets!
    // ========================================================================
    SimulationConfig config_cascade = config;

    // Inject massive volatility shock to NVDA (Index 0)
    config_cascade.X0(0) = -1.00;     // Instant volatility spike
    config_cascade.theta(0) = -1.00;  // Sustain the volatility

    run_regime("2. NVDA SHOCK PROPAGATING VIA LOADED W MATRIX", config_cascade);

    return 0;
}