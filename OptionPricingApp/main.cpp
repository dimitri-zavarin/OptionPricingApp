#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <Eigen/Dense>

#include "simulation_config.h"
#include "option.h"
#include "network_simulator.h"
#include "network_pricers.h"
#include "config_loader.h"

/**
 * @brief Executes a single simulation regime, computes option prices, and prints diagnostics.
 */
void run_regime(const std::string& regime_name, const SimulationConfig& config, bool enable_diagnostics = true) {
    std::cout << "\n==================================================================================================" << std::endl;
    std::cout << " REGIME CONFIGURATION: " << regime_name << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    // --- Print Structural Parameters ---
    std::cout << "--- HESTON NETWORK PARAMETER MATRIX ---\n";
    std::cout << "Ticker    S0        q         Kappa     Theta     X0        Gamma     Xi        Rho\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < config.num_assets; ++i) {
        printf("%-9s %-9.1f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f\n",
            config.tickers[i].c_str(), config.S0(i), config.q(i), config.kappa(i),
            config.theta(i), config.X0(i), config.gamma(i), config.xi(i), config.rho(i));
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

    // Pass enable_diagnostics (verbose) to test standard errors, R^2, and ITM paths
    AmericanPricer<AmericanCall> am_call_pricer(config, asset_paths, variance_paths, config.W, enable_diagnostics);
    AmericanPricer<AmericanPut>  am_put_pricer(config, asset_paths, variance_paths, config.W, enable_diagnostics);

    double strike = 100.0;
    double maturity = 1.0;

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
    std::cout << "\n--- DERIVATIVES SURFACE VALUATIONS (T = 1.0000 Year, Strike = 100.0) ---\n";
    std::cout << "Ticker  Euro Call      Amer Call      Call Prem      Euro Put       Amer Put       Put Prem\n";
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
    config.num_paths = 50000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.risk_free_rate = 0.04;
    config.tickers = tickers;

    // Standardize initial baseline parameters
    config.S0 = Eigen::VectorXd::Constant(num_assets, 100.0);
    config.kappa = Eigen::VectorXd::Constant(num_assets, 2.0);
    config.theta = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.X0 = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
    config.xi = Eigen::VectorXd::Constant(num_assets, 0.60);
    config.rho = Eigen::VectorXd::Constant(num_assets, -0.70);

    config.q = Eigen::VectorXd::Zero(num_assets);
    config.q(2) = 0.05; // AAPL
    config.q(3) = 0.08; // MSFT
    config.q(6) = 0.20; // JPM
    config.q(8) = 0.35; // XOM

    // --- File I/O: Load Python Matrices ---
    std::string cholesky_file = "cholesky_L_matrix.csv";
    std::string w_matrix_file = "calibrated_W_matrix.csv";

    if (!ConfigLoader::load_matrices_into_config(config, cholesky_file, w_matrix_file)) {
        std::cerr << "\n[FATAL] Exiting simulation due to matrix loading failure.\n";
        return 1;
    }

    // ========================================================================
    // REGIME 1: Baseline Market (Verbose Diagnostics = true)
    // ========================================================================
    run_regime("1. BASELINE MARKET (TESTING DIAGNOSTICS)", config, true);

    // ========================================================================
    // REGIME 2: Organic Network Contagion Shock
    // ========================================================================
    SimulationConfig config_cascade = config;
    config_cascade.X0(0) = -1.00;     // NVDA Volatility Spike
    config_cascade.theta(0) = -1.00;

    run_regime("2. NVDA SHOCK PROPAGATING VIA LOADED W MATRIX", config_cascade, false);

    // ========================================================================
    // REGIME 3: Fully Heterogeneous (Jumbled) Parameter Basket
    // Tests how the engine and LSMC regressors react to non-uniform parameters
    // ========================================================================
    SimulationConfig config_jumbled = config;
    std::mt19937 rng(12345); // Fixed seed for reproducible jumbling

    std::uniform_real_distribution<double> dist_s0(80.0, 120.0);
    std::uniform_real_distribution<double> dist_kappa(0.5, 4.0);
    std::uniform_real_distribution<double> dist_theta(-4.5, -1.5);
    std::uniform_real_distribution<double> dist_gamma(0.1, 1.5);
    std::uniform_real_distribution<double> dist_xi(0.2, 0.9);
    std::uniform_real_distribution<double> dist_rho(-0.9, -0.2);

    for (int i = 0; i < num_assets; ++i) {
        config_jumbled.S0(i) = dist_s0(rng);
        config_jumbled.kappa(i) = dist_kappa(rng);
        config_jumbled.theta(i) = dist_theta(rng);
        config_jumbled.X0(i) = config_jumbled.theta(i); // Start at idiosyncratic long-run target
        config_jumbled.gamma(i) = dist_gamma(rng);
        config_jumbled.xi(i) = dist_xi(rng);
        config_jumbled.rho(i) = dist_rho(rng);
    }

    run_regime("3. FULLY HETEROGENEOUS (JUMBLED) PARAMETER BASKET", config_jumbled, true);

    return 0;
}