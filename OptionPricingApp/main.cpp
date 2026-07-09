#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <Eigen/Dense>

#include "simulation_config.h"
#include "option.h"
#include "network_simulator.h"
#include "network_pricers.h"

void run_regime(const std::string& regime_name, const SimulationConfig& config,
    const std::vector<std::string>& tickers,
    const Eigen::MatrixXd& W) {

    std::cout << "\n==================================================================================================" << std::endl;
    std::cout << " REGIME CONFIGURATION: " << regime_name << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    // --- Print Network Topology ---
    std::cout << "--- CRUCIAL STRUCTURAL PARAMETERS ---\n";
    std::cout << "Ticker    S0        Div (q)   Kappa     Theta     X0        Gamma (Sensitivity)\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < config.num_assets; ++i) {
        printf("%-9s %-9.1f %-9.2f %-9.1f %-9.2f %-9.2f %-9.2f\n",
            tickers[i].c_str(), config.S0(i), config.q(i), config.kappa(i),
            config.theta(i), config.X0(i), config.gamma(i));
    }

    std::cout << "\n  Spatial Topology Matrix (W):\n              ";
    for (const auto& t : tickers) std::cout << t << "    ";
    std::cout << "\n";
    for (int i = 0; i < config.num_assets; ++i) {
        std::cout << "    " << tickers[i] << "  [ ";
        for (int j = 0; j < config.num_assets; ++j) {
            printf("%7.2f ", W(i, j));
        }
        std::cout << " ]\n";
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

    AmericanPricer<AmericanCall> am_call_pricer(config, asset_paths, variance_paths, W);
    AmericanPricer<AmericanPut>  am_put_pricer(config, asset_paths, variance_paths, W);

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
            tickers[i].c_str(), euro_calls(i), amer_calls[i], c_prem,
            euro_puts(i), amer_puts[i], p_prem);
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "          SPATIOTEMPORAL GRAPH LAPLACIAN ANALYSIS\n";
    std::cout << "               (5-NODE CASCADING NETWORK)\n";
    std::cout << "=================================================================\n";

    int num_assets = 5;
    std::vector<std::string> tickers = { "NVDA", "AMD", "AAPL", "MSFT", "GOOG" };

    // --- Base Configuration ---
    SimulationConfig config;
    config.num_assets = num_assets;
    config.num_paths = 150000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.risk_free_rate = 0.04;
    config.tickers = tickers;  // <-- POPULATE THE TICKERS VECTOR

    // Standardize initial parameters across all 5 assets
    config.S0 = Eigen::VectorXd::Constant(num_assets, 100.0);
    config.kappa = Eigen::VectorXd::Constant(num_assets, 2.0);
    config.theta = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.X0 = Eigen::VectorXd::Constant(num_assets, -3.20);
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
    config.xi = Eigen::VectorXd::Constant(num_assets, 0.60);    // Vol of vol
    config.rho = Eigen::VectorXd::Constant(num_assets, -0.70);  // Leverage effect

    // Set Dividends (q)
    config.q = Eigen::VectorXd(num_assets);
    config.q << 0.00, 0.00, 0.08, 0.00, 0.12;


    // ========================================================================
    // REGIME 1: The Cascading Supply Chain Shock
    // NVDA acts as the source, spilling into AMD, which cascades into Big Tech.
    // ========================================================================
    SimulationConfig config_cascade = config;  // This now inherits tickers
    config_cascade.X0(0) = -1.00;
    config_cascade.theta(0) = -1.00;

    Eigen::MatrixXd W_cascade = Eigen::MatrixXd::Zero(num_assets, num_assets);
    // Row 0 (NVDA): Pure standalone source
    W_cascade(0, 0) = 1.00;
    // Row 1 (AMD): 50% tied to NVDA's variance, 50% local
    W_cascade(1, 0) = 0.50; W_cascade(1, 1) = 0.50;
    // Row 2 (AAPL): Heavy hardware reliance on NVDA/AMD
    W_cascade(2, 0) = 0.40; W_cascade(2, 1) = 0.40; W_cascade(2, 2) = 0.20;
    // Row 3 (MSFT): Software, slower absorption of the hardware shock
    W_cascade(3, 2) = 0.60; W_cascade(3, 3) = 0.40;
    // Row 4 (GOOG): Absorbs from MSFT and AAPL
    W_cascade(4, 2) = 0.30; W_cascade(4, 3) = 0.30; W_cascade(4, 4) = 0.40;

    config_cascade.W = W_cascade;
    run_regime("1. CASCADING SUPPLY CHAIN SHOCK", config_cascade, tickers, W_cascade);


    // ========================================================================
    // REGIME 2: The Global Volatility Sink (The Black Hole)
    // MSFT acts as a massive variance sink. All other assets tether to it.
    // ========================================================================
    SimulationConfig config_sink = config;  // This now inherits tickers
    config_sink.kappa(3) = 8.0;     // MSFT mean-reverts violently
    config_sink.theta(3) = -4.00;   // MSFT baseline variance is crushed
    config_sink.X0(3) = -4.00;

    Eigen::MatrixXd W_sink = Eigen::MatrixXd::Zero(num_assets, num_assets);
    for (int i = 0; i < num_assets; ++i) {
        if (i == 3) {
            W_sink(i, i) = 1.00; // MSFT is standalone
        }
        else {
            W_sink(i, i) = 0.20; // 20% Local
            W_sink(i, 3) = 0.80; // 80% Routed to Sink
        }
    }

    config_sink.W = W_sink;
    run_regime("2. GLOBAL VOLATILITY SINK (MSFT BLACK HOLE)", config_sink, tickers, W_sink);

    return 0;
}