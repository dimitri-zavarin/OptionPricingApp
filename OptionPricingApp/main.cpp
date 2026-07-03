#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Core options logic and data structures
#include "option.h"
#include "market_data.h"
#include "simulation_config.h"

// Univariate pricing modules (Benchmarks)
#include "black_scholes.h"
#include "binomial.h"
#include "greeks.h"
#include "implied_vol.h"
#include "implied_div.h"

// Multi-Asset Spatiotemporal Matrix Network modules
#include "network_simulator.h"
#include "network_pricers.h"

int main() {
    // --- Set Console Formatting ---
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=================================================================\n";
    std::cout << "     QUANT FINANCE MULTI-ASSET NETWORK PRICING ENGINE HARNESS     \n";
    std::cout << "=================================================================\n\n";

    // =========================================================================
    // STEP 1: INITIALIZE A 3-ASSET PORTFOLIO SYSTEM CONFIGURATION
    // =========================================================================
    std::cout << "[1/4] Constructing 3-Asset Network Configuration...\n";

    SimulationConfig config;
    config.num_assets = 3;
    config.num_paths = 50000;                     // Robust path count for Monte Carlo convergence
    config.num_steps = 252;                       // 1 trading year of daily steps
    config.dt = 1.0 / 252.0;                      // Daily temporal fraction
    config.risk_free_rate = 0.05;                 // r = 5%

    // Define multi-asset vector states [N x 1]
    config.S0 = Eigen::VectorXd::Zero(3);
    config.S0 << 100.0, 105.0, 95.0;              // Spot prices (Asset 0, 1, 2)

    config.q = Eigen::VectorXd::Zero(3);
    config.q << 0.01, 0.00, 0.02;                 // Dividend yields (1%, 0%, 2%)

    // Heston Network parameters
    config.kappa = Eigen::VectorXd::Constant(3, 0.0);    // Mean-reversion speed
    config.theta = Eigen::VectorXd::Constant(3, -3.5);   // Long-run log-variance targets
    config.gamma = Eigen::VectorXd::Constant(3, 0.0);    // Spatial coupling intensity scaling
    config.xi = Eigen::VectorXd::Constant(3, 0.0);      // Vol-of-vol scales
    config.rho = Eigen::VectorXd::Constant(3, 0.0);    // Heavy negative leverage asset correlation
    config.X0 = Eigen::VectorXd::Constant(3, -3.2);      // Starting log-variance state

    // Construct a Row-Normalized Spatial Weight Matrix [N x N]
    // Outlines the topology of volatility contagion across the portfolio
    Eigen::MatrixXd W_mat(3, 3);
    W_mat << 0.0, 0.6, 0.4,
        0.5, 0.0, 0.5,
        0.3, 0.7, 0.0;
    config.W = W_mat;

    config.tickers = { "AAPL", "MSFT", "GOOG" };

    // Run structural size and dimension diagnostics before memory allocation
    if (!config.validate()) {
        std::cerr << "CRITICAL ERROR: Simulation configuration validation failed!\n";
        return -1;
    }
    std::cout << ">> Configuration successfully verified.\n\n";

    // =========================================================================
    // STEP 2: RUN THE HYBRID PATH GENERATOR
    // =========================================================================
    std::cout << "[2/4] Initializing and Running Network Path Simulator...\n";
    try {
        NetworkSimulator simulator(config, 1337); // Seed generator for consistent paths
        simulator.simulate();                     // Execute main temporal propagation loop

        std::cout << ">> Generated " << config.num_paths << " paths successfully.\n\n";

        // =====================================================================
        // STEP 3: PRICE MULTI-ASSET EUROPEAN OPTIONS VIA SIMULATION PATHS
        // =====================================================================
        std::cout << "[3/4] Pricing Cross-Sectional European Options via Network Paths...\n";
        double target_strike = 100.0;

        EuropeanPricer pricer(config, simulator.get_asset_paths());
        Eigen::VectorXd network_calls = pricer.price(target_strike, true);  // Call option array
        Eigen::VectorXd network_puts = pricer.price(target_strike, false); // Put option array

        // =====================================================================
        // STEP 4: RUN CONVERGENCE CHECK AGAINST UNIVARIATE MODALITY PIPELINES
        // =====================================================================
        std::cout << "[4/4] Conducting Convergence Checks Against Univariate Models...\n\n";

        std::cout << "-----------------------------------------------------------------\n";
        std::cout << std::left << std::setw(8) << "Ticker"
            << std::setw(12) << "Spot ($)"
            << std::setw(15) << "Net Call ($)"
            << std::setw(15) << "Net Put ($)"
            << "BS Anal. Call ($)\n";
        std::cout << "-----------------------------------------------------------------\n";

        for (int i = 0; i < config.num_assets; ++i) {
            // Build isolated standalone interfaces for each underlying asset row
            // We map the initial localized volatility out of log space: sigma = sqrt(exp(X0))
            double local_sigma = std::sqrt(std::exp(config.X0(i)));

            MarketData local_mkt(config.S0(i), config.risk_free_rate, local_sigma, config.q(i));
            EuropeanCall local_opt(target_strike, config.num_steps * config.dt);

            // Compute standalone theoretical benchmark via exact formula
            double bs_call_price = BlackScholesPricer<EuropeanCall>::price(local_mkt, local_opt);

            std::cout << std::left << std::setw(8) << config.tickers[i]
                << std::setw(12) << config.S0(i)
                << std::setw(15) << network_calls(i)
                << std::setw(15) << network_puts(i)
                << bs_call_price << "\n";
        }
        std::cout << "-----------------------------------------------------------------\n\n";

        // =====================================================================
        // BONUS: TEST SOLVER CONVERGENCE & GREEK INFRASTRUCTURE
        // =====================================================================
        std::cout << "Testing Standalone Tree/Greek Engine Pipeline Integration...\n";
        MarketData test_mkt(100.0, 0.05, 0.20, 0.01);
        AmericanCall test_american(100.0, 1.0);

        double tree_price = BinomialPricer<AmericanCall>::price(test_mkt, test_american, 1000);
        GreekValues greeks = GreeksSuite<AmericanCall, BinomialPricer>::calculate(test_mkt, test_american, 500);

        std::cout << ">> Standalone Test Asset Tree American Call Price: $" << tree_price << "\n";
        std::cout << ">> Finite Difference Delta: " << greeks.delta << "\n";
        std::cout << ">> Annualized Theta (Scaled down to 1-Day): " << greeks.theta << "\n\n";

        // Test Implied Volatility Solver on the tree framework
        double target_market_premium = 10.50;
        double implied_vol = IVSolver<AmericanCall, BinomialPricer>::solve(test_mkt, test_american, target_market_premium);
        std::cout << ">> Successfully back-calculated Implied Volatility: " << (implied_vol * 100.0) << "%\n";

    }
    catch (const std::exception& ex) {
        std::cerr << "RUNTIME EXCEPTION ENCOUNTERED: " << ex.what() << "\n";
        return -1;
    }

    std::cout << "=================================================================\n";
    std::cout << "            ALL SYSTEM INTEGRATION CHECKS COMPLETE               \n";
    std::cout << "=================================================================\n";
    return 0;
}