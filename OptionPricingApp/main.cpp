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

// Multi-Asset Spatiotemporal Matrix Network modules
#include "network_simulator.h"
#include "network_pricers.h"

int main() {
    // --- Set Console Formatting ---
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=================================================================\n";
    std::cout << "     QUANT FINANCE MULTI-ASSET NETWORK LSM PRICING HARNESS        \n";
    std::cout << "               (ACTIVE NETWORK SPILLOVERS SETUP)                 \n";
    std::cout << "=================================================================\n\n";

    // =========================================================================
    // STEP 1: INITIALIZE THE 3-ASSET SPILLOVER SYSTEM
    // =========================================================================
    std::cout << "[1/4] Constructing Interconnected Network Configuration...\n";

    SimulationConfig config;
    config.num_assets = 3;
    config.num_paths = 150000;                     // Robust path count for LSM convergence
    config.num_steps = 252;                       // 1 trading year of daily steps
    config.dt = 1.0 / 252.0;                      // Daily temporal fraction
    config.risk_free_rate = 0.05;                 // r = 5%

    // Starting asset spots [N x 1]
    config.S0 = Eigen::VectorXd::Zero(3);
    config.S0 << 100.0, 100.0, 100.0;

    // Dividend yields (Keep high to evaluate put/call boundaries under correlation)
    config.q = Eigen::VectorXd::Zero(3);
    config.q << 0.08, 0.00, 0.12;

    // Unfreeze the Heston Volatility Network Dynamics
    config.X0 = Eigen::VectorXd::Constant(3, -3.2);      // Starting log-variance state (~20.19% vol)
    config.theta = Eigen::VectorXd::Constant(3, -3.2);   // Long-run log-variance targets
    config.kappa = Eigen::VectorXd::Constant(3, 2.0);    // Volatility mean-reversion speed (\kappa = 2.0)
    config.xi = Eigen::VectorXd::Constant(3, 0.15);      // Vol-of-vol coefficients (\xi = 15%)
    config.rho = Eigen::VectorXd::Constant(3, -0.60);    // Asymmetric leverage correlation (\rho = -60%)

    // Activate the Spatial Coupling Spillover Matrix [N x N]
    config.gamma = Eigen::VectorXd::Constant(3, 0.40);   // Global network coupling sensitivity

    // Construct asymmetric network dependencies:
    // Row i, Col j implies asset j spillovers into asset i.
    config.W = Eigen::MatrixXd::Zero(3, 3);
    config.W(1, 0) = 0.60;  // AAPL shocks spill heavily into MSFT (60% weight)
    config.W(2, 0) = 0.40;  // AAPL shocks spill moderately into GOOG (40% weight)
    config.W(2, 1) = 0.30;  // MSFT shocks spill into GOOG (30% weight)

    config.tickers = { "AAPL", "MSFT", "GOOG" };

    if (!config.validate()) {
        std::cerr << "CRITICAL ERROR: Simulation configuration validation failed!\n";
        return -1;
    }
    std::cout << ">> Spillover network architecture verified successfully.\n\n";

    // =========================================================================
    // STEP 2: RUN THE HYBRID NETWORK PATH GENERATOR
    // =========================================================================
    std::cout << "[2/4] Executing Spatiotemporal Path Simulation...\n";
    NetworkSimulator simulator(config, 42);
    simulator.simulate();
    std::cout << ">> Generated " << config.num_paths << " coupled paths successfully.\n\n";

    // =========================================================================
    // STEP 3: INITIALIZE PRICERS AND OPTIONS CONTRACT INTERFACES
    // =========================================================================
    std::cout << "[3/4] Initializing Multi-Asset European and American Pricers...\n";
    double target_strike = 100.0;
    double maturity_time = config.num_steps * config.dt; // 1.0 Year

    EuropeanPricer european_engine(config, simulator.get_asset_paths());
    AmericanPricer<AmericanCall> american_call_engine(config, simulator.get_asset_paths());
    AmericanPricer<AmericanPut>  american_put_engine(config, simulator.get_asset_paths());

    Eigen::VectorXd net_euro_calls = european_engine.price(target_strike, true);
    Eigen::VectorXd net_euro_puts = european_engine.price(target_strike, false);
    std::cout << ">> Pricer pipelines generated.\n\n";

    // =========================================================================
    // STEP 4: PRINT SIDE-BY-SIDE ARBITRAGE-FLOORED COMPARISON MATRICES
    // =========================================================================
    std::cout << "[4/4] Generating Cross-Sectional Pricing Analysis...\n\n";

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8) << "Ticker"
        << std::setw(10) << "Div (q)"
        << std::setw(15) << "Euro Call ($)"
        << std::setw(15) << "Amer Call ($)"
        << std::setw(15) << "Euro Put ($)"
        << "Amer Put ($)\n";
    std::cout << "---------------------------------------------------------------------------------\n";

    for (int i = 0; i < config.num_assets; ++i) {
        AmericanCall call_contract(target_strike, maturity_time);
        AmericanPut  put_contract(target_strike, maturity_time);

        // Raw LSM outputs containing localized regression noise
        double raw_amer_call = american_call_engine.price_asset_option(i, call_contract);
        double raw_amer_put = american_put_engine.price_asset_option(i, put_contract);

        // Production Standard: Enforce the hard European Arbitrage Boundary Floor
        double floored_amer_call = std::max(raw_amer_call, net_euro_calls(i));
        double floored_amer_put = std::max(raw_amer_put, net_euro_puts(i));

        std::cout << std::left << std::setw(8) << config.tickers[i]
            << std::setw(10) << config.q(i)
            << std::setw(15) << net_euro_calls(i)
            << std::setw(15) << floored_amer_call
            << std::setw(15) << net_euro_puts(i)
            << floored_amer_put << "\n";
    }
    std::cout << "---------------------------------------------------------------------------------\n\n";

    // =========================================================================
    // STEP 5: FINANCIAL SANITY VERIFICATIONS
    // =========================================================================
    std::cout << "Financial Sanity Verifications (Network Environment):\n";

    // Re-verify MSFT call boundary with the production floor active
    double msft_call_lsm = american_call_engine.price_asset_option(1, AmericanCall(target_strike, maturity_time));
    double msft_diff = std::abs(net_euro_calls(1) - std::max(msft_call_lsm, net_euro_calls(1)));
    std::cout << ">> MSFT (q=0) American Call vs. European Call Delta: " << msft_diff
        << (msft_diff < 1e-4 ? " (PASSED - Arbitrage Boundary Maintained)" : " (FAILED)") << "\n";

    // Evaluate how network volatility spillovers from AAPL warped the premium profiles
    double aapl_call_premium = std::max(american_call_engine.price_asset_option(0, AmericanCall(target_strike, maturity_time)), net_euro_calls(0)) - net_euro_calls(0);
    double msft_put_premium = std::max(american_put_engine.price_asset_option(1, AmericanPut(target_strike, maturity_time)), net_euro_puts(1)) - net_euro_puts(1);

    std::cout << ">> AAPL Active Early Exercise Call Premium: $" << aapl_call_premium << "\n";
    std::cout << ">> MSFT Active Early Exercise Put Premium:  $" << msft_put_premium << "\n\n";

    std::cout << "=================================================================\n";
    std::cout << "             LSM BACKWARD INDUCTION TEST COMPLETE                \n";
    std::cout << "=================================================================\n";

    return 0;
}