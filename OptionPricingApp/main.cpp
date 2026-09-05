#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdio>
#include <Eigen/Dense>

#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"
#include "config_loader.h"
#include "simulation_runner.h"

/**
 * @brief Prints a formatted table of option prices from simulation results.
 */
void print_results(const std::string& regime_name,
                   const SimulationConfig& config_,
                   const SimulationResults& results_,
                   double strike_,
                   double maturity_) {
    std::cout << "\n==================================================================================================" << std::endl;
    std::cout << " REGIME CONFIGURATION: " << regime_name << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    // --- Print Structural Parameters ---
    std::cout << "--- HESTON NETWORK PARAMETER MATRIX ---\n";
    std::cout << "Ticker    S0        q         Kappa     Theta     X0        Gamma     Xi        Rho\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < config_.num_assets; ++i) {
        printf("%-9s %-9.1f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f\n",
            config_.tickers[i].c_str(), config_.S0(i), config_.q(i), config_.kappa(i),
            config_.theta(i), config_.X0(i), config_.gamma(i), config_.xi(i), config_.rho(i));
    }
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    // --- Print Option Prices ---
    std::cout << "\n--- DERIVATIVES SURFACE VALUATIONS (T = " << std::fixed << std::setprecision(4) 
              << maturity_ << " Year, Strike = " << strike_ << ") ---\n";
    std::cout << "Ticker  Euro Call      Amer Call      Call Prem      Euro Put       Amer Put       Put Prem\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < config_.num_assets; ++i) {
        double c_prem = results_.american_call_prices[i] - results_.european_call_prices(i);
        double p_prem = results_.american_put_prices[i] - results_.european_put_prices(i);

        printf("%-7s %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f\n",
            config_.tickers[i].c_str(),
            results_.european_call_prices(i),
            results_.american_call_prices[i],
            c_prem,
            results_.european_put_prices(i),
            results_.american_put_prices[i],
            p_prem);
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "          SPATIOTEMPORAL GRAPH LAPLACIAN ANALYSIS\n";
    std::cout << "       (11-NODE TECH SUPPLY CHAIN W/ CSV MATRIX LOADING)\n";
    std::cout << "=================================================================\n";

    int num_assets = 11;
    std::vector<std::string> tickers = {
        "AAPL", "CRUS", "SWKS", "BBY", "MU", "QRVO", "NVDA", "SMCI", "MPWR", "AVT", "AMAT"
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
    config.q(0) = 0.05;   // AAPL
    config.q(4) = 0.02;   // MU
    config.q(6) = 0.00;   // NVDA
    config.q(10) = 0.03;  // AMAT

    // --- File I/O: Load Python Matrices ---
    std::string cholesky_file = "cholesky_L_matrix.csv";
    std::string w_matrix_file = "calibrated_W_matrix.csv";

    if (!ConfigLoader::load_matrices_into_config(config, cholesky_file, w_matrix_file)) {
        std::cerr << "\n[FATAL] Exiting simulation due to matrix loading failure.\n";
        return 1;
    }

    // ========================================================================
    // REGIME 1: Baseline Market (Stable parameters, no localized shocks)
    // ========================================================================
    SimulationResults baseline_results = SimulationRunner::run_simulation(config, 100.0, 1.0, 42, false);
    print_results("1. BASELINE MARKET (STABLE PARAMETERS)", config, baseline_results, 100.0, 1.0);

    // ========================================================================
    // REGIME 2: The Volatility Injection Case (Apple Anchors Down)
    // ========================================================================
    SimulationConfig config_injection = config;
    config_injection.X0(0) = -1.00;     // AAPL massive immediate volatility spike
    config_injection.theta(0) = -1.00;  // AAPL massive long-term structural volatility

    SimulationResults injection_results = SimulationRunner::run_simulation(config_injection, 100.0, 1.0, 42, true);
    print_results("2. VOLATILITY INJECTION (AAPL SHOCKS THE NETWORK)", config_injection, injection_results, 100.0, 1.0);

    return 0;
}