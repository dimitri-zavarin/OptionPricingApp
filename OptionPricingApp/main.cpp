#include <iostream>
#include <Eigen/Dense>
#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"

// Templated execution wrapper
template <typename PricerType>
Eigen::VectorXd executePricing(const PricerType& pricer, double strike, bool is_call) {
    return pricer.price(strike, is_call);
}

int main() {
    std::cout << "--- Initializing Synthetic 3-Asset Heston Network ---" << std::endl;

    SimulationConfig config;

    // 1. Core Dimensions & Market Data
    config.num_assets = 3;
    config.num_paths = 50000;      // 50k paths for stable Monte Carlo convergence
    config.num_steps = 252;        // 1 Year to maturity
    config.dt = 1.0 / 252.0;       // Daily time steps
    config.risk_free_rate = 0.05;  // 5% interest rate

    // 2. Initialize N-Dimensional Vectors
    config.S0 = Eigen::VectorXd::Constant(config.num_assets, 100.0); // All start at $100
    config.q = Eigen::VectorXd::Zero(config.num_assets);             // No dividends for simplicity

    // 3. Spatiotemporal Heston Parameters
    config.kappa = Eigen::VectorXd::Constant(config.num_assets, 2.0);   // Mean-reversion speed
    config.theta = Eigen::VectorXd::Constant(config.num_assets, -2.0);  // Baseline target
    config.gamma = Eigen::VectorXd::Constant(config.num_assets, 0.5);   // Network spillover intensity
    config.xi = Eigen::VectorXd::Constant(config.num_assets, 0.3);      // Volatility of volatility
    config.rho = Eigen::VectorXd::Constant(config.num_assets, -0.7);    // Steep equity leverage effect
    config.X0 = Eigen::VectorXd::Constant(config.num_assets, -2.0);     // Start at equilibrium

    // 4. The Synthetic Network Topology Matrix (W)
    // Row-normalized: each asset is equally influenced by its two peers
    config.W = Eigen::MatrixXd::Zero(config.num_assets, config.num_assets);
    config.W << 0.0, 0.5, 0.5,
        0.5, 0.0, 0.5,
        0.5, 0.5, 0.0;

    // 5. Validation Check
    if (!config.validate()) {
        std::cerr << "Fatal Error: Simulation Config Validation Failed!" << std::endl;
        return -1;
    }

    std::cout << "Configuration Validated. Running " << config.num_paths
        << " paths..." << std::endl;

    // 6. Run the Multi-Asset Simulator
    NetworkSimulator simulator(config, 42); // Seeded for reproducibility
    simulator.simulate();

    std::cout << "Simulation Complete. Calculating European Option Prices..." << std::endl;

    // 7. Instantiate the Pricer
    EuropeanPricer euro_pricer(config, simulator.getAssetPaths());

    // 8. Execute Pricing (At-The-Money Calls, Strike = 100)
    double strike = 100.0;
    Eigen::VectorXd call_prices = executePricing<EuropeanPricer>(euro_pricer, strike, true);

    // 9. Execute Pricing (At-The-Money Puts, Strike = 100)
    Eigen::VectorXd put_prices = executePricing<EuropeanPricer>(euro_pricer, strike, false);

    // Output Results
    std::cout << "\n--- Pricing Results (T = 1.0 Year, K = 100) ---" << std::endl;
    std::cout << "Call Options:\n" << call_prices << "\n" << std::endl;
    std::cout << "Put Options:\n" << put_prices << "\n" << std::endl;

    return 0;
}