#include <iostream>
#include <Eigen/Dense>
#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"

// Templated execution wrapper
template <typename PricerType>
Eigen::VectorXd execute_pricing(const PricerType& pricer, double strike, bool is_call) {
    return pricer.price(strike, is_call);
}

int main() {
    std::cout << "--- Initializing Synthetic 3-Asset Heston Network ---" << std::endl;

    SimulationConfig config;

    config.num_assets = 3;
    config.num_paths = 50000;
    config.num_steps = 252;
    config.dt = 1.0 / 252.0;
    config.risk_free_rate = 0.05;

    config.S0 = Eigen::VectorXd::Constant(config.num_assets, 100.0);
    config.q = Eigen::VectorXd::Zero(config.num_assets);

    config.kappa = Eigen::VectorXd::Constant(config.num_assets, 2.0);
    config.theta = Eigen::VectorXd::Constant(config.num_assets, -2.0);
    config.gamma = Eigen::VectorXd::Constant(config.num_assets, 0.5);
    config.xi = Eigen::VectorXd::Constant(config.num_assets, 0.3);
    config.rho = Eigen::VectorXd::Constant(config.num_assets, -0.7);
    config.X0 = Eigen::VectorXd::Constant(config.num_assets, -2.0);

    config.W = Eigen::MatrixXd::Zero(config.num_assets, config.num_assets);
    config.W << 0.0, 0.5, 0.5,
        0.5, 0.0, 0.5,
        0.5, 0.5, 0.0;

    if (!config.validate()) {
        std::cerr << "Fatal Error: Simulation Config Validation Failed!" << std::endl;
        return -1;
    }

    std::cout << "Configuration Validated. Running " << config.num_paths << " paths..." << std::endl;

    NetworkSimulator simulator(config, 42);
    simulator.simulate();

    std::cout << "Simulation Complete. Calculating European Option Prices..." << std::endl;

    EuropeanPricer euro_pricer(config, simulator.get_asset_paths());

    double strike = 100.0;
    Eigen::VectorXd call_prices = execute_pricing<EuropeanPricer>(euro_pricer, strike, true);
    Eigen::VectorXd put_prices = execute_pricing<EuropeanPricer>(euro_pricer, strike, false);

    std::cout << "\n--- Pricing Results (T = 1.0 Year, K = 100) ---" << std::endl;
    std::cout << "Call Options:\n" << call_prices << "\n" << std::endl;
    std::cout << "Put Options:\n" << put_prices << "\n" << std::endl;

    return 0;
}