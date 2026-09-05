#pragma once

#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"
#include "option.h"
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @struct SimulationResults
 * @brief Container for both American and European option prices from a single simulation.
 */
struct SimulationResults {
    Eigen::VectorXd european_call_prices;
    Eigen::VectorXd european_put_prices;
    Eigen::VectorXd american_call_prices;
    Eigen::VectorXd american_put_prices;
};

/**
 * @class SimulationRunner
 * @brief Simplifies the workflow of running a complete simulation with both
 *        American and European option pricing for all assets.
 */
class SimulationRunner {
public:
    /**
     * @brief Executes a full Monte Carlo simulation and prices both American and European options.
     *
     * @param config_ SimulationConfig with all market and network parameters
     * @param strike_ Strike price for options (applied uniformly across all assets)
     * @param maturity_ Time to maturity in years
     * @param seed_ Random seed for reproducibility (default: 42)
     * @param verbose_ Enable diagnostic output (default: true)
     * @return SimulationResults struct containing all computed option prices
     */
    static SimulationResults run_simulation(
        const SimulationConfig& config_,
        double strike_,
        double maturity_,
        int seed_ = 42,
        bool verbose_ = true
    ) {
        SimulationResults results;

        // --- 1. Generate Monte Carlo Paths ---
        NetworkSimulator simulator(config_, seed_);
        simulator.simulate();

        const auto& asset_paths = simulator.get_asset_paths();
        const auto& variance_paths = simulator.get_log_variance_paths();

        // --- 2. Initialize Pricers ---
        EuropeanPricer<EuropeanCall> eu_call_pricer(config_, asset_paths);
        EuropeanPricer<EuropeanPut> eu_put_pricer(config_, asset_paths);
        AmericanPricer<AmericanCall> am_call_pricer(config_, asset_paths, variance_paths, config_.W, verbose_);
        AmericanPricer<AmericanPut> am_put_pricer(config_, asset_paths, variance_paths, config_.W, verbose_);

        // --- 3. Price European Options ---
        EuropeanCall eu_call(strike_, maturity_);
        EuropeanPut eu_put(strike_, maturity_);
        results.european_call_prices = eu_call_pricer.price(eu_call);
        results.european_put_prices = eu_put_pricer.price(eu_put);

        // --- 4. Price American Options ---
        AmericanCall am_call(strike_, maturity_);
        AmericanPut am_put(strike_, maturity_);
        results.american_call_prices = am_call_pricer.price(am_call);
        results.american_put_prices = am_put_pricer.price(am_put);

        return results;
    }
};