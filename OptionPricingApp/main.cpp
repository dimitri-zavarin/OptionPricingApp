#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include "market_data.h"
#include "config_loader.h"
#include "network_monte_carlo.h"
#include "option.h"
#include "black_scholes.h"
#include "implied_div.h"

int main() {
    try {
        std::cout << "=========================================================\n";
        std::cout << "   INITIALIZING QUANTITATIVE FINANCE NETWORK ENGINE      \n";
        std::cout << "   Framework: Mattera & Otto (2024) Network log-ARCH    \n";
        std::cout << "=========================================================\n\n";

        // Step 1: Initialize baseline multi-asset containers
        std::vector<MarketData> market_universe;
        std::vector<std::string> tickers;
        NetworkLogArchConfig log_arch_config;
        log_arch_config.rho_global = 0.15;

        // Vectors to temporarily hold option pricing bounds parsed from our data pipeline
        std::vector<double> opt_strikes;
        std::vector<double> opt_maturities;
        std::vector<double> opt_target_prices;

        // Change these lines in main.cpp to your absolute project folder:
        std::string config_path = "C:\\Users\\Dimitri\\source\\repos\\OptionPricingApp\\OptionPricingApp\\network_config.csv";
        std::string matrix_path = "C:\\Users\\Dimitri\\source\\repos\\OptionPricingApp\\OptionPricingApp\\weight_matrix.csv";

        // Step 2: Stream variables from the Python pipeline CSV artifacts
        std::cout << "[INFO] Ingesting parameters from " << config_path << "...\n";
        NetworkConfigLoader::load_system_config(config_path, market_universe, tickers, log_arch_config,
            opt_strikes, opt_maturities, opt_target_prices);

        std::cout << "[INFO] Ingesting spatial topology matrix from " << matrix_path << "...\n";
        NetworkConfigLoader::load_weight_matrix(matrix_path, log_arch_config);

        size_t n_assets = tickers.size();
        std::cout << "[SUCCESS] Adjacency matrix linked. Dimension: " << n_assets << " assets.\n\n";

        // Step 3: Run your Custom Template Implied Dividend Solver natively!
        std::cout << "--- Launching C++ Template Implied Dividend Optimizations ---\n";

        IDivConfig solver_settings;
        solver_settings.q = 0.015;     // Starting point guess for our Newton-Raphson loops
        solver_settings.tol = 1e-6;     // Match your target precision bounds
        solver_settings.steps = 0;      // Steps parameter not used by BlackScholesPricer, safely set to 0

        for (size_t i = 0; i < n_assets; ++i) {
            std::cout << "  -> Calibrating option skew metrics for " << tickers[i] << "...\n";

            // Instantiate your exact policy-based option type class!
            // Type definition: using EuCall = Option<CallPayoff, European>;
            EuCall market_contract(opt_strikes[i], opt_maturities[i]);

            // Invoke your exact IDivSolver template signature using your analytical BlackScholesPricer engine!
            double calibrated_q = IDivSolver<EuCall, BlackScholesPricer>::solve(
                market_universe[i],
                market_contract,
                opt_target_prices[i],
                solver_settings
            );

            // Save the verified yield straight into our asset profile drift struct
            market_universe[i].q = calibrated_q;

            std::cout << "     * Target Exchange Price: $" << opt_target_prices[i]
                << " -> Calibrated Implied q = " << std::fixed << std::setprecision(4) << calibrated_q * 100.0 << "%\n";
        }
        std::cout << "[SUCCESS] Multi-asset drift channels are risk-neutralized.\n\n";

        // Display an operational state summary to the output console terminal
        std::cout << "-----------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(10) << "Ticker"
            << std::setw(12) << "Spot S0"
            << std::setw(12) << "Risk-Free r"
            << std::setw(14) << "Calibrated q"
            << std::setw(12) << "Initial IV" << "\n";
        std::cout << "-----------------------------------------------------------------------\n";
        for (size_t i = 0; i < n_assets; ++i) {
            std::cout << std::left << std::setw(10) << tickers[i]
                << "$" << std::setw(11) << std::fixed << std::setprecision(2) << market_universe[i].S0
                << std::setw(12) << std::setprecision(4) << market_universe[i].r
                << std::setw(14) << market_universe[i].q * 100.0
                << std::setw(12) << market_universe[i].sigma << "\n";
        }
        std::cout << "-----------------------------------------------------------------------\n\n";

        // Step 4: Define the Option Payoff Logic (Arithmetic Average Basket Option)
        double sum_initial_spots = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            sum_initial_spots += market_universe[i].S0;
        }
        double basket_spot_average = sum_initial_spots / static_cast<double>(n_assets);

        double strike_K = basket_spot_average; // Dynamically binds the strike at-the-money (~$491.78)
        double maturity = 1.0;
        size_t time_steps = 252;
        size_t simulations = 100000;

        std::cout << "--- Configuring Basket Option Specification ---\n";
        std::cout << "  * Type:                 Arithmetic Average Call\n";
        std::cout << "  * Strike (K):          " << strike_K << "\n";
        std::cout << "  * Maturity (T):         " << maturity << " Year\n";
        std::cout << "  * Step Granularity:     " << time_steps << " (Daily increments)\n";
        std::cout << "  * Simulation Iterations: " << simulations << "\n\n";

        auto basket_call_payoff = [strike_K, n_assets](const std::vector<double>& S_T) -> double {
            double sum_prices = 0.0;
            for (size_t i = 0; i < n_assets; ++i) {
                sum_prices += S_T[i];
            }
            double basket_average = sum_prices / static_cast<double>(n_assets);
            double value = basket_average - strike_K;
            return (value > 0.0) ? value : 0.0;
            };

        // Step 5: Initialize the Pricing Module and Launch Monte Carlo Paths
        std::cout << "[INFO] Initializing system matrix inversion (I - rho*W)^-1...\n";
        NetworkMonteCarloPricer pricer(market_universe, log_arch_config);

        std::cout << "[RUNNING] Simulating joint spatiotemporal risk diffusion paths...\n";
        double final_option_price = pricer.price_basket_option(simulations, maturity, time_steps, basket_call_payoff);

        std::cout << "\n=========================================================\n";
        std::cout << "   SIMULATION PRICING RESULTS                            \n";
        std::cout << "=========================================================\n";
        std::cout << "   Network Option Value:  $" << std::fixed << std::setprecision(4) << final_option_price << "\n";
        std::cout << "=========================================================\n";

    }
    catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR EXCEPTION ACCESSED]: " << e.what() << "\n";
        return 1;
    }
    return 0;
}