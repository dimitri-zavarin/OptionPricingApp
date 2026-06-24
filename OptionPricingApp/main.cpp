#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

#include "market_data.h"
#include "option.h"
#include "config_loader.h"
#include "network_monte_carlo.h"
#include "implied_div.h"
#include "black_scholes.h"
#include "binomial.h"

// Helper function to safely match contracts by ticker symbol
VanillaOptionData match_contract(const std::string& target_ticker, const std::vector<VanillaOptionData>& contracts) {
    for (const auto& contract : contracts) {
        if (contract.ticker == target_ticker) {
            return contract;
        }
    }
    throw std::runtime_error("Calibration contract not found for ticker: " + target_ticker);
}

int main() {
    try {
        std::cout << "==================================================\n";
        std::cout << " SPATIOTEMPORAL VOLATILITY NETWORK PRICER\n";
        std::cout << "==================================================\n";

        // 1. Ingest the Macro Environment
        std::vector<MarketData> universe;
        std::vector<std::string> tickers;
        NetworkLogArchConfig net_config;

        std::cout << "[*] Loading Network Configurations...\n";
        NetworkConfigLoader::load_system_config("network_config.csv", universe, tickers, net_config);
        NetworkConfigLoader::load_weight_matrix("weight_matrix.csv", net_config);

        // 2. Ingest the Micro Calibration Contracts
        std::cout << "[*] Loading Calibration Contracts...\n";
        auto contracts = NetworkConfigLoader::load_calibration_options("calibration_options.csv");

        // For this test, we will isolate the very first asset in the universe
        size_t test_idx = 0;
        std::string test_ticker = tickers[test_idx];

        // Match by ticker string
        VanillaOptionData target_contract = match_contract(test_ticker, contracts);

        std::cout << "\nTarget Asset: " << test_ticker << "\n";
        std::cout << "Spot Price: $" << universe[test_idx].S0 << "\n";
        std::cout << "Contract: Strike = $" << target_contract.strike
            << ", TTM = " << target_contract.maturity << " yrs\n";

        // 3. Setup the Option Object using your EuCall alias
        EuCall test_option(target_contract.strike, target_contract.maturity);

        // 4. Calibrate the Implied Dividend Yield
        std::cout << "\n[*] Running Newton-Raphson Implied Dividend Calibration...\n";
        double implied_q = IDivSolver<EuCall, BinomialPricer>::solve(
            universe[test_idx], test_option, target_contract.targetPrice
        );
        universe[test_idx].q = implied_q;
        std::cout << " -> Calibrated " << test_ticker << " Dividend Yield (q): "
            << std::fixed << std::setprecision(4) << (implied_q * 100.0) << "%\n";

        // ==========================================
        // THE PRICING SHOWDOWN
        // ==========================================
        std::cout << "\n==================================================\n";
        std::cout << " PRICING ENGINE COMPARISON\n";
        std::cout << "==================================================\n";

        // Engine 1: Black-Scholes (Closed Form)
        double bs_price = BlackScholesPricer<EuCall>::price(universe[test_idx], test_option);
        std::cout << std::left << std::setw(30) << "[1] Black-Scholes (Constant Vol):"
            << "$" << bs_price << "\n";

        // Engine 2: Binomial Tree (Discrete Space)
        int tree_steps = 1000;
        double binom_price = BinomialPricer<EuCall>::price(universe[test_idx], test_option, tree_steps);
        std::cout << std::left << std::setw(30) << "[2] Binomial Tree (1000 steps):"
            << "$" << binom_price << "\n";

        // Engine 3: Spatiotemporal Network Monte Carlo (Stochastic Vol)
        std::cout << "\n[*] Initializing Network Monte Carlo Engine...\n";
        NetworkMonteCarloPricer mc_pricer(universe, net_config);

        // Define a Lambda Payoff that isolates only our test asset
        double K = target_contract.strike;
        auto single_asset_payoff = [K, test_idx](const std::vector<double>& S_T) {
            return std::max(S_T[test_idx] - K, 0.0);
            };

        size_t num_sims = 50000;
        size_t trading_days = static_cast<size_t>(std::round(target_contract.maturity * 252));

        auto start_time = std::chrono::high_resolution_clock::now();
        double mc_price = mc_pricer.price_basket_option(num_sims, target_contract.maturity, trading_days, single_asset_payoff);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mc_duration = end_time - start_time;

        std::cout << std::left << std::setw(30) << "[3] Network Monte Carlo:"
            << "$" << mc_price << " (" << num_sims << " paths, " << mc_duration.count() << " sec)\n";

        std::cout << "==================================================\n";

    }
    catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}