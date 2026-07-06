#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>

// Core framework integrations
#include "option.h"
#include "market_data.h"
#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"

// Struct to hold cross-sectional results for clean tabulation
struct PricingResult {
    std::string ticker;
    double euro_call;
    double amer_call;
    double call_prem;
    double euro_put;
    double amer_put;
    double put_prem;
};

// Helper function to print headers cleanly
void print_regime_table(const std::string& title, double T, const std::vector<PricingResult>& results) {
    std::cout << "\n--------------------------------------------------------------------------------------------------\n";
    std::cout << " REGIME: " << title << " | TENOR: T = " << std::fixed << std::setprecision(4) << T << " Year (" << int(T * 12) << "M)\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8) << "Ticker"
        << std::setw(15) << "Euro Call"
        << std::setw(15) << "Raw Amer Call"
        << std::setw(15) << "Call Prem"
        << std::setw(15) << "Euro Put"
        << std::setw(15) << "Raw Amer Put"
        << "Put Prem\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (const auto& res : results) {
        std::cout << std::left << std::setw(8) << res.ticker
            << std::setw(15) << res.euro_call << std::setw(15) << res.amer_call << std::setw(15) << res.call_prem
            << std::setw(15) << res.euro_put << std::setw(15) << res.amer_put << res.put_prem << "\n";
    }
}

// Global execution wrapper for a given maturity horizon
std::vector<PricingResult> evaluate_regime(SimulationConfig& config, double strike, double maturity) {
    // Re-run the simulation path matrix for this specific time-horizon step count
    config.num_steps = static_cast<int>(252.0 * maturity);
    if (config.num_steps == 0) config.num_steps = 21; // Baseline for 1-Month boundary
    config.dt = maturity / static_cast<double>(config.num_steps);

    NetworkSimulator simulator(config, 42);
    simulator.simulate();

    // 1. Instantiate contract instances to leverage the unified template payoff architecture
    AmericanCall call_contract(strike, maturity);
    AmericanPut  put_contract(strike, maturity);

    // 2. Initialize templated pricing engines using the contract classes
    EuropeanPricer<AmericanCall> euro_call_engine(config, simulator.get_asset_paths());
    EuropeanPricer<AmericanPut>  euro_put_engine(config, simulator.get_asset_paths());
    AmericanPricer<AmericanCall> amer_call_engine(config, simulator.get_asset_paths());
    AmericanPricer<AmericanPut>  amer_put_engine(config, simulator.get_asset_paths());

    // 3. Execute vector pricing via the refactored .price(opt) signatures
    Eigen::VectorXd euro_calls = euro_call_engine.price(call_contract);
    Eigen::VectorXd euro_puts = euro_put_engine.price(put_contract);

    std::vector<PricingResult> results;
    for (int i = 0; i < config.num_assets; ++i) {
        double ac = amer_call_engine.price_asset_option(i, call_contract);
        double ap = amer_put_engine.price_asset_option(i, put_contract);

        results.push_back({
            config.tickers[i],
            euro_calls(i), ac, (ac - euro_calls(i)),
            euro_puts(i),  ap, (ap - euro_puts(i))
            });
    }
    return results;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "     PROFESSOR ROJAS COMPREHENSIVE LSM SPATIOTEMPORAL SUITE     \n";
    std::cout << "             (RAW ALGORITHMIC OUTPUT MULTI-TENOR MODE)           \n";
    std::cout << "=================================================================\n";

    const int num_assets = 3;
    const int num_paths = 50000;
    const double strike = 100.0;

    // Define our two target temporal maturities for exploration
    std::vector<double> tenors = { 1.0 / 12.0, 1.0 }; // 1 Month vs 1 Year

    SimulationConfig config;
    config.num_assets = num_assets;
    config.num_paths = num_paths;
    config.risk_free_rate = 0.05;
    config.tickers = { "AAPL", "MSFT", "GOOG" };

    config.S0 = Eigen::VectorXd::Constant(num_assets, 100.0);
    config.q = Eigen::VectorXd::Zero(num_assets);
    config.q << 0.08, 0.00, 0.12;

    // Lambda helper to easily reset baseline model physics before setting topology
    auto reset_baseline = [&]() {
        config.X0 = Eigen::VectorXd::Constant(num_assets, -3.2);
        config.theta = Eigen::VectorXd::Constant(num_assets, -3.2);
        config.kappa = Eigen::VectorXd::Constant(num_assets, 2.0);
        config.xi = Eigen::VectorXd::Constant(num_assets, 0.15);
        config.rho = Eigen::VectorXd::Constant(num_assets, -0.60);
        config.gamma = Eigen::VectorXd::Zero(num_assets);
        config.W = Eigen::MatrixXd::Zero(num_assets, num_assets);
        };

    // Loop across both tenors to explore temporal interactions
    for (double T : tenors) {
        std::cout << "\n\n==================================================================================================";
        std::cout << "\n   COMPILING ANALYSIS FOR MATURITY HORIZON: T = " << std::fixed << std::setprecision(4) << T << " YEAR";
        std::cout << "\n==================================================================================================\n";

        // Regime 1: Uncoupled Heston
        reset_baseline();
        print_regime_table("1. UNCOUPLED BASELINE HOUSINGS (W = 0)", T, evaluate_regime(config, strike, T));

        // Regime 2: Standard Apple-Led Case
        reset_baseline();
        config.gamma = Eigen::VectorXd::Constant(num_assets, 0.50);
        config.W(1, 0) = 0.70; // AAPL -> MSFT
        config.W(2, 0) = 0.50; // AAPL -> GOOG
        print_regime_table("2. STANDARD APPLE-LED TOPOLOGY", T, evaluate_regime(config, strike, T));

        // Regime 3: Echo Chamber Case
        reset_baseline();
        config.gamma = Eigen::VectorXd::Constant(num_assets, 0.50);
        config.W(0, 2) = 0.40; // Cyclical loop feedback loop
        config.W(1, 0) = 0.50;
        config.W(2, 1) = 0.50;
        config.xi = Eigen::VectorXd::Constant(num_assets, 0.35); // Elevated vol-of-vol
        print_regime_table("3. CYCLICAL NETWORK ECHO CHAMBER", T, evaluate_regime(config, strike, T));

        // Regime 4: Apple Vol Shock
        reset_baseline();
        config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
        config.X0(0) = -1.0; // Idiosyncratic volatility spike today
        config.theta(0) = -1.0;
        config.W(1, 0) = 1.0;
        config.W(2, 0) = 1.0;
        print_regime_table("4. IDIOSYNCRATIC APPLE VOL SHOCK", T, evaluate_regime(config, strike, T));

        // Regime 5: Volatility Sink
        reset_baseline();
        config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
        config.kappa(1) = 8.0; // Extreme mean-reversion speed
        config.theta(1) = -4.0; // Compressing to low log-variance floors
        config.X0(1) = -4.0;
        config.W(0, 1) = 0.90; // AAPL and GOOG anchored tightly to MSFT
        config.W(2, 1) = 0.90;
        print_regime_table("5. VOLATILITY SINK (MSFT ANCHOR)", T, evaluate_regime(config, strike, T));
    }

    std::cout << "\n=================================================================\n";
    std::cout << "             ALL TIME-HORIZON RUNS LOGGED SUCCESSFUL             \n";
    std::cout << "=================================================================\n";
    return 0;
}