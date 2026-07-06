#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>

#include "option.h"
#include "market_data.h"
#include "simulation_config.h"
#include "network_simulator.h"
#include "network_pricers.h"

struct RunResult {
    std::string ticker;
    double euro_call;
    double amer_call;
    double call_prem;
    double euro_put;
    double amer_put;
    double put_prem;
};

void print_matrix(const std::string& name, const Eigen::MatrixXd& matrix, const std::vector<std::string>& tickers) {
    std::cout << "  " << name << ":\n";
    std::cout << "          ";
    for (const auto& t : tickers) std::cout << std::right << std::setw(8) << t;
    std::cout << "\n";
    for (int i = 0; i < matrix.rows(); ++i) {
        std::cout << "    " << std::left << std::setw(6) << tickers[i] << "[";
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2) << matrix(i, j);
        }
        std::cout << " ]\n";
    }
}

void evaluate_and_display_regime(SimulationConfig& config, const std::string& title, double strike, double maturity) {
    config.num_steps = static_cast<int>(252.0 * maturity);
    if (config.num_steps == 0) config.num_steps = 21;
    config.dt = maturity / static_cast<double>(config.num_steps);

    std::cout << "\n==================================================================================================\n";
    std::cout << " REGIME CONFIGURATION: " << title << "\n";
    std::cout << "==================================================================================================\n";

    std::cout << "--- CRUCIAL STRUCTURAL PARAMETERS ---\n";
    std::cout << std::left << std::setw(10) << "Ticker"
        << std::setw(10) << "S0"
        << std::setw(10) << "Div (q)"
        << std::setw(10) << "Kappa"
        << std::setw(10) << "Theta"
        << std::setw(10) << "X0"
        << "Gamma (Sensitivity)\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < config.num_assets; ++i) {
        std::cout << std::left << std::setw(10) << config.tickers[i]
            << std::setw(10) << std::fixed << std::setprecision(1) << config.S0(i)
            << std::setw(10) << std::fixed << std::setprecision(2) << config.q(i)
            << std::setw(10) << std::fixed << std::setprecision(1) << config.kappa(i)
            << std::setw(10) << std::fixed << std::setprecision(2) << config.theta(i)
            << std::setw(10) << std::fixed << std::setprecision(2) << config.X0(i)
            << std::fixed << std::setprecision(2) << config.gamma(i) << "\n";
    }
    std::cout << "\n";
    print_matrix("Spatial Topology Matrix (W)", config.W, config.tickers);
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    NetworkSimulator simulator(config, 42);
    simulator.simulate();

    const auto& asset_paths = simulator.get_asset_paths();

    AmericanCall call_contract(strike, maturity);
    AmericanPut  put_contract(strike, maturity);

    EuropeanPricer<AmericanCall> euro_call_engine(config, asset_paths);
    EuropeanPricer<AmericanPut>  euro_put_engine(config, asset_paths);
    AmericanPricer<AmericanCall> amer_call_engine(config, asset_paths);
    AmericanPricer<AmericanPut>  amer_put_engine(config, asset_paths);

    Eigen::VectorXd euro_calls = euro_call_engine.price(call_contract);
    Eigen::VectorXd euro_puts = euro_put_engine.price(put_contract);

    std::cout << "\n--- DERIVATIVES SURFACE VALUATIONS (T = " << maturity << " Year) ---\n";
    std::cout << std::left << std::setw(8) << "Ticker"
        << std::setw(15) << "Euro Call"
        << std::setw(15) << "Raw Amer Call"
        << std::setw(15) << "Call Prem"
        << std::setw(15) << "Euro Put"
        << std::setw(15) << "Raw Amer Put"
        << "Put Prem\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < config.num_assets; ++i) {
        double ac = amer_call_engine.price_asset_option(i, call_contract);
        double ap = amer_put_engine.price_asset_option(i, put_contract);

        std::cout << std::left << std::setw(8) << config.tickers[i]
            << std::setw(15) << std::fixed << std::setprecision(4) << euro_calls(i)
            << std::setw(15) << ac
            << std::setw(15) << (ac - euro_calls(i))
            << std::setw(15) << euro_puts(i)
            << std::setw(15) << ap
            << (ap - euro_puts(i)) << "\n";
    }
    std::cout << "==================================================================================================\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "          SPATIOTEMPORAL GRAPH LAPLACIAN ANALYSIS                \n";
    std::cout << "              (ROW-STOCHASTIC CALIBRATION MODE)                  \n";
    std::cout << "=================================================================\n";

    const int num_assets = 3;
    const int num_paths = 50000;
    const double strike = 100.0;
    const double maturity = 1.0;

    SimulationConfig config;
    config.num_assets = num_assets;
    config.num_paths = num_paths;
    config.risk_free_rate = 0.05;
    config.tickers = { "AAPL", "MSFT", "GOOG" };
    config.S0 = Eigen::VectorXd::Constant(num_assets, 100.0);
    config.q = Eigen::VectorXd::Zero(num_assets);
    config.q << 0.08, 0.00, 0.12;

    auto reset_baseline = [&]() {
        config.X0 = Eigen::VectorXd::Constant(num_assets, -3.2);
        config.theta = Eigen::VectorXd::Constant(num_assets, -3.2);
        config.kappa = Eigen::VectorXd::Constant(num_assets, 2.0);
        config.xi = Eigen::VectorXd::Constant(num_assets, 0.15);
        config.rho = Eigen::VectorXd::Constant(num_assets, -0.60);
        config.gamma = Eigen::VectorXd::Zero(num_assets);
        config.W = Eigen::MatrixXd::Zero(num_assets, num_assets);
        };

    // 1. DECOUPLED BASELINE REGIME
    // Every asset maps 100% to itself by default via the identity matrix structure
    reset_baseline();
    config.W = Eigen::MatrixXd::Identity(num_assets, num_assets);
    evaluate_and_display_regime(config, "1. DECOUPLED STANDALONE HESTON (W = I)", strike, maturity);

    // 2. STANDARD LINKED REGIME
    // AAPL is independent (self-loop). MSFT and GOOG pull completely from AAPL.
    reset_baseline();
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.50);
    config.W(0, 0) = 1.00; // AAPL self-loop (Row sum = 1.0)
    config.W(1, 0) = 1.00; // MSFT -> AAPL (Row sum = 1.0)
    config.W(2, 0) = 1.00; // GOOG -> AAPL (Row sum = 1.0)
    evaluate_and_display_regime(config, "2. STANDARD LINKED STAR TOPOLOGY", strike, maturity);

    // 3. VOLATILITY SPIKE REGIME
    // AAPL holds an explosive internal shock. Rows remain strictly normalized.
    reset_baseline();
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
    config.X0(0) = -1.0;
    config.theta(0) = -1.0;
    config.W(0, 0) = 1.00; // AAPL self-loop
    config.W(1, 0) = 1.00; // MSFT -> AAPL
    config.W(2, 0) = 1.00; // GOOG -> AAPL
    evaluate_and_display_regime(config, "3. IDIOSYNCRATIC APPLE VOLATILITY SHOCK", strike, maturity);

    // 4. VOLATILITY SINK REGIME
    // MSFT is an independent low-variance sink node (self-loop). AAPL/GOOG pull 90% from MSFT, 10% from self.
    reset_baseline();
    config.gamma = Eigen::VectorXd::Constant(num_assets, 0.80);
    config.kappa(1) = 8.0;
    config.theta(1) = -4.0;
    config.X0(1) = -4.0;

    config.W(0, 1) = 0.90; config.W(0, 0) = 0.10; // AAPL row sums to 1.0
    config.W(1, 1) = 1.00;                        // MSFT sink self-loop (Row sums to 1.0)
    config.W(2, 1) = 0.90; config.W(2, 2) = 0.10; // GOOG row sums to 1.0
    evaluate_and_display_regime(config, "4. TOPOLOGICAL VOLATILITY SINK DYNAMICS", strike, maturity);

    return 0;
}