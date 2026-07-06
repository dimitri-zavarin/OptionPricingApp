#pragma once

#include "option.h"
#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

/**
 * @class EuropeanPricer
 * @brief Computes European option prices from pre-simulated asset paths.
 *
 * This pricer assumes that `asset_paths_` contains `config_.num_paths` simulation
 * matrices. Each matrix has `config_.num_assets` rows and `config_.num_steps + 1`
 * columns (including the initial time t=0 column). The pricer computes the payoff
 * at maturity for each asset across all paths, averages the payoffs, and applies
 * discounting to return the present value.
 *
 * Notes:
 * - The pricer returns a vector of size `config_.num_assets` containing the price
 *   for each asset in the basket for the given strike and option type (call/put).
 * - Payoffs are computed elementwise for each asset (no cross-asset payoffs).
 */
template <typename OptionType>
class EuropeanPricer {
private:
    const SimulationConfig& config_;
    const std::vector<Eigen::MatrixXd>& asset_paths_;

public:
    EuropeanPricer(const SimulationConfig& cfg, const std::vector<Eigen::MatrixXd>& paths)
        : config_(cfg), asset_paths_(paths) {
    }

    /**
     * @brief Prices the option for each asset in the basket simultaneously.
     * @param opt The templated option contract containing the payoff structure.
     * @return Eigen::VectorXd containing the present value prices for each asset.
     */
    Eigen::VectorXd price(const OptionType& opt) const {
        Eigen::VectorXd option_prices = Eigen::VectorXd::Zero(config_.num_assets);
        double t = config_.num_steps * config_.dt;
        double discount_factor = std::exp(-config_.risk_free_rate * t);

        for (int p = 0; p < config_.num_paths; ++p) {
            Eigen::VectorXd s_t = asset_paths_[p].col(config_.num_steps);

            // Apply the contract's payoff function element-wise to the price vector
            Eigen::VectorXd payoffs = s_t.unaryExpr([&opt](double s) {
                return opt.payoff(s);
                });

            option_prices += payoffs;
        }

        return (option_prices / static_cast<double>(config_.num_paths)) * discount_factor;
    }
};

/**
 * @class AmericanPricer
 * @brief Evaluates American option premiums across simulated Monte Carlo paths
 * utilizing Longstaff-Schwartz (LSM) backward induction with cross-sectional
 * z-scoring and an orthogonal probabilists' Hermite polynomial basis.
 */
template <typename OptionType>
class AmericanPricer {
private:
    const SimulationConfig& config_;
    const std::vector<Eigen::MatrixXd>& asset_paths_;

    /**
     * @brief Generates orthogonal Probabilists' Hermite polynomials.
     * @param z The cross-sectionally standardized log-moneyness (z-score).
     * @return Eigen::VectorXd containing a 4-term orthogonal basis.
     */
    Eigen::VectorXd evaluate_hermite_basis(double z) const {
        Eigen::VectorXd basis(4);
        basis(0) = 1.0;                           // He_0(z)
        basis(1) = z;                             // He_1(z)
        basis(2) = z * z - 1.0;                   // He_2(z)
        basis(3) = std::pow(z, 3) - 3.0 * z;      // He_3(z)
        return basis;
    }

public:
    AmericanPricer(const SimulationConfig& cfg, const std::vector<Eigen::MatrixXd>& paths)
        : config_(cfg), asset_paths_(paths) {
    }

    /**
     * @brief Prices the American option independently for a specific asset index in the basket.
     * @param asset_idx The row index corresponding to the target stock.
     * @param opt The templated American option contract containing the payoff structure.
     * @return The fair-value American option premium today (t=0).
     */
    double price_asset_option(int asset_idx, const OptionType& opt) const {
        int num_paths = config_.num_paths;
        int num_steps = config_.num_steps;
        double discount_factor = std::exp(-config_.risk_free_rate * config_.dt);
        double strike = opt.strike();

        // 1. Initialize the Cash Flow vector at Maturity (T)
        Eigen::VectorXd cash_flows(num_paths);
        for (int p = 0; p < num_paths; ++p) {
            double s_T = asset_paths_[p](asset_idx, num_steps);
            cash_flows(p) = opt.payoff(s_T);
        }

        // 2. Roll backward through time (T-1 down to 1)
        for (int t = num_steps - 1; t >= 1; --t) {

            cash_flows *= discount_factor;

            std::vector<int> itm_paths;
            itm_paths.reserve(num_paths);
            std::vector<double> itm_x; // Store raw log-moneyness
            itm_x.reserve(num_paths);
            double sum_x = 0.0;

            // 3. Identify ITM paths and extract log-moneyness
            for (int p = 0; p < num_paths; ++p) {
                double s_t = asset_paths_[p](asset_idx, t);
                if (opt.payoff(s_t) > 0.0) {
                    itm_paths.push_back(p);
                    double x = std::log(s_t / strike);
                    itm_x.push_back(x);
                    sum_x += x;
                }
            }

            int num_itm = static_cast<int>(itm_paths.size());
            if (num_itm < 3) continue;

            // 4. Compute cross-sectional Mean and Standard Deviation for Z-Scoring
            double mean_x = sum_x / num_itm;
            double variance_x = 0.0;
            for (double x : itm_x) {
                variance_x += (x - mean_x) * (x - mean_x);
            }
            double std_x = std::sqrt(variance_x / (num_itm - 1.0));
            if (std_x < 1e-8) std_x = 1.0; // Prevent division by zero on identical paths

            // 5. Construct Design Matrix (A) and Target Vector (Y)
            Eigen::MatrixXd mat_a(num_itm, 4);
            Eigen::VectorXd vec_y(num_itm);

            for (int i = 0; i < num_itm; ++i) {
                int p_idx = itm_paths[i];
                double z = (itm_x[i] - mean_x) / std_x; // Standardize to z-score

                mat_a.row(i) = evaluate_hermite_basis(z);
                vec_y(i) = cash_flows(p_idx);
            }

            // 6. Execute Least Squares Regression via Rank-Revealing QR
            Eigen::VectorXd beta = mat_a.colPivHouseholderQr().solve(vec_y);

            // 7. Evaluate the Early Exercise Condition
            for (int i = 0; i < num_itm; ++i) {
                int p_idx = itm_paths[i];
                double s_t = asset_paths_[p_idx](asset_idx, t);

                double immediate_exercise = opt.payoff(s_t);
                double z = (itm_x[i] - mean_x) / std_x;
                double continuation_value = evaluate_hermite_basis(z).dot(beta);

                if (immediate_exercise > continuation_value) {
                    cash_flows(p_idx) = immediate_exercise;
                }
            }
        }

        // 8. Discount final cash flows to t=0 and average
        cash_flows *= discount_factor;
        return cash_flows.mean();
    }
};