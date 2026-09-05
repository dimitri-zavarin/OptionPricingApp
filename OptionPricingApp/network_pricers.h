#pragma once

#include "option.h"
#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

/**
 * @brief Template trait class to extract human-readable option type metadata
 */
template <typename PayoffType, typename ExerciseType>
struct OptionTraits;

// Specializations for all 4 option types
template <>
struct OptionTraits<CallPayoff, European> {
    static constexpr const char* payoff_type() { return "Call"; }
    static constexpr const char* exercise_type() { return "European"; }
};

template <>
struct OptionTraits<PutPayoff, European> {
    static constexpr const char* payoff_type() { return "Put"; }
    static constexpr const char* exercise_type() { return "European"; }
};

template <>
struct OptionTraits<CallPayoff, American> {
    static constexpr const char* payoff_type() { return "Call"; }
    static constexpr const char* exercise_type() { return "American"; }
};

template <>
struct OptionTraits<PutPayoff, American> {
    static constexpr const char* payoff_type() { return "Put"; }
    static constexpr const char* exercise_type() { return "American"; }
};

/**
 * @class EuropeanPricer
 * @brief Computes European option prices from pre-simulated asset paths.
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

    Eigen::VectorXd price(const OptionType& opt) const {
        Eigen::VectorXd option_prices = Eigen::VectorXd::Zero(config_.num_assets);
        double t = config_.num_steps * config_.dt;
        double discount_factor = std::exp(-config_.risk_free_rate * t);

        for (int p = 0; p < config_.num_paths; ++p) {
            Eigen::VectorXd s_t = asset_paths_[p].col(config_.num_steps);
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
 * @brief Evaluates American option premiums utilizing an optimized, dynamically
 * truncating Longstaff-Schwartz regression basis (4-Term or 5-Term), complete
 * with a rolling statistical significance and standard error profiler.
 */
template <typename OptionType>
class AmericanPricer {
private:
    const SimulationConfig& config_;
    const std::vector<Eigen::MatrixXd>& asset_paths_;
    const std::vector<Eigen::MatrixXd>& variance_paths_;
    const Eigen::MatrixXd& W_;
    bool verbose_;

    /**
     * @brief Generates an optimized 4-term or 5-term Hermite basis depending on
     * the presence of active exogenous network connections.
     */
    Eigen::VectorXd evaluate_optimized_basis(double S_i, double K, double X_i, double N_x_exo, bool is_network_linked) const {
        int num_features = is_network_linked ? 5 : 4;
        Eigen::VectorXd basis(num_features);
        
        double log_moneyness = std::log(S_i / K);           // Log moneyness
        
        basis(0) = 1.0;                                     // Constant
        basis(1) = log_moneyness;                           // Log(S/K) (Log Moneyness)
        basis(2) = log_moneyness * log_moneyness - 1.0;     // (Log(S/K))^2 - 1 (Convexity)
        basis(3) = X_i;                                     // X (Local Variance)

        if (is_network_linked) {
            basis(4) = N_x_exo;                             // Nx_exo (Network Variance Contagion)
        }

        return basis;
    }

public:
    AmericanPricer(const SimulationConfig& cfg,
        const std::vector<Eigen::MatrixXd>& a_paths,
        const std::vector<Eigen::MatrixXd>& v_paths,
        const Eigen::MatrixXd& W,
        bool verbose = false)
        : config_(cfg), asset_paths_(a_paths), variance_paths_(v_paths), W_(W), verbose_(verbose) {
    }

    double price_asset_option(int asset_idx, const OptionType& opt) const {
        int num_paths = config_.num_paths;
        int num_steps = config_.num_steps;
        double discount_factor = std::exp(-config_.risk_free_rate * config_.dt);
        double strike = opt.strike();

        // --- Topological Scan: Determine if asset relies on neighbors ---
        bool is_network_linked = false;
        for (int col = 0; col < config_.num_assets; ++col) {
            if (col != asset_idx && std::abs(W_(asset_idx, col)) > 1e-9) {
                is_network_linked = true;
                break;
            }
        }

        int num_features = is_network_linked ? 5 : 4;
        std::vector<std::string> term_names = is_network_linked ?
            std::vector<std::string>{"Const", "S", "S^2-1", "X", "Nx_exo"} :
            std::vector<std::string>{ "Const", "S", "S^2-1", "X" };

        // --- Initialize Diagnostic Trackers ---
        std::vector<int> significance_hits(num_features, 0);
        std::vector<double> r2_history;
        std::vector<int> itm_history;
        int valid_regressions = 0;

        // 1. Initialize Cash Flow vector at T
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

            // 3. Identify ITM paths
            for (int p = 0; p < num_paths; ++p) {
                double s_t = asset_paths_[p](asset_idx, t);
                if (opt.payoff(s_t) > 0.0) {
                    itm_paths.push_back(p);
                }
            }

            int num_itm = static_cast<int>(itm_paths.size());
            if (num_itm < num_features + 5) continue; // Minimum required for stable regression

            // 4. Construct Design Matrix (A) and Target Vector (Y)
            Eigen::MatrixXd mat_a(num_itm, num_features);
            Eigen::VectorXd vec_y(num_itm);

            for (int i = 0; i < num_itm; ++i) {
                int p_idx = itm_paths[i];
                double S_i = asset_paths_[p_idx](asset_idx, t);
                double X_i = variance_paths_[p_idx](asset_idx, t);

                double N_x_exo = 0.0;
                if (is_network_linked) {
                    for (int col = 0; col < config_.num_assets; ++col) {
                        if (col != asset_idx) {
                            N_x_exo += W_(asset_idx, col) * variance_paths_[p_idx](col, t);
                        }
                    }
                }

                mat_a.row(i) = evaluate_optimized_basis(S_i, strike, X_i, N_x_exo, is_network_linked);
                vec_y(i) = cash_flows(p_idx);
            }

            // 5. Execute Least Squares Regression
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(mat_a);
            Eigen::VectorXd beta = qr.solve(vec_y);
            
            // 6. Rolling Diagnostic: R^2, Standard Errors, T-Stats (ONLY IF VERBOSE)
            if (verbose_) {
                Eigen::VectorXd residuals = vec_y - (mat_a * beta);
                double rss = residuals.squaredNorm();

                // Calculate R^2
                double mean_y = vec_y.mean();
                double tss = (vec_y.array() - mean_y).square().sum();
                double r_squared = (tss > 0.0) ? (1.0 - (rss / tss)) : 0.0;
                r2_history.push_back(r_squared);
                itm_history.push_back(num_itm);

                // Calculate Standard Errors & T-Stats (Heavy Matrix Math)
                double sigma_squared = rss / (num_itm - static_cast<double>(num_features));
                Eigen::MatrixXd ata = mat_a.transpose() * mat_a;
                Eigen::MatrixXd cov_matrix = sigma_squared * ata.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(num_features, num_features));

                valid_regressions++;
                for (int j = 0; j < num_features; ++j) {
                    double se = std::sqrt(std::max(0.0, cov_matrix(j, j)));
                    double t_stat = beta(j) / (se + 1e-12);
                    if (std::abs(t_stat) > 1.96) {
                        significance_hits[j]++;
                    }
                }
            }
            
            // 7. Evaluate the Early Exercise Condition
            for (int i = 0; i < num_itm; ++i) {
                int p_idx = itm_paths[i];
                double S_i = asset_paths_[p_idx](asset_idx, t);
                double X_i = variance_paths_[p_idx](asset_idx, t);

                double N_x_exo = 0.0;
                if (is_network_linked) {
                    for (int col = 0; col < config_.num_assets; ++col) {
                        if (col != asset_idx) {
                            N_x_exo += W_(asset_idx, col) * variance_paths_[p_idx](col, t);
                        }
                    }
                }

                double immediate_exercise = opt.payoff(S_i);
                double continuation_value = evaluate_optimized_basis(S_i, strike, X_i, N_x_exo, is_network_linked).dot(beta);

                if (immediate_exercise > continuation_value) {
                    cash_flows(p_idx) = immediate_exercise;
                }
            }
        }

        // 8. Print the Final Significance Summary Block
        if (verbose_ && valid_regressions > 0) {
            std::string ticker = config_.tickers[asset_idx];
            std::string exercise = OptionTraits<typename OptionType::Payoff, typename OptionType::Exercise>::exercise_type();

            // Calculate averages for R2 and ITM paths
            double avg_r2 = 0.0, avg_itm = 0.0;
            if (!r2_history.empty()) {
                for (double r2 : r2_history) avg_r2 += r2;
                avg_r2 /= r2_history.size();
            }
            if (!itm_history.empty()) {
                for (int itm : itm_history) avg_itm += itm;
                avg_itm /= itm_history.size();
            }

            std::cout << "\n--- LSMC Diagnostics (Ticker: " << ticker << " | " << exercise << " | " << num_features << "-Term Basis) ---" << std::endl;
            std::cout << "Valid Regressions : " << valid_regressions << " / " << num_steps - 1 << std::endl;
            std::cout << "Avg ITM Paths     : " << std::fixed << std::setprecision(0) << avg_itm << std::endl;
            std::cout << "Avg R-Squared     : " << std::fixed << std::setprecision(4) << avg_r2 << std::endl;

            std::cout << std::string(45, '-') << std::endl;
            std::cout << std::left << std::setw(20) << "Basis Term" << "Significance (T > 1.96)" << std::endl;
            std::cout << std::string(45, '-') << std::endl;

            for (int j = 0; j < num_features; ++j) {
                double hit_percentage = (static_cast<double>(significance_hits[j]) / valid_regressions) * 100.0;
                std::cout << std::left << std::setw(20) << term_names[j]
                    << std::fixed << std::setprecision(1) << hit_percentage << "%" << std::endl;
            }
            std::cout << std::string(45, '-') << "\n" << std::endl;
        }

        cash_flows *= discount_factor;
        return cash_flows.mean();
    }

    /**
     * @brief Prices options for all assets and returns results as a vector.
     * 
     * @param opt Option specification for pricing
     * @return Eigen::VectorXd with prices for each asset
     */
    Eigen::VectorXd price(const OptionType& opt) const {
        Eigen::VectorXd prices(config_.num_assets);
        for (int i = 0; i < config_.num_assets; ++i) {
            prices(i) = price_asset_option(i, opt);
        }
        return prices;
    }
};