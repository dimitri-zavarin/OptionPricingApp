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
        const Eigen::MatrixXd& W)
        : config_(cfg), asset_paths_(a_paths), variance_paths_(v_paths), W_(W) {
    }

    double price_asset_option(int asset_idx, const OptionType& opt) const {
        int num_paths = config_.num_paths;
        int num_steps = config_.num_steps;
        double discount_factor = std::exp(-config_.risk_free_rate * config_.dt);
        double strike = opt.strike();  // Get the strike price

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

        // --- Initialize Significance, Standard Error, and Beta Trackers ---
        std::vector<int> significance_hits(num_features, 0);
        std::vector<std::vector<double>> se_history(num_features);      // Store all SEs per parameter
        std::vector<std::vector<double>> beta_history(num_features);    // Store all betas per parameter
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
            
            // 6. Rolling Diagnostic: Accumulate Standard Errors, T-Stats, and Beta values
            Eigen::VectorXd residuals = vec_y - (mat_a * beta);
            double rss = residuals.squaredNorm();
            double sigma_squared = rss / (num_itm - static_cast<double>(num_features));

            Eigen::MatrixXd ata = mat_a.transpose() * mat_a;
            Eigen::MatrixXd cov_matrix = sigma_squared * ata.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(num_features, num_features));

            valid_regressions++;
            for (int j = 0; j < num_features; ++j) {
                double se = std::sqrt(std::max(0.0, cov_matrix(j, j)));
                se_history[j].push_back(se);            // Store standard error
                beta_history[j].push_back(beta(j));     // Store beta coefficient

                double t_stat = beta(j) / (se + 1e-12); // Prevent div by zero
                if (std::abs(t_stat) > 1.96) {          // 95% Confidence threshold
                    significance_hits[j]++;
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

        // 8. Print the Final Significance, Error, and Beta Summary Block
        if (valid_regressions > 0) {
            std::cout << "\n--- OLS Diagnostics (Asset " << asset_idx
                << " | " << num_features << "-Term Basis | Valid Regressions: "
                << valid_regressions << "/" << num_steps - 1 << ") ---" << std::endl;

            std::cout << std::left << std::setw(12) << "Term"
                << std::setw(18) << "Significance %"
                << std::setw(15) << "Median Beta"
                << std::setw(15) << "Median SE"
                << "Avg SE" << std::endl;
            std::cout << std::string(75, '-') << std::endl;

            for (int j = 0; j < num_features; ++j) {
                double hit_percentage = (static_cast<double>(significance_hits[j]) / valid_regressions) * 100.0;
                
                // Compute median beta
                std::vector<double> sorted_betas = beta_history[j];
                std::sort(sorted_betas.begin(), sorted_betas.end());
                double median_beta;
                if (sorted_betas.size() % 2 == 0) {
                    median_beta = (sorted_betas[sorted_betas.size() / 2 - 1] + sorted_betas[sorted_betas.size() / 2]) / 2.0;
                } else {
                    median_beta = sorted_betas[sorted_betas.size() / 2];
                }

                // Compute median SE
                std::vector<double> sorted_ses = se_history[j];
                std::sort(sorted_ses.begin(), sorted_ses.end());
                double median_se;
                if (sorted_ses.size() % 2 == 0) {
                    median_se = (sorted_ses[sorted_ses.size() / 2 - 1] + sorted_ses[sorted_ses.size() / 2]) / 2.0;
                } else {
                    median_se = sorted_ses[sorted_ses.size() / 2];
                }

                double avg_se = 0.0;
                for (double se : se_history[j]) {
                    avg_se += se;
                }
                avg_se /= se_history[j].size();

                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << hit_percentage << "%";

                std::cout << std::left << std::setw(12) << term_names[j]
                    << std::setw(18) << ss.str()
                    << std::setw(15) << std::fixed << std::setprecision(4) << median_beta
                    << std::setw(15) << std::fixed << std::setprecision(4) << median_se
                    << std::fixed << std::setprecision(4) << avg_se << std::endl;
            }
            std::cout << std::string(75, '-') << "\n" << std::endl;
        }

        cash_flows *= discount_factor;
        return cash_flows.mean();
    }
};