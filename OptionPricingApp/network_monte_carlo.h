#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>
#include "market_data.h"

struct NetworkLogArchConfig {
    std::vector<std::vector<double>> W;
    std::vector<double> omega;
    std::vector<double> gamma;
    double rho = 0.05;
    std::vector<std::vector<double>> inv_I_rhoW;
    std::vector<double> mu_smearing;
};

struct NetworkMonteCarloPricer {
private:
    std::vector<MarketData> market_universe;
    NetworkLogArchConfig config;
    size_t num_assets;

public:
    NetworkMonteCarloPricer(const std::vector<MarketData>& universe, const NetworkLogArchConfig& network_config)
        : market_universe(universe), config(network_config), num_assets(universe.size()) {

        if (num_assets == 0) throw std::invalid_argument("Market universe cannot be empty.");

        // Dimension Guardrail
        if (config.mu_smearing.size() < num_assets) {
            throw std::runtime_error("Config Error: mu_smearing vector size does not match asset universe size.");
        }

        // PRE-INVERT (I - rho * W) WITH EIGEN
        Eigen::MatrixXd I_rhoW(num_assets, num_assets);

        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                double identity_val = (i == j) ? 1.0 : 0.0;
                I_rhoW(i, j) = identity_val - (config.rho * config.W[i][j]);
            }
        }

        Eigen::MatrixXd inv_matrix = I_rhoW.inverse();

        config.inv_I_rhoW.assign(num_assets, std::vector<double>(num_assets, 0.0));
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                config.inv_I_rhoW[i][j] = inv_matrix(i, j);
            }
        }
    }

    double price_basket_option(size_t num_simulations, double maturity_years, size_t num_days,
        const std::function<double(const std::vector<double>&)>& payoff_lambda) {

        double dt = maturity_years / static_cast<double>(num_days);
        double sqrt_dt = std::sqrt(dt);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> normal(0.0, 1.0);

        double total_payoff_accumulator = 0.0;
        const double E_LN_EPS_SQ = -1.27036; // Keep as baseline anchor for raw random generation

        for (size_t sim = 0; sim < num_simulations; ++sim) {

            std::vector<double> S_t(num_assets);
            std::vector<double> X_t(num_assets);

            // Initialize Asset Prices and the Log-Squared Returns State (X_t)
            for (size_t i = 0; i < num_assets; ++i) {
                S_t[i] = market_universe[i].S0;
                X_t[i] = std::log(market_universe[i].sigma * market_universe[i].sigma) + E_LN_EPS_SQ;
            }

            for (size_t day = 0; day < num_days; ++day) {

                std::vector<double> epsilon(num_assets);
                std::vector<double> u_next(num_assets);

                // 1. Generate standard normal shocks and the zero-mean ARMA innovation (u_t)
                for (size_t i = 0; i < num_assets; ++i) {
                    epsilon[i] = normal(gen);
                    double eps_sq = std::max(1e-12, epsilon[i] * epsilon[i]);
                    u_next[i] = std::log(eps_sq) - E_LN_EPS_SQ; // Zero-mean transformation
                }

                // 2. Build the Core Forcing Vector (Right side of Equation 8)
                std::vector<double> core_forcing(num_assets, 0.0);
                for (size_t i = 0; i < num_assets; ++i) {
                    // Match Mattera-Otto precisely: phi_i = omega_i + mu_smearing_i
                    double phi_i = config.omega[i] + config.mu_smearing[i];
                    core_forcing[i] = phi_i + (config.gamma[i] * X_t[i]) + u_next[i];
                }

                // 3. Resolve the Spatial Contagion via Matrix Multiplication
                std::vector<double> next_X_t(num_assets, 0.0);
                for (size_t i = 0; i < num_assets; ++i) {
                    for (size_t j = 0; j < num_assets; ++j) {
                        next_X_t[i] += config.inv_I_rhoW[i][j] * core_forcing[j];
                    }
                }

                // Apply stability caps to prevent structural explosion
                for (size_t i = 0; i < num_assets; ++i) {
                    next_X_t[i] = std::max(-10.0, std::min(1.5, next_X_t[i]));
                }

                // 4. Extract continuous variance and step geometric Brownian motion
                for (size_t i = 0; i < num_assets; ++i) {
                    double eps_sq = std::max(1e-12, epsilon[i] * epsilon[i]);

                    // Isolate variance from endogenous returns grid: ln(h_t) = X_t - ln(eps^2)
                    double ln_h_curr = next_X_t[i] - std::log(eps_sq);
                    ln_h_curr = std::max(-10.0, std::min(1.5, ln_h_curr));

                    double h_curr = std::exp(ln_h_curr);

                    if (day % 10 == 0 && sim == 0) {
                        std::cout << "Day " << day << " | Volatility (h): " << h_curr
                            << " | Drift Term: " << (market_universe[i].r - market_universe[i].q - 0.5 * h_curr)
                            << " | S_t: " << S_t[i] << "\n";
                    }

                    double drift = market_universe[i].r - market_universe[i].q - 0.5 * h_curr;
                    S_t[i] *= std::exp(drift * dt + std::sqrt(h_curr) * epsilon[i] * sqrt_dt);
                }

                X_t = next_X_t;
            }

            total_payoff_accumulator += payoff_lambda(S_t);
        }

        double expected_payoff = total_payoff_accumulator / static_cast<double>(num_simulations);
        double continuous_discount = std::exp(-market_universe[0].r * maturity_years);

        return expected_payoff * continuous_discount;
    }
};

/*

class NetworkMonteCarloPricer {
private:
    std::vector<MarketData> market_universe;
    NetworkLogArchConfig config;
    size_t num_assets;

    // Matrix inversion helper utility using standard Gauss-Jordan elimination
    std::vector<std::vector<double>> compute_matrix_inverse(const std::vector<std::vector<double>>& mat) {
        size_t n = mat.size();
        std::vector<std::vector<double>> A = mat;
        std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;

        for (size_t i = 0; i < n; ++i) {
            double pivot = A[i][i];
            if (std::abs(pivot) < 1e-9) {
                throw std::runtime_error("Matrix inversion failed: Near-singular network structure.");
            }
            for (size_t j = 0; j < n; ++j) {
                A[i][j] /= pivot;
                I[i][j] /= pivot;
            }
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = A[k][i];
                    for (size_t j = 0; j < n; ++j) {
                        A[k][j] -= factor * A[i][j];
                        I[k][j] -= factor * I[i][j];
                    }
                }
            }
        }
        return I;
    }

public:
    NetworkMonteCarloPricer(const std::vector<MarketData>& universe, const NetworkLogArchConfig& network_config)
        : market_universe(universe), config(network_config), num_assets(universe.size()) {

        if (num_assets == 0) throw std::invalid_argument("Market universe cannot be empty.");

        // Compute (I - \rho * W) matrix dimensions
        std::vector<std::vector<double>> I_rhoW(num_assets, std::vector<double>(num_assets, 0.0));
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                double identity_val = (i == j) ? 1.0 : 0.0;
                I_rhoW[i][j] = identity_val - (config.rho_global * config.W[i][j]);
            }
        }

        // Pre-invert the system matrix to resolve simultaneous network shocks
        config.inv_I_rhoW = compute_matrix_inverse(I_rhoW);
    }

    // Runs the multi-asset path simulation to price a derivative payoff
    double price_basket_option(size_t num_simulations, double maturity_years, size_t num_days,
        const std::function<double(const std::vector<double>&)>& payoff_lambda) {

        double dt = maturity_years / static_cast<double>(num_days);
        double sqrt_dt = std::sqrt(dt);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> normal(0.0, 1.0);

        double total_payoff_accumulator = 0.0;

        for (size_t sim = 0; sim < num_simulations; ++sim) {

            std::vector<double> S_t(num_assets);
            std::vector<double> ln_h_t(num_assets);

            for (size_t i = 0; i < num_assets; ++i) {
                S_t[i] = market_universe[i].S0;
                ln_h_t[i] = std::log(market_universe[i].sigma * market_universe[i].sigma);
            }

            for (size_t day = 0; day < num_days; ++day) {

                // 1. Generate independent risk-neutral asset return shocks (\epsilon_i)
                std::vector<double> epsilon(num_assets);
                for (size_t i = 0; i < num_assets; ++i) {
                    epsilon[i] = normal(gen);
                }

                // 2. Evolve the Volatility State Matrix
                std::vector<double> core_forcing(num_assets, 0.0);

                // NEW: Normalize the projection so the spatial inverse doesn't inflate the baseline
                double spatial_norm = 1.0 - config.rho_global;

                for (size_t i = 0; i < num_assets; ++i) {
                    // Apply normalization to the local Log-ARCH projection
                    core_forcing[i] = spatial_norm * (config.omega_baseline[i] + (config.gamma_memory[i] * ln_h_t[i]));
                }

                std::vector<double> next_ln_h_t(num_assets, 0.0);
                for (size_t i = 0; i < num_assets; ++i) {
                    for (size_t j = 0; j < num_assets; ++j) {
                        next_ln_h_t[i] += config.inv_I_rhoW[i][j] * core_forcing[j];
                    }
                }

                for (size_t i = 0; i < num_assets; ++i) {
                    next_ln_h_t[i] = std::max(-12.0, std::min(4.0, next_ln_h_t[i]));
                }

                // Step the Underlying Asset Prices Forward (Native Dividend Support Included)
                for (size_t i = 0; i < num_assets; ++i) {
                    double h_curr = std::exp(ln_h_t[i]);
                    // market_universe[i].q accurately incorporates your calibrated implied dividend
                    double drift = market_universe[i].r - market_universe[i].q - 0.5 * h_curr;

                    S_t[i] *= std::exp(drift * dt + std::sqrt(h_curr) * epsilon[i] * sqrt_dt);
                }

                ln_h_t = next_ln_h_t;
            }

            total_payoff_accumulator += payoff_lambda(S_t);
        }

        double expected_payoff = total_payoff_accumulator / static_cast<double>(num_simulations);
        double continuous_discount = std::exp(-market_universe[0].r * maturity_years);

        return expected_payoff * continuous_discount;
    }
};

*/