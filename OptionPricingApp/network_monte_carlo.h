#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include "market_data.h"

struct NetworkLogArchConfig {
    std::vector<std::vector<double>> W;       // Sparse, row-normalized k-NN weight matrix (NxN)
    std::vector<double> omega_baseline;       // Idiosyncratic variance floor (\omega_i)
    std::vector<double> gamma_memory;         // Standalone volatility memory retention (\gamma_i)
    double rho_global;                        // Global network instantaneous spillover intensity (\rho)

    // Identity minus Rho * W matrix, pre-inverted for fast runtime execution: (I - \rho * W)^-1
    std::vector<std::vector<double>> inv_I_rhoW;
};

class NetworkMonteCarloPricer {
private:
    std::vector<MarketData> market_universe;
    NetworkLogArchConfig config;
    size_t num_assets;

    // Matrix inversion helper utility using standard Gauss-Jordan elimination
    // Used to pre-calculate (I - \rho * W)^-1 once to solve the paper's simultaneity problem
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

        // Step 1: Compute (I - \rho * W) matrix dimensions
        std::vector<std::vector<double>> I_rhoW(num_assets, std::vector<double>(num_assets, 0.0));
        for (size_t i = 0; i < num_assets; ++i) {
            for (size_t j = 0; j < num_assets; ++j) {
                double identity_val = (i == j) ? 1.0 : 0.0;
                I_rhoW[i][j] = identity_val - (config.rho_global * config.W[i][j]);
            }
        }

        // Step 2: Pre-invert the system matrix to resolve simultaneous network shocks (Equation 10)
        config.inv_I_rhoW = compute_matrix_inverse(I_rhoW);
    }

    // Runs the multi-asset path simulation to price a derivative payoff
    double price_basket_option(size_t num_simulations, double maturity_years, size_t num_days,
        const auto& payoff_lambda) {

        double dt = maturity_years / static_cast<double>(num_days);
        double sqrt_dt = std::sqrt(dt);

        // Thread-safe random number generation structures
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> normal(0.0, 1.0);

        double total_payoff_accumulator = 0.0;

        // Core Monte Carlo Loop
        for (size_t sim = 0; sim < num_simulations; ++sim) {

            // Initialize spot prices (S0) and variance states (h_t = sigma^2) for this path run
            std::vector<double> S_t(num_assets);
            std::vector<double> ln_h_t(num_assets);

            for (size_t i = 0; i < num_assets; ++i) {
                S_t[i] = market_universe[i].S0;
                ln_h_t[i] = std::log(market_universe[i].sigma * market_universe[i].sigma);
            }

            // Time Stepping Loop (Daily increments)
            for (size_t day = 0; day < num_days; ++day) {

                // 1. Generate independent risk-neutral asset return shocks (\epsilon_i)
                std::vector<double> epsilon(num_assets);
                std::vector<double> ln_Y_squared_lagged(num_assets);

                for (size_t i = 0; i < num_assets; ++i) {
                    epsilon[i] = normal(gen);

                    // The asset return shock: Y_t = \sqrt{h_t} * \epsilon_t
                    double Y_t = std::sqrt(std::exp(ln_h_t[i])) * epsilon[i];

                    // Guard against log(0) boundaries if epsilon hits exactly 0
                    double Y_squared = (Y_t * Y_t < 1e-12) ? 1e-12 : Y_t * Y_t;
                    ln_Y_squared_lagged[i] = std::log(Y_squared);
                }

                // 2. Evolve the Volatility State Matrix using pre-computed Matrix Inversion (Equation 10)
                // This cleanly resolves the paper's instantaneous spatial endogeneity loop in O(N^2) time
                std::vector<double> core_forcing(num_assets, 0.0);
                for (size_t i = 0; i < num_assets; ++i) {
                    core_forcing[i] = config.omega_baseline[i] + (config.gamma_memory[i] * ln_Y_squared_lagged[i]);
                }

                std::vector<double> next_ln_h_t(num_assets, 0.0);
                for (size_t i = 0; i < num_assets; ++i) {
                    for (size_t j = 0; j < num_assets; ++j) {
                        next_ln_h_t[i] += config.inv_I_rhoW[i][j] * core_forcing[j];
                    }
                }

                // Enforce stationary guardrails on the newly evolved variance state bounds
                for (size_t i = 0; i < num_assets; ++i) {
                    next_ln_h_t[i] = std::max(-12.0, std::min(4.0, next_ln_h_t[i]));
                }

                // 3. Step the Underlying Asset Prices Forward via Risk-Neutral Geometrics
                for (size_t i = 0; i < num_assets; ++i) {
                    double h_curr = std::exp(ln_h_t[i]);
                    double drift = market_universe[i].r - market_universe[i].q - 0.5 * h_curr;

                    S_t[i] *= std::exp(drift * dt + std::sqrt(h_curr) * epsilon[i] * sqrt_dt);
                }

                // Update variance states for the next day's step
                ln_h_t = next_ln_h_t;
            }

            // Evaluate the multi-asset option payoff at maturity
            total_payoff_accumulator += payoff_lambda(S_t);
        }

        // Discount the expected terminal payoff back to today using risk-free compounding (r)
        double expected_payoff = total_payoff_accumulator / static_cast<double>(num_simulations);
        double continuous_discount = std::exp(-market_universe[0].r * maturity_years);

        return expected_payoff * continuous_discount;
    }
};