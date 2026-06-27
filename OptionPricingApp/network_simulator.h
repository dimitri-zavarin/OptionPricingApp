#pragma once

#include "simulation_config.h" // The struct we just built
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class NetworkSimulator {
private:
    SimulationConfig config;

    // === Path Storage for American Options (LSM) ===
    // Outer vector: Paths (size: num_paths)
    // Inner matrix: Rows = Assets (N), Cols = Time Steps (T+1)
    std::vector<Eigen::MatrixXd> asset_paths;

    // Optional but highly recommended: Store variance paths too.
    // LSM regressions often use current variance as a state variable!
    std::vector<Eigen::MatrixXd> log_variance_paths;

    // === Random Number Generation ===
    std::mt19937 rng; // Mersenne Twister engine
    std::normal_distribution<double> std_norm;

public:
    /**
     * @brief Constructor initializes the simulator, allocates memory,
     * and seeds the random number generator.
     */
    NetworkSimulator(const SimulationConfig& cfg, int seed = 42)
        : config(cfg), rng(seed), std_norm(0.0, 1.0) {

        // 1. Validate configuration
        if (!config.validate()) {
            throw std::invalid_argument("Simulation configuration is invalid or mismatched.");
        }

        // 2. Pre-allocate memory for all paths to prevent reallocation overhead
        asset_paths.resize(config.num_paths, Eigen::MatrixXd(config.num_assets, config.num_steps + 1));
        log_variance_paths.resize(config.num_paths, Eigen::MatrixXd(config.num_assets, config.num_steps + 1));
    }

    /**
     * @brief The core Monte Carlo engine. Executes the Euler-Maruyama
     * discretization of the multi-asset spatiotemporal Heston model.
     */
    void simulate() {
        double sqrt_dt = std::sqrt(config.dt);

        // Loop 1: Iterate over independent Monte Carlo paths
        for (int p = 0; p < config.num_paths; ++p) {

            // Set initial t=0 states for this specific path
            asset_paths[p].col(0) = config.S0;
            log_variance_paths[p].col(0) = config.X0;

            // Loop 2: Step forward through time (Days 1 to T)
            for (int t = 1; t <= config.num_steps; ++t) {

                // Fetch yesterday's states
                Eigen::VectorXd S_prev = asset_paths[p].col(t - 1);
                Eigen::VectorXd X_prev = log_variance_paths[p].col(t - 1);

                // Initialize today's shock vectors
                Eigen::VectorXd Z1(config.num_assets);
                Eigen::VectorXd Z2(config.num_assets);
                for (int i = 0; i < config.num_assets; ++i) {
                    Z1(i) = std_norm(rng);
                    Z2(i) = std_norm(rng);
                }

                // Construct the Correlated Shocks (Leverage Effect)
                Eigen::VectorXd dWs = sqrt_dt * Z1;
                Eigen::VectorXd dWv = sqrt_dt * (config.rho.cwiseProduct(Z1) +
                    (1.0 - config.rho.array().square()).sqrt().matrix().cwiseProduct(Z2));

                // --- 1. Update Latent Log-Variance Network (X) ---
                // Calculate network spillover: W * X_prev
                Eigen::VectorXd spatial_spillover = config.W * X_prev;

                // Calculate dynamic network target: theta + gamma * (W * X_prev)
                Eigen::VectorXd dynamic_target = config.theta + config.gamma.cwiseProduct(spatial_spillover);

                // Euler step for X_t
                Eigen::VectorXd X_curr = X_prev +
                    config.kappa.cwiseProduct(dynamic_target - X_prev) * config.dt +
                    config.xi.cwiseProduct(dWv);

                // --- 2. Update Asset Prices (S) ---
                // CRITICAL FIX: Must use X_prev to satisfy Itô's Lemma and preserve Put-Call Parity.
                Eigen::VectorXd V_prev = X_prev.array().exp().matrix();

                // Drift: (config.risk_free_rate - config.q - 0.5 * V_prev) * config.dt
                Eigen::VectorXd price_drift = (Eigen::VectorXd::Constant(config.num_assets, config.risk_free_rate)
                    - config.q
                    - 0.5 * V_prev) * config.dt;

                // Diffusion: sqrt(V_prev) * dWs
                Eigen::VectorXd price_diffusion = V_prev.cwiseSqrt().cwiseProduct(dWs);

                // Geometric Euler step for S_t
                Eigen::VectorXd S_curr = S_prev.cwiseProduct((price_drift + price_diffusion).array().exp().matrix());

                // --- 3. Save states to the matrix history ---
                log_variance_paths[p].col(t) = X_curr;
                asset_paths[p].col(t) = S_curr;
            }
        }
    }

    // === Getters for the Pricer Class ===
    const std::vector<Eigen::MatrixXd>& getAssetPaths() const { return asset_paths; }
    const std::vector<Eigen::MatrixXd>& getLogVariancePaths() const { return log_variance_paths; }
};