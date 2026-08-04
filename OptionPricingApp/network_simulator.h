#pragma once

#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

/**
 * @class NetworkSimulator
 * @brief High-performance Monte Carlo simulation engine for a multi-asset
 * basket exhibiting spatiotemporal stochastic volatility contagion.
 * 
 * This class implements a hybrid numerical scheme: an Euler-Maruyama
 * discretization for the coupled log-variance network, combined with a
 * Log-Normal Exact Discretization for the underlying asset price paths
 * to strictly prevent non-positive prices and eliminate structural drift leaks.
 * 
 * CRITICAL UPDATE: Asset price Brownian increments now incorporate direct
 * equity return correlations via Cholesky decomposition (L * z1), while
 * volatility leverage effects (rho) remain independent.
 */
class NetworkSimulator {
private:
    SimulationConfig config_;                           ///< Copied configuration struct containing market and network variables

    std::vector<Eigen::MatrixXd> asset_paths_;          ///< Simulated underlying price paths [num_paths][num_assets x (num_steps + 1)]
    std::vector<Eigen::MatrixXd> log_variance_paths_;   ///< Simulated latent log-variance paths [num_paths][num_assets x (num_steps + 1)]

    std::mt19937 rng_;
    std::normal_distribution<double> std_norm_;

public:
    /**
     * @brief Constructs the simulator and allocates memory for all Monte Carlo paths.
     * @param cfg Fully populated SimulationConfig structure.
     * @param seed Seed for the pseudo-random number generator (defaults to 42 for reproducibility).
     * @throws std::invalid_argument If the configuration parameters fail dimensions/validation checks.
     */
    NetworkSimulator(const SimulationConfig& cfg, int seed = 42)
        : config_(cfg), rng_(seed), std_norm_(0.0, 1.0) {

        // Enforce structural integrity of vectors and matrices before allocation
        if (!config_.validate()) {
            throw std::invalid_argument("Simulation configuration is invalid or mismatched.");
        }

        // Pre-allocate dense matrix storage for all paths to prevent heap fragmentation during simulation
        asset_paths_.resize(config_.num_paths, Eigen::MatrixXd(config_.num_assets, config_.num_steps + 1));
        log_variance_paths_.resize(config_.num_paths, Eigen::MatrixXd(config_.num_assets, config_.num_steps + 1));
    }

    /**
     * @brief Executes the vectorized spatiotemporal path generation loop.
     * 
     * Iterates across all independent paths and sequentially marches the entire multi-asset system
     * forward day-by-day. Employs a left-endpoint rule for volatility evaluation to preserve
     * the non-anticipating property of Itô integration and fulfill Put-Call Parity conditions.
     * 
     * KEY MODIFICATION: Incorporates Cholesky decomposition L for correlated equity returns
     * while maintaining independent Heston leverage effects (rho).
     */
    void simulate() {
        double sqrt_dt = std::sqrt(config_.dt);

        // --- Outer Loop: Independent Monte Carlo Trajectories ---
        for (int p = 0; p < config_.num_paths; ++p) {
            // Initialize t=0 boundary conditions
            asset_paths_[p].col(0) = config_.S0;
            log_variance_paths_[p].col(0) = config_.X0;

            // --- Inner Loop: Temporal Propagation (Day-by-Day Marching) ---
            for (int t = 1; t <= config_.num_steps; ++t) {
                // Fetch states at the left-endpoint (t-1) to preserve adaptedness
                Eigen::VectorXd s_prev = asset_paths_[p].col(t - 1);
                Eigen::VectorXd x_prev = log_variance_paths_[p].col(t - 1);

                // ============================================================================
                // STEP 1: Generate Independent Standard Normal Shocks
                // ============================================================================
                // Generate two independent vectors of N(0,1) variates:
                //   z1: Used for BOTH (a) equity return correlations and (b) leverage effects
                //   z2: Used for volatility process orthogonal component
                Eigen::VectorXd z1(config_.num_assets);
                Eigen::VectorXd z2(config_.num_assets);
                for (int i = 0; i < config_.num_assets; ++i) {
                    z1(i) = std_norm_(rng_);
                    z2(i) = std_norm_(rng_);
                }

                // ============================================================================
                // STEP 2: Apply Cholesky Decomposition for Correlated Asset Returns
                // ============================================================================
                // Transform independent shocks into correlated shocks:
                //   z1_corr = L * z1, where L is lower triangular from Cholesky(correlation_matrix)
                // This ensures asset price returns exhibit the empirically observed correlation structure
                Eigen::VectorXd z1_correlated = config_.L * z1;

                // ============================================================================
                // STEP 3: Construct Brownian Increments for Asset Returns
                // ============================================================================
                // dW_S = sqrt(dt) * z1_corr
                // Asset returns now incorporate direct contemporaneous correlations
                Eigen::VectorXd dws = sqrt_dt * z1_correlated;

                // ============================================================================
                // STEP 4: Construct Brownian Increments for Volatility Process (Heston Leverage)
                // ============================================================================
                // dW_v = rho * sqrt(dt) * z1_original + sqrt(1 - rho^2) * sqrt(dt) * z2
                // CRITICAL: We use ORIGINAL z1 (NOT z1_correlated) to maintain Heston structure.
                // This ensures the leverage effect (return-variance correlation) operates
                // at the asset-specific level independently of cross-asset correlations.
                Eigen::VectorXd dwv = sqrt_dt * (config_.rho.cwiseProduct(z1) +
                    (1.0 - config_.rho.array().square()).sqrt().matrix().cwiseProduct(z2));

                // ============================================================================
                // STEP 5: Compute Spatiotemporal Volatility Contagion via Network Matrix W
                // ============================================================================
                // Spatial spillover models how volatility innovations propagate across the network
                // via the row-normalized weight matrix W (Diebold-Yilmaz spillovers from IV data)
                Eigen::VectorXd spatial_spillover = config_.W * x_prev;
                Eigen::VectorXd dynamic_target = config_.theta + config_.gamma.cwiseProduct(spatial_spillover - x_prev);

                // ============================================================================
                // STEP 6: Update Latent Log-Variance Network via Euler-Maruyama
                // ============================================================================
                // dX_t = kappa * (dynamic_target - X_t) * dt + xi * dW_v
                // The coupled network system allows volatility shocks to transmit across assets
                Eigen::VectorXd x_curr = x_prev +
                    config_.kappa.cwiseProduct(dynamic_target - x_prev) * config_.dt +
                    config_.xi.cwiseProduct(dwv);

                // ============================================================================
                // STEP 7: Transform Log-Variance to Variance Space
                // ============================================================================
                // V_prev = exp(X_prev) ensures strict positivity and prevents pathological prices
                Eigen::VectorXd v_prev = x_prev.array().exp().matrix();

                // ============================================================================
                // STEP 8: Advance Asset Prices via Log-Normal Exact Discretization
                // ============================================================================
                // Drift: S_t drift component includes risk-neutral drift, dividend yield, and Itô correction
                Eigen::VectorXd price_drift = (Eigen::VectorXd::Constant(config_.num_assets, config_.risk_free_rate)
                    - config_.q
                    - 0.5 * v_prev) * config_.dt;

                // Diffusion: sqrt(V_t-1) * dW_S
                // Left-endpoint rule: use v_prev (not v_curr) to avoid look-ahead bias
                // The correlated Brownian increments (dws) now capture multi-asset correlations
                Eigen::VectorXd price_diffusion = v_prev.cwiseSqrt().cwiseProduct(dws);

                // Geometric update: S_t = S_t-1 * exp(drift + diffusion)
                // Log-normal discretization strictly enforces S_t > 0
                Eigen::VectorXd s_curr = s_prev.cwiseProduct((price_drift + price_diffusion).array().exp().matrix());

                // ============================================================================
                // STEP 9: Persist Updated States
                // ============================================================================
                log_variance_paths_[p].col(t) = x_curr;
                asset_paths_[p].col(t) = s_curr;
            }
        }
    }

    /**
     * @brief Accessor to retrieve the complete simulated underlying asset price tensor.
     * @return Const reference to a vector of matrices containing the asset price trajectories.
     */
    const std::vector<Eigen::MatrixXd>& get_asset_paths() const { return asset_paths_; }

    /**
     * @brief Accessor to retrieve the complete simulated latent log-variance paths.
     * @return Const reference to a vector of matrices containing the log-variance trajectories.
     */
    const std::vector<Eigen::MatrixXd>& get_log_variance_paths() const { return log_variance_paths_; }
};