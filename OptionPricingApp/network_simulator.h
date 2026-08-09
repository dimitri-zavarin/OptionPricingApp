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
 * ARCHITECTURE: Both asset return and volatility processes use Cholesky-transformed
 * shocks (z1_corr = L * z1), ensuring:
 *   - Multi-asset return correlations via empirical L matrix
 *   - Heston leverage effect (rho) preserved within each asset
 *   - Cross-asset volatility spillovers via network matrix W
 */
class NetworkSimulator {
private:
    SimulationConfig config_;                           ///< Configuration struct with market and network parameters

    std::vector<Eigen::MatrixXd> asset_paths_;          ///< Simulated asset price paths [num_paths x num_assets x (num_steps+1)]
    std::vector<Eigen::MatrixXd> log_variance_paths_;   ///< Simulated log-variance paths [num_paths x num_assets x (num_steps+1)]

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
                // Generate two independent vectors of N(0,1) variates
                Eigen::VectorXd z1(config_.num_assets);
                Eigen::VectorXd z2(config_.num_assets);
                for (int i = 0; i < config_.num_assets; ++i) {
                    z1(i) = std_norm_(rng_);
                    z2(i) = std_norm_(rng_);
                }

                // ============================================================================
                // STEP 2: Apply Cholesky Transformation for Multi-Asset Correlations
                // ============================================================================
                // Transform independent shocks into Cholesky-correlated shocks:
                //   z1_corr = L * z1, where L is lower triangular from Cholesky(Sigma)
                // This induces empirical return correlations across all assets
                Eigen::VectorXd z1_corr = config_.L * z1;

                // ============================================================================
                // STEP 3: Construct Brownian Increments for Asset Returns
                // ============================================================================
                // dW_S = sqrt(dt) * z1_corr
                // Asset returns now exhibit multi-asset correlations via L matrix
                Eigen::VectorXd dws = sqrt_dt * z1_corr;

                // ============================================================================
                // STEP 4: Construct Brownian Increments for Volatility (Heston Leverage)
                // ============================================================================
                // dW_v = rho * sqrt(dt) * z1_corr + sqrt(1 - rho^2) * sqrt(dt) * z2
                // KEY: Uses z1_corr (NOT original z1) to preserve Heston structure while
                // incorporating multi-asset correlations. This ensures:
                //   - Correlation within asset i: Corr(dW_S_i, dW_v_i) = rho_i ✓
                //   - Correlation across assets: depends on L matrix ✓
                Eigen::VectorXd dwv = sqrt_dt * (config_.rho.cwiseProduct(z1_corr) +
                    (1.0 - config_.rho.array().square()).sqrt().matrix().cwiseProduct(z2));

                // ============================================================================
                // STEP 5: Compute Spatiotemporal Volatility Contagion via Network Matrix
                // ============================================================================
                // Network-weighted spillover captures how volatility shocks propagate across assets
                Eigen::VectorXd spatial_spillover = config_.W * x_prev;
                Eigen::VectorXd dynamic_target = config_.theta + config_.gamma.cwiseProduct(spatial_spillover - x_prev);

                // ============================================================================
                // STEP 6: Update Latent Log-Variance Network via Euler-Maruyama
                // ============================================================================
                // dX_t = kappa * (dynamic_target - X_t) * dt + xi * dW_v
                // The coupled network system propagates volatility shocks via W matrix
                Eigen::VectorXd x_curr = x_prev +
                    config_.kappa.cwiseProduct(dynamic_target - x_prev) * config_.dt +
                    config_.xi.cwiseProduct(dwv);

                // ============================================================================
                // STEP 7: Transform Log-Variance to Variance Space
                // ============================================================================
                // V_prev = exp(X_prev) ensures strict positivity
                Eigen::VectorXd v_prev = x_prev.array().exp().matrix();

                // ============================================================================
                // STEP 8: Advance Asset Prices via Log-Normal Exact Discretization
                // ============================================================================
                // Drift component: includes risk-neutral rate, dividend yield, and Itô correction
                Eigen::VectorXd price_drift = (Eigen::VectorXd::Constant(config_.num_assets, config_.risk_free_rate)
                    - config_.q
                    - 0.5 * v_prev) * config_.dt;

                // Diffusion component: sqrt(V) * dW_S
                // Uses Cholesky-correlated Brownian increments (dws) for multi-asset correlations
                Eigen::VectorXd price_diffusion = v_prev.cwiseSqrt().cwiseProduct(dws);

                // Log-normal update: S_t = S_t-1 * exp(drift + diffusion)
                // Guarantees S_t > 0 for all paths and all time steps
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