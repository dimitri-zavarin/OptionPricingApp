#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @brief Container holding all market constants and spatiotemporal
 * Heston network parameters for an N-asset basket.
 */
struct SimulationConfig {
    // === Market & Simulation Dimensions ===
    int num_assets;           // Number of stocks in the basket (N)
    int num_paths;            // Total Monte Carlo paths (e.g., 50,000)
    int num_steps;            // Number of daily time steps to maturity (T)
    double dt;                // Temporal increment (typically 1.0 / 252.0)
    double risk_free_rate;    // Risk-free interest rate (r)

    // === Multi-Asset Market Vectors (N x 1) ===
    Eigen::VectorXd S0;       // Initial asset prices at t=0
    Eigen::VectorXd q;        // Dividend yields for each asset

    // === Heston Network Parameter Vectors (N x 1) ===
    Eigen::VectorXd kappa;    // Mean-reversion speeds
    Eigen::VectorXd theta;    // Idiosyncratic long-run log-variance targets
    Eigen::VectorXd gamma;    // Network coupling intensities
    Eigen::VectorXd xi;       // Volatility-of-volatility scales
    Eigen::VectorXd rho;      // Asset-specific leverage correlation coefficients
    Eigen::VectorXd X0;       // Initial log-variance states at t=0 (ln(v0))

    // === Static Network Topology Matrix (N x N) ===
    Eigen::MatrixXd W;        // Row-normalized spatial weight matrix

    // === Metadata ===
    std::vector<std::string> tickers; // Order matching the matrix rows

    /**
     * @brief Validates the dimensions of the parameter vectors and matrices
     * to prevent out-of-bounds runtime memory faults in Eigen loops.
     */
    bool validate() const {
        if (num_assets <= 0 || num_paths <= 0 || num_steps <= 0) return false;
        if (S0.size() != num_assets) return false;
        if (q.size() != num_assets) return false;
        if (kappa.size() != num_assets) return false;
        if (theta.size() != num_assets) return false;
        if (gamma.size() != num_assets) return false;
        if (xi.size() != num_assets) return false;
        if (rho.size() != num_assets) return false;
        if (X0.size() != num_assets) return false;
        if (W.rows() != num_assets || W.cols() != num_assets) return false;
        return true;
    }
};