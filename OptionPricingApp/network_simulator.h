#pragma once

#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class NetworkSimulator {
private:
    SimulationConfig config_;

    std::vector<Eigen::MatrixXd> asset_paths_;
    std::vector<Eigen::MatrixXd> log_variance_paths_;

    std::mt19937 rng_;
    std::normal_distribution<double> std_norm_;

public:
    NetworkSimulator(const SimulationConfig& cfg, int seed = 42)
        : config_(cfg), rng_(seed), std_norm_(0.0, 1.0) {

        if (!config_.validate()) {
            throw std::invalid_argument("Simulation configuration is invalid or mismatched.");
        }

        asset_paths_.resize(config_.num_paths, Eigen::MatrixXd(config_.num_assets, config_.num_steps + 1));
        log_variance_paths_.resize(config_.num_paths, Eigen::MatrixXd(config_.num_assets, config_.num_steps + 1));
    }

    void simulate() {
        double sqrt_dt = std::sqrt(config_.dt);

        for (int p = 0; p < config_.num_paths; ++p) {
            asset_paths_[p].col(0) = config_.S0;
            log_variance_paths_[p].col(0) = config_.X0;

            for (int t = 1; t <= config_.num_steps; ++t) {
                Eigen::VectorXd s_prev = asset_paths_[p].col(t - 1);
                Eigen::VectorXd x_prev = log_variance_paths_[p].col(t - 1);

                Eigen::VectorXd z1(config_.num_assets);
                Eigen::VectorXd z2(config_.num_assets);
                for (int i = 0; i < config_.num_assets; ++i) {
                    z1(i) = std_norm_(rng_);
                    z2(i) = std_norm_(rng_);
                }

                Eigen::VectorXd dws = sqrt_dt * z1;
                Eigen::VectorXd dwv = sqrt_dt * (config_.rho.cwiseProduct(z1) +
                    (1.0 - config_.rho.array().square()).sqrt().matrix().cwiseProduct(z2));

                Eigen::VectorXd spatial_spillover = config_.W * x_prev;
                Eigen::VectorXd dynamic_target = config_.theta + config_.gamma.cwiseProduct(spatial_spillover);

                Eigen::VectorXd x_curr = x_prev +
                    config_.kappa.cwiseProduct(dynamic_target - x_prev) * config_.dt +
                    config_.xi.cwiseProduct(dwv);

                Eigen::VectorXd v_prev = x_prev.array().exp().matrix();

                Eigen::VectorXd price_drift = (Eigen::VectorXd::Constant(config_.num_assets, config_.risk_free_rate)
                    - config_.q
                    - 0.5 * v_prev) * config_.dt;

                Eigen::VectorXd price_diffusion = v_prev.cwiseSqrt().cwiseProduct(dws);

                Eigen::VectorXd s_curr = s_prev.cwiseProduct((price_drift + price_diffusion).array().exp().matrix());

                log_variance_paths_[p].col(t) = x_curr;
                asset_paths_[p].col(t) = s_curr;
            }
        }
    }

    const std::vector<Eigen::MatrixXd>& get_asset_paths() const { return asset_paths_; }
    const std::vector<Eigen::MatrixXd>& get_log_variance_paths() const { return log_variance_paths_; }
};