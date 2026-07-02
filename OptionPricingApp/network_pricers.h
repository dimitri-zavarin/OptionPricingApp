#pragma once

#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

class EuropeanPricer {
private:
    const SimulationConfig& config_;
    const std::vector<Eigen::MatrixXd>& asset_paths_;

public:
    EuropeanPricer(const SimulationConfig& cfg, const std::vector<Eigen::MatrixXd>& paths)
        : config_(cfg), asset_paths_(paths) {
    }

    Eigen::VectorXd price(double strike, bool is_call) const {
        Eigen::VectorXd option_prices = Eigen::VectorXd::Zero(config_.num_assets);
        double t = config_.num_steps * config_.dt;
        double discount_factor = std::exp(-config_.risk_free_rate * t);

        for (int p = 0; p < config_.num_paths; ++p) {
            Eigen::VectorXd s_t = asset_paths_[p].col(config_.num_steps);

            Eigen::VectorXd payoffs(config_.num_assets);
            if (is_call) {
                payoffs = (s_t.array() - strike).cwiseMax(0.0).matrix();
            } else {
                payoffs = (strike - s_t.array()).cwiseMax(0.0).matrix();
            }

            option_prices += payoffs;
        }

        return (option_prices / static_cast<double>(config_.num_paths)) * discount_factor;
    }
};