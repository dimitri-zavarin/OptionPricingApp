#pragma once

#include "simulation_config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

class EuropeanPricer {
private:
    const SimulationConfig& config;
    const std::vector<Eigen::MatrixXd>& asset_paths;

public:
    EuropeanPricer(const SimulationConfig& cfg, const std::vector<Eigen::MatrixXd>& paths)
        : config(cfg), asset_paths(paths) {
    }

    Eigen::VectorXd price(double strike, bool is_call) const {
        Eigen::VectorXd option_prices = Eigen::VectorXd::Zero(config.num_assets);
        double T = config.num_steps * config.dt;
        double discount_factor = std::exp(-config.risk_free_rate * T);

        for (int p = 0; p < config.num_paths; ++p) {
            Eigen::VectorXd S_T = asset_paths[p].col(config.num_steps);

            // Fix: Use an explicit if-else block to isolate the different template types
            Eigen::VectorXd payoffs(config.num_assets);
            if (is_call) {
                payoffs = (S_T.array() - strike).cwiseMax(0.0).matrix();
            }
            else {
                payoffs = (strike - S_T.array()).cwiseMax(0.0).matrix();
            }

            option_prices += payoffs;
        }

        return (option_prices / static_cast<double>(config.num_paths)) * discount_factor;
    }
};