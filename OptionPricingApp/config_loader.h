#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "market_data.h"
#include "network_monte_carlo.h"

struct NetworkConfigLoader {
    // Reads network_config.csv and maps properties directly to our Multi-Asset Market Space
    static void load_system_config(const std::string& filepath,
        std::vector<MarketData>& market_universe,
        std::vector<std::string>& tickers,
        NetworkLogArchConfig& config,
        std::vector<double>& opt_strikes,
        std::vector<double>& opt_maturities,
        std::vector<double>& opt_target_prices) {

        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Critical Error: Unable to open configuration path: " + filepath);
        }

        std::string line;
        // Strip the Python column headers: Ticker,S0,v0,omega_baseline,gamma_memory,r,opt_strike,opt_maturity,opt_target_price
        if (!std::getline(file, line)) return;

        market_universe.clear();
        tickers.clear();
        config.omega_baseline.clear();
        config.gamma_memory.clear();
        opt_strikes.clear();
        opt_maturities.clear();
        opt_target_prices.clear();

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;

            std::string ticker;
            double S0, v0, omega_b, gamma_m, r;
            double o_strike, o_mat, o_target;

            std::getline(ss, ticker, ',');
            std::getline(ss, cell, ','); S0 = std::stod(cell);
            std::getline(ss, cell, ','); v0 = std::stod(cell);
            std::getline(ss, cell, ','); omega_b = std::stod(cell);
            std::getline(ss, cell, ','); gamma_m = std::stod(cell);
            std::getline(ss, cell, ','); r = std::stod(cell);
            std::getline(ss, cell, ','); o_strike = std::stod(cell);
            std::getline(ss, cell, ','); o_mat = std::stod(cell);
            std::getline(ss, cell, ','); o_target = std::stod(cell);

            tickers.push_back(ticker);
            config.omega_baseline.push_back(omega_b);
            config.gamma_memory.push_back(gamma_m);
            opt_strikes.push_back(o_strike);
            opt_maturities.push_back(o_mat);
            opt_target_prices.push_back(o_target);

            // Construct our market data block using your exact structure parameters
            MarketData asset_data(S0, r, std::sqrt(v0), 0.0);
            market_universe.push_back(asset_data);
        }
        file.close();
    }

    // Reads weight_matrix.csv and builds the 2D Adjacency Array (W)
    static void load_weight_matrix(const std::string& filepath, NetworkLogArchConfig& config) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Critical Error: Unable to open matrix target pathway: " + filepath);
        }

        config.W.clear();
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stod(cell));
            }
            config.W.push_back(row);
        }
        file.close();
    }
};