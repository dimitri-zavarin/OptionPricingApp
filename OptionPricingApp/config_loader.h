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

// Structure to hold our option calibration targets
struct VanillaOptionData {
    std::string ticker;
    double strike;
    double maturity;
    double targetPrice;
    int isCall;
};

struct NetworkConfigLoader {

    // Reads network_config.csv (Ticker, S0, v0, omega_baseline, gamma_memory, r, q)
    static void load_system_config(const std::string& filepath,
        std::vector<MarketData>& market_universe,
        std::vector<std::string>& tickers,
        NetworkLogArchConfig& config) {

        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Critical Error: Unable to open configuration path: " + filepath);
        }

        std::string line;
        // Strip the Python column headers
        if (!std::getline(file, line)) return;

        market_universe.clear();
        tickers.clear();
        config.omega_baseline.clear();
        config.gamma_memory.clear();

        while (std::getline(file, line)) {
            if (line.empty()) continue; // Skip trailing blank lines
            std::stringstream ss(line);
            std::string cell;

            std::string ticker;
            double S0, v0, omega_b, gamma_m, r, q;

            std::getline(ss, ticker, ',');
            std::getline(ss, cell, ','); S0 = std::stod(cell);
            std::getline(ss, cell, ','); v0 = std::stod(cell);
            std::getline(ss, cell, ','); omega_b = std::stod(cell);
            std::getline(ss, cell, ','); gamma_m = std::stod(cell);
            std::getline(ss, cell, ','); r = std::stod(cell);
            std::getline(ss, cell, ','); q = std::stod(cell);

            tickers.push_back(ticker);
            config.omega_baseline.push_back(omega_b);
            config.gamma_memory.push_back(gamma_m);

            // Construct our market data block, natively passing the imported dividend yield
            MarketData asset_data(S0, r, std::sqrt(v0), q);
            market_universe.push_back(asset_data);
        }
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
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stod(cell));
            }
            config.W.push_back(row);
        }
    }

    // Reads calibration_options.csv to feed the implied_div.h solver
    static std::vector<VanillaOptionData> load_calibration_options(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Critical Error: Unable to open calibration options path: " + filepath);
        }

        std::vector<VanillaOptionData> contracts;
        std::string line;

        // Strip the Python column headers
        if (!std::getline(file, line)) return contracts;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            VanillaOptionData opt;

            std::getline(ss, opt.ticker, ',');
            std::getline(ss, cell, ','); opt.strike = std::stod(cell);
            std::getline(ss, cell, ','); opt.maturity = std::stod(cell);
            std::getline(ss, cell, ','); opt.targetPrice = std::stod(cell);
            std::getline(ss, cell, ','); opt.isCall = std::stoi(cell);

            contracts.push_back(opt);
        }
        return contracts;
    }
};