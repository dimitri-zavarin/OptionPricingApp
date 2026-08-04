#pragma once

#include "simulation_config.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

/**
 * @class ConfigLoader
 * @brief Loads market parameters and matrices from CSV files into SimulationConfig.
 * 
 * This class provides utility functions to:
 * - Load the lower triangular Cholesky matrix L from "cholesky_L_matrix.csv"
 * - Load the network weight matrix W from "calibrated_W_matrix.csv"
 * - Populate and validate a SimulationConfig instance
 */
class ConfigLoader {
public:
    /**
     * @brief Loads the lower triangular Cholesky decomposition matrix from CSV.
     * 
     * @param filename Path to the cholesky_L_matrix.csv file
     * @param L Reference to store the loaded N x N lower triangular matrix
     * @param num_assets Expected matrix dimension (must match configuration)
     * @return True if successful, false if file I/O or dimension mismatch occurs
     */
    static bool load_cholesky_matrix(const std::string& filename, 
                                      Eigen::MatrixXd& L, 
                                      int num_assets);

    /**
     * @brief Loads the row-normalized network weight matrix from CSV.
     * 
     * @param filename Path to the calibrated_W_matrix.csv file
     * @param W Reference to store the loaded N x N weight matrix
     * @param num_assets Expected matrix dimension (must match configuration)
     * @return True if successful, false if file I/O or dimension mismatch occurs
     */
    static bool load_network_weight_matrix(const std::string& filename, 
                                           Eigen::MatrixXd& W, 
                                           int num_assets);

    /**
     * @brief Loads both matrices into a SimulationConfig instance.
     * 
     * @param config SimulationConfig instance to populate with L and W
     * @param cholesky_filename Path to cholesky_L_matrix.csv
     * @param weight_filename Path to calibrated_W_matrix.csv
     * @return True if both matrices loaded successfully and dimensions validated
     */
    static bool load_matrices_into_config(SimulationConfig& config,
                                          const std::string& cholesky_filename,
                                          const std::string& weight_filename);

private:
    /**
     * @brief Parses a single CSV row and extracts numeric values.
     * 
     * @param line CSV row as string (comma-separated values)
     * @param values Reference vector to populate with parsed doubles
     * @param skip_first_column If true, skips the first column (ticker/index label)
     * @return True if parsing succeeded, false if malformed data
     */
    static bool parse_csv_row(const std::string& line, 
                              std::vector<double>& values, 
                              bool skip_first_column = true);
};