#include "config_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

bool ConfigLoader::parse_csv_row(const std::string& line, 
								  std::vector<double>& values, 
								  bool skip_first_column) {
	values.clear();
	std::stringstream ss(line);
	std::string cell;
	int col = 0;

	while (std::getline(ss, cell, ',')) {
		// Skip first column if it contains row labels (tickers/indices)
		if (col == 0 && skip_first_column) {
			col++;
			continue;
		}

		try {
			// Handle potential whitespace
			cell.erase(0, cell.find_first_not_of(" \t\r\n"));
			cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

			if (!cell.empty()) {
				values.push_back(std::stod(cell));
			}
		} catch (const std::invalid_argument& e) {
			std::cerr << "[ERROR] Failed to parse numeric value: '" << cell << "'" << std::endl;
			return false;
		}
		col++;
	}

	return !values.empty();
}


bool ConfigLoader::load_cholesky_matrix(const std::string& filename, 
										Eigen::MatrixXd& L, 
										int num_assets) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "[ERROR] Cannot open Cholesky matrix file: " << filename << std::endl;
		return false;
	}

	std::cout << "[INFO] Loading Cholesky matrix L from: " << filename << std::endl;

	L.resize(num_assets, num_assets);
	std::string line;
	int row = 0;

	// Skip header line (contains ticker column names)
	if (!std::getline(file, line)) {
		std::cerr << "[ERROR] File is empty or cannot read header: " << filename << std::endl;
		return false;
	}

	// Parse data rows
	while (std::getline(file, line) && row < num_assets) {
		if (line.empty() || line[0] == '#') continue;  // Skip empty/comment lines

		std::vector<double> values;
		if (!parse_csv_row(line, values, true)) {
			std::cerr << "[ERROR] Failed to parse row " << row << " in " << filename << std::endl;
			return false;
		}

		if (static_cast<int>(values.size()) != num_assets) {
			std::cerr << "[ERROR] Row " << row << " has " << values.size() 
					  << " columns, expected " << num_assets << std::endl;
			return false;
		}

		for (int col = 0; col < num_assets; ++col) {
			L(row, col) = values[col];
		}
		row++;
	}

	if (row != num_assets) {
		std::cerr << "[ERROR] Expected " << num_assets << " rows, found " << row << std::endl;
		return false;
	}

	std::cout << "[OK] Loaded Cholesky L matrix (" << num_assets << " x " << num_assets << ")" << std::endl;

	// Diagnostic: print first few entries
	std::cout << "      L[0:3, 0:3]:\n";
	for (int i = 0; i < std::min(3, num_assets); ++i) {
		std::cout << "      ";
		for (int j = 0; j < std::min(3, num_assets); ++j) {
			std::cout << std::setw(10) << std::fixed << std::setprecision(6) << L(i, j);
		}
		std::cout << "\n";
	}

	return true;
}


bool ConfigLoader::load_network_weight_matrix(const std::string& filename, 
											  Eigen::MatrixXd& W, 
											  int num_assets) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "[ERROR] Cannot open network weight matrix file: " << filename << std::endl;
		return false;
	}

	std::cout << "[INFO] Loading network weight matrix W from: " << filename << std::endl;

	W.resize(num_assets, num_assets);
	std::string line;
	int row = 0;

	// Skip header line (contains ticker column names)
	if (!std::getline(file, line)) {
		std::cerr << "[ERROR] File is empty or cannot read header: " << filename << std::endl;
		return false;
	}

	// Parse data rows
	while (std::getline(file, line) && row < num_assets) {
		if (line.empty() || line[0] == '#') continue;  // Skip empty/comment lines

		std::vector<double> values;
		if (!parse_csv_row(line, values, true)) {
			std::cerr << "[ERROR] Failed to parse row " << row << " in " << filename << std::endl;
			return false;
		}

		if (static_cast<int>(values.size()) != num_assets) {
			std::cerr << "[ERROR] Row " << row << " has " << values.size() 
					  << " columns, expected " << num_assets << std::endl;
			return false;
		}

		for (int col = 0; col < num_assets; ++col) {
			W(row, col) = values[col];
		}
		row++;
	}

	if (row != num_assets) {
		std::cerr << "[ERROR] Expected " << num_assets << " rows, found " << row << std::endl;
		return false;
	}

	std::cout << "[OK] Loaded network weight matrix W (" << num_assets << " x " << num_assets << ")" << std::endl;

	// Diagnostic: verify row-stochasticity (rows should sum to ~1.0)
	std::cout << "      Row sums (should be ~1.0):\n";
	for (int i = 0; i < std::min(3, num_assets); ++i) {
		double row_sum = W.row(i).sum();
		std::cout << "      Row " << i << ": " << std::fixed << std::setprecision(6) << row_sum << "\n";
	}

	return true;
}


bool ConfigLoader::load_matrices_into_config(SimulationConfig& config,
											 const std::string& cholesky_filename,
											 const std::string& weight_filename) {
	std::cout << "\n" << std::string(80, '=') << std::endl;
	std::cout << " LOADING MATRICES INTO SimulationConfig" << std::endl;
	std::cout << std::string(80, '=') << std::endl;

	// Load Cholesky matrix L
	if (!load_cholesky_matrix(cholesky_filename, config.L, config.num_assets)) {
		std::cerr << "[FATAL] Failed to load Cholesky matrix L" << std::endl;
		return false;
	}

	// Load network weight matrix W
	if (!load_network_weight_matrix(weight_filename, config.W, config.num_assets)) {
		std::cerr << "[FATAL] Failed to load network weight matrix W" << std::endl;
		return false;
	}

	// Validate the updated configuration
	if (!config.validate()) {
		std::cerr << "[FATAL] Configuration validation failed after loading matrices" << std::endl;
		return false;
	}

	std::cout << "\n[SUCCESS] Both matrices loaded and validated successfully!" << std::endl;
	std::cout << std::string(80, '=') << "\n" << std::endl;

	return true;
}