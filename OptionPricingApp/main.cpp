#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "market_data.h"
#include "option.h"
#include "binomial.h"
#include "black_scholes.h"
#include "greeks.h"

int main() {
    // 1. Setup Market Data (S=100, r=5%, sigma=20%, q=0%)
    MarketData mkt(100.0, 0.05, 0.20, 0.0);

    // 2. Define Instruments
    EuCall call(100.0, 1.0);  // 1-Year ATM European Call
    AmPut put(100.0, 1.0);   // 1-Year ATM American Put

    // 3. Convergence & Comparison Test
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================================================" << std::endl;
    std::cout << "PRICER COMPARISON: Binomial vs Black-Scholes" << std::endl;
    std::cout << "================================================" << std::endl;

    double bs_price = BlackScholesPricer<EuCall>::price(mkt, call);
    std::cout << "Black-Scholes Price: " << bs_price << std::endl;

    std::vector<int> steps_to_test = { 100, 1000, 5000, 10000 };
    for (int n : steps_to_test) {
        double bin_price = BinomialPricer<EuCall>::price(mkt, call, n);
        double error = std::abs(bin_price - bs_price);
        std::cout << "Steps: " << std::setw(6) << n
            << " | Binomial: " << bin_price
            << " | Error: " << error << std::endl;
    }

    // 4. Greeks Modularity Test
    // We run the Greeks calculator twice: once using the Binomial Engine 
    // and once using the Black-Scholes Analytical Engine.
    std::cout << "\n================================================" << std::endl;
    std::cout << "MODULAR GREEKS: European Call" << std::endl;
    std::cout << "================================================" << std::endl;

    auto bin_greeks = Greeks<EuCall, BinomialPricer>::calculate(mkt, call, 10000);
    auto bs_greeks = Greeks<EuCall, BlackScholesPricer>::calculate(mkt, call, 0); // steps=0 for BS

    std::cout << std::setw(10) << "" << std::setw(15) << "Binomial" << std::setw(15) << "Analytical" << std::endl;
    std::cout << "Delta: " << std::setw(15) << bin_greeks.delta << std::setw(15) << bs_greeks.delta << std::endl;
    std::cout << "Gamma: " << std::setw(15) << bin_greeks.gamma << std::setw(15) << bs_greeks.gamma << std::endl;
    std::cout << "Vega:  " << std::setw(15) << bin_greeks.vega << std::setw(15) << bs_greeks.vega << std::endl;

    // 5. American Put Test
    std::cout << "\n================================================" << std::endl;
    std::cout << "AMERICAN PUT (Early Exercise Premium)" << std::endl;
    std::cout << "================================================" << std::endl;

    EuPut euro_put(100.0, 1.0);
    double ep = BinomialPricer<EuPut>::price(mkt, euro_put, 1000);
    double ap = BinomialPricer<AmPut>::price(mkt, put, 1000);

    std::cout << "European Put: " << ep << std::endl;
    std::cout << "American Put: " << ap << std::endl;
    std::cout << "Premium:      " << (ap - ep) << std::endl;

    return 0;
}