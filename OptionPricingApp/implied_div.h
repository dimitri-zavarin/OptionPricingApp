#pragma once

#include <iostream>
#include <cmath>
#include "market_data.h"
#include "greeks.h"

struct IDivConfig {
    double q = 0.005; // initial guess
    double tol = 1e-6;
    int max_iter = 100;
    int steps = 1000;
};

template <typename OptionType, template <typename> typename PricerType>
struct IDivSolver {
    static double solve(MarketData mkt,
        const OptionType& opt,
        double targetPrice,
        const IDivConfig& config = IDivConfig()) {

        double q = config.q;

        for (int i = 0; i < config.max_iter; ++i) {
            mkt.q = q;

            double price = PricerType<OptionType>::price(mkt, opt, config.steps);
            double diff = price - targetPrice;

            if (std::abs(diff) < config.tol) { return q; }

            double epsilon = Epsilon<OptionType, PricerType>::calculate(mkt, opt, config.steps);

            if (std::abs(epsilon) < 1e-10) { break; }

            q -= diff / epsilon;

            if (q < 0) { q = 1e-6; }
            if (q > 0.5) { q = 0.5; }
        }

        return q;
    }
};