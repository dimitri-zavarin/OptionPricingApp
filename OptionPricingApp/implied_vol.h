#pragma once

#include <iostream>
#include <cmath>
#include "market_data.h"
#include "greeks.h"

struct IVConfig {
    double sigma = 0.3; // initial guess
    double tol = 1e-6;
    int max_iter = 100;
    int steps = 1000;
};

template <typename OptionType, template <typename> typename PricerType>
struct IVSolver {
    static double solve(MarketData mkt,
                        const OptionType& opt,
                        double targetPrice,
                        const IVConfig& config = IVConfig()) {
    
        double sigma = config.sigma;

        for (int i = 0; i < config.max_iter; ++i) {
            mkt.sigma = sigma;

            double price = PricerType<OptionType>::price(mkt, opt, config.steps);
            double diff = price - targetPrice;

            if (std::abs(diff) < config.tol) { return sigma; }

            double vega = Greeks<OptionType, PricerType>::calculate(mkt, opt, config.steps).vega;

            if (std::abs(vega) < 1e-10) { break; }

            sigma -= diff / vega;

            if (sigma <= 0) { sigma = 1e-4; }
            if (sigma > 4.0) { sigma = 4.0; }
        }

        return sigma;
    }
};