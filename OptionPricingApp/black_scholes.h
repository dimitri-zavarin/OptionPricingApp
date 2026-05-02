#pragma once

#include <cmath>
#include <algorithm>
#include <type_traits>
#include "option.h"
#include "market_data.h"

inline double norm_cdf(double x) {
	return 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
}

template <typename T>
concept EuropeanOption = std::is_same_v<typename T::Exercise, European>;

template <EuropeanOption OptionType>
struct BlackScholesPricer {
    using PayoffType = typename OptionType::Payoff;

    static double price(const MarketData& data, const OptionType& opt, int steps = 0) {
        // steps not relevant, kept for consistent calculate function calls
        double S = data.S0;
        double K = opt.K;
        double T = opt.T;
        double r = data.r;
        double sigma = data.sigma;
        double q = data.q;

        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        if constexpr (std::is_same_v<PayoffType, CallPayoff>) {
            return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        }
        else {
            return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
        }
    }
};