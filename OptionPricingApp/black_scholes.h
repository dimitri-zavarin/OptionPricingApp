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
        double s = data.s0_;
        double k = opt.strike();
        double t = opt.maturity();
        double r = data.risk_free_rate_;
        double sigma = data.volatility_;
        double q = data.dividend_yield_;

        double d1 = (std::log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t));
        double d2 = d1 - sigma * std::sqrt(t);

        if constexpr (std::is_same_v<PayoffType, CallPayoff>) {
            return s * std::exp(-q * t) * norm_cdf(d1) - k * std::exp(-r * t) * norm_cdf(d2);
        } else {
            return k * std::exp(-r * t) * norm_cdf(-d2) - s * std::exp(-q * t) * norm_cdf(-d1);
        }
    }
};