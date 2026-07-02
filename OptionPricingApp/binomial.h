#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include "option.h"
#include "market_data.h"

template <typename OptionType>
struct BinomialPricer {
    using ExerciseType = typename OptionType::Exercise;

    static double price(const MarketData& data, const OptionType& opt, int steps) {
        double dt = opt.maturity() / steps;                                      // step size
        double u = std::exp(data.volatility_ * std::sqrt(dt));                   // upward move factor
        double d = 1.0 / u;                                                      // downward move factor
        double p = (std::exp((data.risk_free_rate_ - data.dividend_yield_) * dt) - d) / (u - d); // risk neutral prob
        double discount = std::exp(-data.risk_free_rate_ * dt);                  // discount per step

        std::vector<double> v(steps + 1);

        double top_node_price = data.s0_ * std::pow(u, steps);                   // stock price at top leaf
        double st = top_node_price;
        double ratio = d / u;

        for (int i = 0; i <= steps; ++i) {
            v[i] = opt.payoff(st);
            st *= ratio;
        }

        top_node_price /= u;

        for (int j = steps - 1; j >= 0; --j) {
            double s_j = top_node_price;

            for (int i = 0; i <= j; ++i) {
                v[i] = discount * (p * v[i] + (1 - p) * v[i + 1]);

                if constexpr (std::is_same_v<ExerciseType, American>) {
                    v[i] = std::max(v[i], opt.payoff(s_j));
                }
                s_j *= ratio;
            }
            top_node_price /= u;
        }
        return v[0];
    }
};