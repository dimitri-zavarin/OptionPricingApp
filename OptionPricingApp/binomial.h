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
		double dt = opt.T / steps;										// step size
		double u = std::exp(data.sigma * std::sqrt(dt));				// upward move factor
		double d = 1.0 / u;												// downward move factor
		double p = (std::exp((data.r - data.q) * dt) - d) / (u - d);	// risk free probability of upward move
		double discount = std::exp(-data.r * dt);						// discount rate per step

		// (steps+1) length vector to store option prices at each tree node
		std::vector<double> v(steps + 1);

		double topNodePrice = data.S0 * std::pow(u, steps);		// stock price at top leaf
		double ST = topNodePrice;
		double ratio = d / u;									// factor to obtain stock price at the leaf below

		for (int i = 0; i <= steps; ++i) {
			v[i] = opt.payoff(ST);
			ST *= ratio;
		}

		// calculate top node price for (steps-1)th row
		topNodePrice /= u;

		// backward induction
		for (int j = steps - 1; j >= 0; --j) {
			double Sj = topNodePrice;

			for (int i = 0; i <= j; ++i) {
				v[i] = discount * (p * v[i] + (1 - p) * v[i + 1]);

				// check early exercise for American options
				if constexpr (std::is_same_v<ExerciseType, American>) {
					v[i] = std::max(v[i], opt.payoff(Sj));
					Sj *= ratio;
				}
			}
			topNodePrice /= u;
		}
		return v[0]; // return current option price
	}
};