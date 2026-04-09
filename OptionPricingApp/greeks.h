#pragma once

#include "binomial.h"

struct GreeksConfig {
	double S_factor = 0.01;		// shift factor for stock price
	double sigma_factor = 0.01;	// shift factor for volatility
};

template <typename OptionType, template <typename> typename PricerType>
struct Greeks {
	struct Values {
		double delta;
		double gamma;
		double vega;
	};

	static Values calculate(const MarketData& data, const OptionType& opt, int steps, const GreeksConfig& config = GreeksConfig()) {
		double S = data.S0;
		double sigma = data.sigma;
		double dS = S * config.S_factor;
		double dsigma = sigma * config.sigma_factor;

		double opt_price = PricerType<OptionType>::price(data, opt, steps);

		// Option price if stock price bumped up
		MarketData S_up = data;
		S_up.S0 = S + dS;
		double opt_S_up = PricerType<OptionType>::price(S_up, opt, steps);

		// Option price if stock price bumped down
		MarketData S_down = data;
		S_down.S0 = S - dS;
		double opt_S_down = PricerType<OptionType>::price(S_down, opt, steps);

		// Option price if volatility bumped up
		MarketData sigma_up = data;
		sigma_up.sigma = sigma + dsigma;
		double opt_sigma_up = PricerType<OptionType>::price(sigma_up, opt, steps);

		// Option price if volatility bumped down
		MarketData sigma_down = data;
		sigma_down.sigma = sigma - dsigma;
		double opt_sigma_down = PricerType<OptionType>::price(sigma_down, opt, steps);

		// Approximate Greeks via central difference
		Values results;

		// d(option price) / d(stock price)
		results.delta = (opt_S_up - opt_S_down) / (2.0 * dS);

		// d^2(option price) / d(stock price)^2
		results.gamma = (opt_S_up - 2.0 * opt_price + opt_S_down) / (dS * dS);

		// d(option price) / d(volatility)
		results.vega = (opt_sigma_up - opt_sigma_down) / (2.0 * dsigma); 

		return results;
	}
};