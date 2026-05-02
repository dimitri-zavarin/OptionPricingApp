#pragma once

#include "market_data.h"
#include "binomial.h"

template <typename OptionType, template <typename> typename PricerType>
struct Delta {
	static double calculate(MarketData mkt, const OptionType& opt, int steps, double shift = 0.01) {
		double dS = mkt.S0 * shift;

		mkt.S0 += dS;
		double up_price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.S0 -= 2.0 * dS;
		double down_price = PricerType<OptionType>::price(mkt, opt, steps);

		// central difference approximation
		// f'(x) = (f(x+h) - f(x-h)) / 2h

		return (up_price - down_price) / (2.0 * dS);
	}
};

template <typename OptionType, template <typename> typename PricerType>
struct Gamma {
	static double calculate(MarketData mkt, const OptionType& opt, int steps, double shift = 0.01) {
		double dS = mkt.S0 * shift;
		double price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.S0 += dS;
		double up_price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.S0 -= 2.0 * dS;
		double down_price = PricerType<OptionType>::price(mkt, opt, steps);

		// central difference approximation
		// f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2

		return (up_price - 2.0 * price + down_price) / (dS * dS);
	}
};

template <typename OptionType, template <typename> typename PricerType>
struct Vega {
	static double calculate(MarketData mkt, const OptionType& opt, int steps, double shift = 0.01) {
		double dSigma = mkt.sigma * shift;
		if (dSigma < 1e-10) { dSigma = 1e-4; }

		mkt.sigma += dSigma;
		double up_price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.sigma -= 2.0 * dSigma;
		double down_price = PricerType<OptionType>::price(mkt, opt, steps);

		// central difference approximation
		// f'(x) = (f(x+h) - f(x-h)) / 2h

		return (up_price - down_price) / (2.0 * dSigma);
	}
};

template <typename OptionType, template <typename> typename PricerType>
struct Epsilon {
	static double calculate(MarketData mkt, const OptionType& opt, int steps, double dq = 0.0001) {
		mkt.q += dq;
		double up_price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.q -= 2.0 * dq;
		double down_price = PricerType<OptionType>::price(mkt, opt, steps);

		// central difference approximation
		// f'(x) = (f(x+h) - f(x-h)) / 2h

		return (up_price - down_price) / (2.0 * dq);
	}
};