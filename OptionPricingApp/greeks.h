#pragma once

#include "market_data.h"
#include "binomial.h"

// Flags for each Greek
enum GreekFlags {
	G_DELTA = 1 << 0, // 00001
	G_GAMMA = 1 << 1, // 00010
	G_VEGA = 1 << 2, // 00100
	G_THETA = 1 << 3, // 01000
	G_EPSILON = 1 << 4, // 10000
	G_ALL = 0x1F    // 11111
};

struct GreekValues {
	double delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0, epsilon = 0.0;
};

template <typename OptionType, template <typename> typename PricerType>
struct GreeksSuite {
	static GreekValues calculate(MarketData mkt, const OptionType& opt, int steps, int requested_greeks = G_ALL) {
		GreekValues values;

		double cur_price = 0.0;
		bool cur_price_needed = (requested_greeks & (G_GAMMA | G_THETA));

		if (cur_price_needed) {
			cur_price = PricerType<OptionType>::price(mkt, opt, steps);
		}

		if (requested_greeks & G_DELTA) {
			values.delta = Delta<OptionType, PricerType>::calculate(mkt, opt, steps);
		}

		if (requested_greeks & G_GAMMA) {
			values.gamma = Gamma<OptionType, PricerType>::calculate(mkt, opt, steps, cur_price);
		}

		if (requested_greeks & G_VEGA) {
			values.vega = Vega<OptionType, PricerType>::calculate(mkt, opt, steps);
		}

		if (requested_greeks & G_THETA) {
			values.theta = Theta<OptionType, PricerType>::calculate(mkt, opt, steps, cur_price);
		}

		if (requested_greeks & G_EPSILON) {
			values.epsilon = Epsilon<OptionType, PricerType>::calculate(mkt, opt, steps);
		}

		return values;
	}
};

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
	static double calculate(MarketData mkt, const OptionType& opt, int steps, double cur_price = 0.0, double shift = 0.01) {
		double dS = mkt.S0 * shift;
		if (cur_price == 0.0) {
			cur_price = PricerType<OptionType>::price(mkt, opt, steps);
		}

		mkt.S0 += dS;
		double up_price = PricerType<OptionType>::price(mkt, opt, steps);

		mkt.S0 -= 2.0 * dS;
		double down_price = PricerType<OptionType>::price(mkt, opt, steps);

		// central difference approximation
		// f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2

		return (up_price - 2.0 * cur_price + down_price) / (dS * dS);
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
struct Theta {
	static double calculate(MarketData mkt, OptionType opt, int steps, double cur_price = 0.0, double dt = 0.0001) {
		if (cur_price == 0.0) {
			cur_price = PricerType<OptionType>::price(mkt, opt, steps);
		}

		// Make sure time to maturity is not negative
		if (opt.T <= dt) { dt = opt.T; }

		// Prevent division by 0
		if (dt < 1e-10) { return 0.0; }
		
		// Advance time by dt on the option
		opt.T -= dt;

		double future_price = PricerType<OptionType>::price(mkt, opt, steps);

		// backward difference approximation (time to maturity decreases at time goes forward)
		// f'(x) = (f(T-dt)-f(T))/dt
		double annualized_theta = (future_price - cur_price) / dt;

		// Return a one-day scaled theta
		return annualized_theta / 365.0;
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