#pragma once

#include <algorithm>

struct CallPayoff {
	static double calculate(double S, double K) {
		return std::max(S - K, 0.0);
	}
};

struct PutPayoff {
	static double calculate(double S, double K) {
		return std::max(K - S, 0.0);
	}
};

struct European {};
struct American {};

template <typename PayoffType, typename ExerciseType>
class Option {
public:
	double K; // strike price
	double T; // time to maturity

	using Exercise = ExerciseType;

	Option(double strike, double maturity) : K(strike), T(maturity) {}

	double payoff(double S) const {
		return PayoffType::calculate(S, K);
	}
};

using EuCall = Option<CallPayoff, European>;
using EuPut = Option<PutPayoff, European>;
using AmCall = Option<CallPayoff, American>;
using AmPut = Option<PutPayoff, American>;