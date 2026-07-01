#pragma once

#include <algorithm>

struct CallPayoff {
    static double calculate(double s, double k) {
        return std::max(s - k, 0.0);
    }
};

struct PutPayoff {
    static double calculate(double s, double k) {
        return std::max(k - s, 0.0);
    }
};

struct European {};
struct American {};

template <typename PayoffType, typename ExerciseType>
class Option {
public:
    Option(double strike, double maturity) : strike_(strike), maturity_(maturity) {}

    double payoff(double s) const { return PayoffType::calculate(s, strike_); }

    double strike() const { return strike_; }
    double maturity() const { return maturity_; }

    Option with_maturity(double new_maturity) const {
        return Option(strike_, new_maturity);
    }

    using Payoff = PayoffType;
    using Exercise = ExerciseType;

private:
    double strike_;
    double maturity_;
};

using EuropeanCall = Option<CallPayoff, European>;
using EuropeanPut  = Option<PutPayoff, European>;
using AmericanCall  = Option<CallPayoff, American>;
using AmericanPut   = Option<PutPayoff, American>;