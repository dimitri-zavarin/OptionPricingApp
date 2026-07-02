#pragma once

#include "market_data.h"
#include "binomial.h"
#include <cstdint>

// Strongly-typed flags for Greeks
enum class GreekFlag : uint32_t {
    None    = 0u,
    Delta   = 1u << 0,
    Gamma   = 1u << 1,
    Vega    = 1u << 2,
    Theta   = 1u << 3,
    Epsilon = 1u << 4,
    All     = Delta | Gamma | Vega | Theta | Epsilon
};

inline constexpr GreekFlag operator|(GreekFlag a, GreekFlag b) {
    return static_cast<GreekFlag>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr bool has_flag(GreekFlag flags, GreekFlag flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

struct GreekValues {
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;
    double theta = 0.0;
    double epsilon = 0.0;
};

template <typename OptionType, template <typename> typename PricerType>
struct Delta {
    static double calculate(MarketData mkt, const OptionType& opt, int steps, double shift = 0.01) {
        double d_s = mkt.s0_ * shift;

        mkt.s0_ += d_s;
        double up_price = PricerType<OptionType>::price(mkt, opt, steps);

        mkt.s0_ -= 2.0 * d_s;
        double down_price = PricerType<OptionType>::price(mkt, opt, steps);

        return (up_price - down_price) / (2.0 * d_s);
    }
};

template <typename OptionType, template <typename> typename PricerType>
struct Gamma {
    static double calculate(MarketData mkt, const OptionType& opt, int steps, double cur_price = 0.0, double shift = 0.01) {
        double d_s = mkt.s0_ * shift;
        if (cur_price == 0.0) {
            cur_price = PricerType<OptionType>::price(mkt, opt, steps);
        }

        mkt.s0_ += d_s;
        double up_price = PricerType<OptionType>::price(mkt, opt, steps);

        mkt.s0_ -= 2.0 * d_s;
        double down_price = PricerType<OptionType>::price(mkt, opt, steps);

        return (up_price - 2.0 * cur_price + down_price) / (d_s * d_s);
    }
};

template <typename OptionType, template <typename> typename PricerType>
struct Vega {
    static double calculate(MarketData mkt, const OptionType& opt, int steps, double shift = 0.01) {
        double d_sigma = mkt.volatility_ * shift;
        if (d_sigma < 1e-10) { d_sigma = 1e-4; }

        mkt.volatility_ += d_sigma;
        double up_price = PricerType<OptionType>::price(mkt, opt, steps);

        mkt.volatility_ -= 2.0 * d_sigma;
        double down_price = PricerType<OptionType>::price(mkt, opt, steps);

        return (up_price - down_price) / (2.0 * d_sigma);
    }
};

template <typename OptionType, template <typename> typename PricerType>
struct Theta {
    static double calculate(MarketData mkt, const OptionType& opt, int steps, double cur_price = 0.0, double dt = 0.0001) {
        if (cur_price == 0.0) {
            cur_price = PricerType<OptionType>::price(mkt, opt, steps);
        }

        double maturity = opt.maturity();
        if (maturity <= dt) { dt = maturity; }
        if (dt < 1e-10) { return 0.0; }

        // create a reduced-maturity option using the public helper
        OptionType opt_future = opt.with_maturity(maturity - dt);
        double future_price = PricerType<OptionType>::price(mkt, opt_future, steps);

        // backward difference approximation (time to maturity decreases as time advances)
        double annualized_theta = (future_price - cur_price) / dt;

        // Return a one-day scaled theta
        return annualized_theta / 365.0;
    }
};

template <typename OptionType, template <typename> typename PricerType>
struct Epsilon {
    static double calculate(MarketData mkt, const OptionType& opt, int steps, double dq = 0.0001) {
        mkt.dividend_yield_ += dq;
        double up_price = PricerType<OptionType>::price(mkt, opt, steps);

        mkt.dividend_yield_ -= 2.0 * dq;
        double down_price = PricerType<OptionType>::price(mkt, opt, steps);

        return (up_price - down_price) / (2.0 * dq);
    }
};

template <typename OptionType, template <typename> typename PricerType>
struct GreeksSuite {
    static GreekValues calculate(MarketData mkt, const OptionType& opt, int steps, GreekFlag requested_greeks = GreekFlag::All) {
        GreekValues values;
        double cur_price = 0.0;

        if (has_flag(requested_greeks, GreekFlag::Gamma) || has_flag(requested_greeks, GreekFlag::Theta)) {
            cur_price = PricerType<OptionType>::price(mkt, opt, steps);
        }

        if (has_flag(requested_greeks, GreekFlag::Delta)) {
            values.delta = Delta<OptionType, PricerType>::calculate(mkt, opt, steps);
        }

        if (has_flag(requested_greeks, GreekFlag::Gamma)) {
            values.gamma = Gamma<OptionType, PricerType>::calculate(mkt, opt, steps, cur_price);
        }

        if (has_flag(requested_greeks, GreekFlag::Vega)) {
            values.vega = Vega<OptionType, PricerType>::calculate(mkt, opt, steps);
        }

        if (has_flag(requested_greeks, GreekFlag::Theta)) {
            values.theta = Theta<OptionType, PricerType>::calculate(mkt, opt, steps, cur_price);
        }

        if (has_flag(requested_greeks, GreekFlag::Epsilon)) {
            values.epsilon = Epsilon<OptionType, PricerType>::calculate(mkt, opt, steps);
        }

        return values;
    }
};};