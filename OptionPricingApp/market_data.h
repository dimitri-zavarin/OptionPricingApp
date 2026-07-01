#pragma once

struct MarketData {
    MarketData(double s0, double risk_free_rate, double volatility, double dividend_yield = 0.0)
        : s0_(s0), risk_free_rate_(risk_free_rate), volatility_(volatility), dividend_yield_(dividend_yield) {}

    double s0_;
    double risk_free_rate_;
    double volatility_;
    double dividend_yield_;
};