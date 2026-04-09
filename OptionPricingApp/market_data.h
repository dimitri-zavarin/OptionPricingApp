#pragma once

struct MarketData {
	double S0;		// current stock price
	double r;		// risk free rate
	double sigma;	// volatility
	double q;		// dividend yield

	MarketData(double stockprice, double intrate, double vol, double div = 0.0) :
		S0(stockprice), r(intrate), sigma(vol), q(div) {
	}
};