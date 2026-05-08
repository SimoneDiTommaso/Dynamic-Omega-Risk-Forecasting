# Dynamic Omega Ratios in Risk Management

### Executive Summary
This research develops a novel approach to forecasting joint **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** by modeling the **Omega Ratio** as a dynamic, time-varying parameter. Unlike traditional models that assume a constant relationship between tail risk metrics, this framework utilizes **GARCH(1,1)** volatility modeling to capture the leptokurtosis (fat tails) inherent in modern financial markets.

### Technical Framework
* **Backtesting & Validation:** Implemented joint scoring functions to evaluate the elicitability of VaR and ES, moving beyond traditional independent backtests.
* **Volatility Modeling:** Utilized GARCH(1,1) with Gaussian and Student-t distributions to isolate the impact of dynamic volatility on tail risk estimation.
* **Empirical Scope:** Analyzed 1,250+ daily observations for the **DAX 40**, **S&P 500**, and **FTSE MIB** (2021–2025).
* **Key Finding:** Proved that dynamic Omega ratios significantly outperform static benchmarks in predicting extreme market events, particularly during high-volatility regimes.

### Mathematical Core
The model identifies ES as a function of the dynamic Omega ratio ($\Omega_{t}$):
$$ES_{t}(\alpha)=\left(1+\frac{1}{\alpha(\Omega_{t}(VaR_{t}(\alpha))-1)}\right)VaR_{t}(\alpha)$$

### Tech Stack
* **Language:** Python
* **Libraries:** `arch` (GARCH modeling), `pandas` (Time-series data), `scipy.stats` (Statistical validation), `yfinance` (API integration).
