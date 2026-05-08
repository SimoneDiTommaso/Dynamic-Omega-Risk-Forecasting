import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model

# ---------------------------------------------------------
# CONFIGURATION & PARAMETERS
# ---------------------------------------------------------
TICKERS = {"S&P 500": "^GSPC", "FTSE MIB": "FTSEMIB.MI", "DAX 40": "^GDAXI"}
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
CONFIDENCE_LEVEL = 0.01  # alpha for 99% VaR/ES

def fetch_and_clean_data(tickers, start, end):
    """Retrieves adjusted close prices and returns cleaned log-returns."""
    data = yf.download(list(tickers.values()), start=start, end=end, auto_adjust=True)['Close']
    mapping = {v: k for k, v in tickers.items()}
    data = data.rename(columns=mapping)
    return np.log(data / data.shift(1)).dropna()

def estimate_risk_metrics(series, alpha):
    """Calculates static parametric and dynamic GARCH risk metrics."""
    # Basic Statistics
    mu = series.mean()
    sigma = series.std()
    jb_stat, p_value = stats.jarque_bera(series)
    z_alpha = stats.norm.ppf(alpha)
    
    # Static Parametric Estimation
    var_static = -(mu + sigma * z_alpha)
    es_static = -(mu - sigma * (stats.norm.pdf(z_alpha) / alpha))
    
    # Dynamic GARCH(1,1) Modeling
    # Scaling by 100 improves convergence stability for the optimizer
    model = arch_model(series * 100, vol='Garch', p=1, q=1, dist='normal')
    model_fit = model.fit(disp='off')
    
    # Scale conditional volatility back to original units
    cond_vol = model_fit.conditional_volatility / 100
    garch_mu = model_fit.params['mu'] / 100
    
    var_garch = -(garch_mu + z_alpha * cond_vol.mean())
    es_garch = -(garch_mu - cond_vol.mean() * (stats.norm.pdf(z_alpha) / alpha))
    
    # Taylor (2022) Implied Omega Analysis
    garch_ratio = es_garch / var_garch
    implied_omega = 1 + (1 / (alpha * (garch_ratio - 1)))
    
    return {
        "Volatility": sigma,
        "Skewness": series.skew(),
        "Kurtosis": series.kurtosis(),
        "JB_p_value": p_value,
        "Static_VaR": var_static,
        "Static_ES": es_static,
        "GARCH_VaR": var_garch,
        "GARCH_ES": es_garch,
        "ES_VaR_Ratio": garch_ratio,
        "Implied_Omega": implied_omega
    }

def visualize_results(series, name, metrics):
    """Generates distribution histograms and Q-Q plots for risk validation."""
    # Histogram with VaR/ES Markers
    plt.figure(figsize=(10, 6))
    plt.hist(series, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    x = np.linspace(series.min(), series.max(), 100)
    plt.plot(x, stats.norm.pdf(x, series.mean(), series.std()), 'r-', lw=2, label='Normal Dist')
    
    plt.axvline(-metrics["Static_VaR"], color='red', linestyle='--', label=f'VaR 99%: {metrics["Static_VaR"]:.2%}')
    plt.axvline(-metrics["Static_ES"], color='darkred', linestyle=':', label=f'ES 99%: {metrics["Static_ES"]:.2%}')
    
    plt.title(f"Log-Returns Distribution & Risk Metrics: {name}")
    plt.xlabel("Daily Log-Returns")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"distribution_{name.replace(' ', '_')}.png", dpi=300)
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(8, 6))
    stats.probplot(series, dist="norm", plot=plt)
    plt.title(f"Normal Q-Q Plot: {name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"qqplot_{name.replace(' ', '_')}.png", dpi=300)
    plt.show()

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    returns = fetch_and_clean_data(TICKERS, START_DATE, END_DATE)
    
    all_results = {}
    for name in TICKERS.keys():
        print(f"Processing {name}...")
        metrics = estimate_risk_metrics(returns[name], CONFIDENCE_LEVEL)
        all_results[name] = metrics
        visualize_results(returns[name], name, metrics)
    
    # Final Table Output
    res_df = pd.DataFrame(all_results).T
    print("\n--- Quantitative Risk Analysis Summary ---")
    print(res_df.round(4))
