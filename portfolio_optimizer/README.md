# Portfolio Optimizer 📈

A professional-grade Python framework for **Capital Allocation** and **Asset Allocation** based on Modern Portfolio Theory (Bodie, Kane & Marcus) with institutional-level enhancements.

## Features

- **Data Fetching**: Automated download via `yfinance` for equities, ETFs, bonds, and commodities
- **Return Estimation**: Historical, EWMA (RiskMetrics), and shrinkage-adjusted
- **Covariance Estimation**: Sample, Ledoit-Wolf shrinkage, Factor-based (PCA)
- **Optimization Strategies**:
  - Markowitz Mean-Variance (long-only, QP)
  - Minimum Variance Portfolio
  - Maximum Sharpe Ratio (Tangency Portfolio)
  - Risk Parity (Equal Risk Contribution)
  - Black-Litterman with investor views
  - CVaR (Conditional Value-at-Risk) optimization
- **Capital Allocation**: Optimal split between risky portfolio and risk-free asset given investor risk aversion `A`
- **Risk Analytics**: VaR, CVaR, max drawdown, Sharpe, Sortino, Calmar ratios
- **Visualization**: Efficient frontier, risk decomposition, weight evolution

## Structure

```
portfolio_optimizer/
│
├── src/
│   ├── data_fetcher/
│   │   └── fetcher.py          # Download & clean price data
│   ├── estimators/
│   │   ├── returns.py          # Expected return estimators
│   │   └── covariance.py       # Covariance matrix estimators
│   ├── optimizers/
│   │   ├── markowitz.py        # MVO, Min-Var, Max-Sharpe
│   │   ├── risk_parity.py      # Equal Risk Contribution
│   │   ├── black_litterman.py  # BL model
│   │   └── cvar_optimizer.py   # CVaR minimization
│   ├── risk/
│   │   └── metrics.py          # VaR, CVaR, drawdown, ratios
│   └── utils/
│       └── plotting.py         # Visualizations
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_covariance_estimation.ipynb
│   ├── 03_optimization_strategies.ipynb
│   ├── 04_capital_allocation.ipynb
│   └── 05_full_pipeline.ipynb
│
├── data/                        # Cached price data (gitignored)
├── results/                     # Output charts and weights
├── tests/
├── requirements.txt
└── config.py                    # Universe, parameters, risk-free rate
```

## Quickstart

```bash
git clone https://github.com/<your-username>/portfolio_optimizer.git
cd portfolio_optimizer
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/05_full_pipeline.ipynb` for the complete end-to-end example.

## Theoretical Background

This project implements the framework from:

- **Bodie, Kane & Marcus** — *Investments* (Capital Allocation, MPT, CAPM)
- **Ledoit & Wolf (2004)** — *A well-conditioned estimator for large-dimensional covariance matrices*
- **Black & Litterman (1992)** — *Global Portfolio Optimization*
- **Maillard, Roncalli & Teïletche (2010)** — *The Properties of Equally Weighted Risk Contributions Portfolios*
- **Rockafellar & Uryasev (2000)** — *Optimization of Conditional Value-at-Risk*

## Asset Universe (default)

| Category        | Ticker | Description                      |
|-----------------|--------|----------------------------------|
| Equity USA      | SPY    | S&P 500 ETF                      |
| Equity Europe   | EZU    | MSCI Eurozone ETF                |
| Equity EM       | EEM    | MSCI Emerging Markets ETF        |
| Bonds USA       | TLT    | 20+ Year Treasury Bond ETF       |
| Bonds Corp      | LQD    | Investment Grade Corp Bond ETF   |
| Bonds EM        | EMB    | Emerging Markets Bond ETF        |
| Commodities     | GSG    | iShares S&P GSCI Commodity ETF   |
| Gold            | GLD    | SPDR Gold Trust                  |

Customize in `config.py`.
