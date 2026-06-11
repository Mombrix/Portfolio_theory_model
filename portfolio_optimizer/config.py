# config.py — Central configuration for the portfolio optimizer

# ─────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────
ASSET_UNIVERSE = {
    # Equities
    "SPY":  "S&P 500 (USA Equity)",
    "EZU":  "MSCI Eurozone (EU Equity)",
    "EEM":  "MSCI Emerging Markets (EM Equity)",
    # Bonds
    "TLT":  "20yr+ US Treasuries",
    "LQD":  "US Investment Grade Corp Bonds",
    "EMB":  "Emerging Markets Bonds",
    # Alternatives
    "GLD":  "Gold (SPDR)",
    "GSG":  "Commodities (S&P GSCI)",
}

TICKERS = list(ASSET_UNIVERSE.keys())

# ─────────────────────────────────────────────
# DATA PARAMETERS
# ─────────────────────────────────────────────
START_DATE         = "2014-01-01"   # 10 years of history
END_DATE           = None            # None = today
PRICE_FREQUENCY    = "1mo"          # '1mo' monthly | '1wk' weekly | '1d' daily
RETURN_TYPE        = "log"           # 'log' or 'simple'

# ─────────────────────────────────────────────
# RISK-FREE RATE
# ─────────────────────────────────────────────
# Annual risk-free rate (approx. 3-month T-Bill or ECB deposit rate)
RISK_FREE_RATE_ANNUAL = 0.043       # 4.3% as of 2024-2025 — update as needed
RISK_FREE_RATE_MONTHLY = RISK_FREE_RATE_ANNUAL / 12

# ─────────────────────────────────────────────
# INVESTOR PARAMETERS
# ─────────────────────────────────────────────
# Risk aversion coefficient A in U = E[r] - (A/2)*sigma^2
# Conservative: 6-10 | Moderate: 3-6 | Aggressive: 1-3
RISK_AVERSION = 4.0

# ─────────────────────────────────────────────
# COVARIANCE ESTIMATION
# ─────────────────────────────────────────────
COV_METHOD = "ledoit_wolf"  # 'sample' | 'ledoit_wolf' | 'ewma' | 'factor'
EWMA_LAMBDA = 0.97          # Decay factor for EWMA (RiskMetrics monthly)
N_FACTORS   = 3             # Number of PCA factors for factor covariance

# ─────────────────────────────────────────────
# EXPECTED RETURNS ESTIMATION
# ─────────────────────────────────────────────
MU_METHOD = "historical"    # 'historical' | 'capm' | 'black_litterman'

# ─────────────────────────────────────────────
# OPTIMIZATION CONSTRAINTS
# ─────────────────────────────────────────────
MIN_WEIGHT  = 0.00   # No short selling
MAX_WEIGHT  = 0.40   # Max 40% on any single asset (concentration limit)
MIN_WEIGHT_IF_NONZERO = 0.05  # If asset is included, min 5% position

# CVaR parameters
CVAR_ALPHA  = 0.05   # Confidence level: losses in worst 5% of scenarios

# Budget
BUDGET_EUR = 10_000 # Total capital to invest in EUR

# Class exposure limits
MAX_EQUITY = 0.60  # Max 60% in equities
MAX_BONDS  = 0.60  # Max 60% in bonds
MAX_ALTS   = 0.40  # Max 40% in alternatives

# Risk constraints
MAX_VOL = 0.15       # Max portfolio volatility (15% annualized)
TURNOVER_MAX = None # Max turnover (e.g., 0.5 for 50%) — set to None for no limit

EQUITY_TICKERS = ["SPY", "EZU", "EEM"]
BOND_TICKERS   = ["TLT", "LQD", "EMB"]
ALT_TICKERS    = ["GLD", "GSG"]

# ─────────────────────────────────────────────
# BLACK-LITTERMAN PARAMETERS
# ─────────────────────────────────────────────
BL_TAU = 0.05        # Scaling factor for prior uncertainty
BL_DELTA = 2.5       # Implied risk aversion for market equilibrium weights

# ─────────────────────────────────────────────
# EFFICIENT FRONTIER
# ─────────────────────────────────────────────
N_FRONTIER_POINTS = 100   # Number of portfolios to plot on frontier

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
RESULTS_DIR = "results/"
DATA_DIR    = "data/"
