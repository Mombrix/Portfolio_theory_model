# src/risk/metrics.py
"""
Risk and performance metrics.

Comprehensive set of measures used by portfolio managers:
- Risk-adjusted return ratios (Sharpe, Sortino, Calmar, Information Ratio)
- Drawdown analysis
- VaR / CVaR
- Capital Allocation Line computation

All metrics assume monthly returns as input and annualize by sqrt(12) or *12.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Return / Risk Ratios
# ─────────────────────────────────────────────────────────────────

def sharpe_ratio(
    port_returns: pd.Series,
    risk_free_rate_annual: float = 0.043,
    freq: int = 12,
) -> float:
    """Annualized Sharpe = (E[r] - rf) / σ"""
    rf_monthly = risk_free_rate_annual / freq
    excess = port_returns - rf_monthly
    return float((excess.mean() * freq) / (excess.std() * np.sqrt(freq)))


def sortino_ratio(
    port_returns: pd.Series,
    risk_free_rate_annual: float = 0.043,
    freq: int = 12,
) -> float:
    """
    Sortino Ratio — like Sharpe but penalizes only downside deviation.
    More appropriate when return distributions are asymmetric.
    """
    rf_monthly = risk_free_rate_annual / freq
    excess = port_returns - rf_monthly
    downside = excess[excess < 0].std() * np.sqrt(freq)
    if downside == 0:
        return np.inf
    return float(excess.mean() * freq / downside)


def calmar_ratio(
    port_returns: pd.Series,
    freq: int = 12,
) -> float:
    """
    Calmar Ratio = Annualized Return / Max Drawdown.
    Used by hedge funds to evaluate return per unit of drawdown risk.
    """
    ann_ret = port_returns.mean() * freq
    mdd = max_drawdown(port_returns)
    if mdd == 0:
        return np.inf
    return float(ann_ret / abs(mdd))


def information_ratio(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    freq: int = 12,
) -> float:
    """
    Information Ratio = Active Return / Tracking Error.
    Measures skill of active management relative to a benchmark.
    """
    active = port_returns - benchmark_returns
    te = active.std() * np.sqrt(freq)
    if te == 0:
        return np.inf
    return float(active.mean() * freq / te)


# ─────────────────────────────────────────────────────────────────
# Drawdown
# ─────────────────────────────────────────────────────────────────

def drawdown_series(port_returns: pd.Series) -> pd.Series:
    """
    Compute time series of drawdown from peak.
    Drawdown_t = (Wealth_t - Peak_t) / Peak_t
    """
    wealth = (1 + port_returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak


def max_drawdown(port_returns: pd.Series) -> float:
    """Maximum peak-to-trough loss (negative number)."""
    return float(drawdown_series(port_returns).min())


def drawdown_duration(port_returns: pd.Series) -> int:
    """Longest drawdown duration in periods."""
    dd = drawdown_series(port_returns)
    in_dd = (dd < 0).astype(int)
    # Count consecutive periods in drawdown
    durations = []
    count = 0
    for v in in_dd:
        if v:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    return max(durations) if durations else 0


# ─────────────────────────────────────────────────────────────────
# VaR / CVaR (Historical Simulation)
# ─────────────────────────────────────────────────────────────────

def historical_var(port_returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical VaR at confidence level (1-alpha).
    Returns the loss threshold (positive = loss).
    """
    return float(-np.quantile(port_returns, alpha))


def historical_cvar(port_returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical CVaR (Expected Shortfall) at confidence level (1-alpha).
    Returns expected loss in the worst alpha fraction of periods.
    """
    var = np.quantile(port_returns, alpha)
    tail = port_returns[port_returns <= var]
    return float(-tail.mean())


# ─────────────────────────────────────────────────────────────────
# Capital Allocation (BKM Chapter 6)
# ─────────────────────────────────────────────────────────────────

def optimal_risky_weight(
    risky_return: float,
    risky_vol: float,
    risk_free_rate: float,
    risk_aversion: float,
) -> float:
    """
    Optimal fraction y* to invest in the risky portfolio.

    From BKM utility maximization:
        y* = (E[rp] - rf) / (A · σp²)

    If y* > 1: invest more than 100% in risky (leverage) — we cap at 1.0
    If y* < 0: short the risky portfolio — we floor at 0.0 (no shorting).

    Parameters
    ----------
    risky_return  : Annualized expected return of risky portfolio
    risky_vol     : Annualized volatility of risky portfolio
    risk_free_rate: Annual risk-free rate
    risk_aversion : Investor's A coefficient
    """
    excess = risky_return - risk_free_rate
    y_star = excess / (risk_aversion * risky_vol ** 2)
    return float(np.clip(y_star, 0.0, 1.0))


def capital_allocation_line(
    risky_return: float,
    risky_vol: float,
    risk_free_rate: float,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute the Capital Allocation Line (CAL).

    The CAL connects the risk-free asset to the tangency portfolio.
    Points on the CAL:
        E[rc] = rf + y * (E[rp] - rf)
        σ_c   = y * σ_p

    where y ∈ [0, 1] is the fraction invested in the risky portfolio.

    Returns a DataFrame with columns: [y, return, volatility, sharpe]
    """
    y_range = np.linspace(0, 1, n_points)
    cal = pd.DataFrame({
        "y (risky weight)": y_range,
        "return": risk_free_rate + y_range * (risky_return - risk_free_rate),
        "volatility": y_range * risky_vol,
    })
    cal["sharpe"] = (cal["return"] - risk_free_rate) / cal["volatility"].replace(0, np.nan)
    return cal


# ─────────────────────────────────────────────────────────────────
# Comprehensive Portfolio Report
# ─────────────────────────────────────────────────────────────────

def portfolio_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    risk_aversion: float = 4.0,
    benchmark_returns: pd.Series | None = None,
    freq: int = 12,
) -> dict:
    """
    Compute a comprehensive set of risk/return metrics for a given portfolio.
    """
    port_returns = returns @ weights

    ann_return = float(weights @ mu)
    ann_vol    = float(np.sqrt(weights @ cov @ weights))

    report = {
        # Return metrics
        "Annualized Return (%)":   round(ann_return * 100, 2),
        "Annualized Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio":            round(sharpe_ratio(port_returns, risk_free_rate, freq), 3),
        "Sortino Ratio":           round(sortino_ratio(port_returns, risk_free_rate, freq), 3),
        "Calmar Ratio":            round(calmar_ratio(port_returns, freq), 3),

        # Risk metrics
        "VaR 95% (monthly, %)":   round(historical_var(port_returns, 0.05) * 100, 2),
        "CVaR 95% (monthly, %)":  round(historical_cvar(port_returns, 0.05) * 100, 2),
        "Max Drawdown (%)":        round(max_drawdown(port_returns) * 100, 2),
        "Max DD Duration (periods)": drawdown_duration(port_returns),

        # Capital allocation
        "Optimal Risky Weight (y*)": round(
            optimal_risky_weight(ann_return, ann_vol, risk_free_rate, risk_aversion), 3
        ),
        "Combined E[r] (%)": round(
            (optimal_risky_weight(ann_return, ann_vol, risk_free_rate, risk_aversion) * ann_return
             + (1 - optimal_risky_weight(ann_return, ann_vol, risk_free_rate, risk_aversion)) * risk_free_rate) * 100,
            2
        ),
    }

    if benchmark_returns is not None:
        report["Information Ratio"] = round(
            information_ratio(port_returns, benchmark_returns, freq), 3
        )

    return report
