# src/optimizers/cvar_optimizer.py
"""
CVaR (Conditional Value-at-Risk) Portfolio Optimization.

CVaR (also called Expected Shortfall, ES) measures the expected loss
in the worst α fraction of scenarios. Unlike VaR, it:
- Is coherent (satisfies subadditivity)
- Is convex in portfolio weights → tractable linear programming
- Better captures tail risk / fat tails

CVaR minimization via Rockafellar-Uryasev (2000) LP formulation:
    min  VaR_α + 1/(T·α) · Σ_t [losses_t - VaR_α]₊

This is equivalent to:
    min  z + 1/(T·α) · Σ_t u_t
    s.t. u_t >= -r_portfolio_t - z
         u_t >= 0
         w'1 = 1,  w >= 0

Reference: Rockafellar & Uryasev (2000), Journal of Risk
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from .markowitz import PortfolioResult


def cvar_optimization(
    returns: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    alpha: float = 0.05,
    risk_free_rate: float = 0.043,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    min_return: float | None = None,
) -> PortfolioResult:
    """
    Minimize CVaR (Expected Shortfall) at confidence level (1-α).

    By default minimizes CVaR₀.₀₅ — the expected loss in the worst 5% of months.

    Parameters
    ----------
    alpha     : tail probability (0.05 = 5% worst scenarios)
    min_return: optional annualized minimum expected return constraint
    """
    T, N = returns.shape
    R = returns.values  # (T x N) matrix of historical returns

    w = cp.Variable(N)
    z = cp.Variable()          # VaR level (the CVaR auxiliary variable)
    u = cp.Variable(T)         # Auxiliary variables for losses exceeding VaR

    # Portfolio losses each period (negative = loss)
    losses = -R @ w            # Shape: (T,)

    # CVaR objective: z + 1/(T*alpha) * sum(u)
    cvar_obj = z + (1 / (T * alpha)) * cp.sum(u)

    constraints = [
        u >= losses - z,
        u >= 0,
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    if min_return is not None:
        # Convert annualized constraint to per-period
        freq = 12  # monthly
        constraints.append(mu.values @ w >= min_return / freq)

    prob = cp.Problem(cp.Minimize(cvar_obj), constraints)
    prob.solve(solver=cp.ECOS, warm_starting=True)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVaR optimization failed: {prob.status}")

    weights = pd.Series(w.value, index=returns.columns)
    weights = weights.clip(lower=0)
    weights /= weights.sum()

    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol

    return PortfolioResult(weights, ret, vol, sharpe, f"Min CVaR (α={alpha})")


def portfolio_cvar(
    weights: pd.Series,
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """
    Compute VaR and CVaR for a given portfolio and historical return series.

    Returns dict with 'VaR', 'CVaR', 'worst_period_return', 'cvar_periods'.
    Values are expressed as losses (positive = loss).
    """
    port_returns = returns @ weights
    losses = -port_returns  # Convert to loss

    # VaR: quantile of losses
    var = float(np.quantile(losses, 1 - alpha))

    # CVaR: mean of losses beyond VaR
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean())

    return {
        "VaR (monthly)":   var,
        "CVaR (monthly)":  cvar,
        "VaR (annual)":    var * np.sqrt(12),
        "CVaR (annual)":   cvar * np.sqrt(12),
        "Worst return":    float(port_returns.min()),
        "Tail obs.":       len(tail_losses),
    }
