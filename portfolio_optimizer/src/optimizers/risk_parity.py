# src/optimizers/risk_parity.py
"""
Risk Parity / Equal Risk Contribution (ERC) Portfolio.

Instead of equalizing weights or returns, ERC equalizes the
contribution of each asset to total portfolio risk.

Risk contribution of asset i:
    RC_i = w_i * (Σw)_i / sqrt(w'Σw)

ERC condition: RC_i = RC_j for all i, j  (= 1/N of total risk each)

Why institutions use it (Bridgewater All-Weather being the famous example):
- More diversified in risk space than naive 1/N
- Doesn't rely on noisy expected return estimates
- More stable weights over time vs mean-variance
- Naturally overweights low-volatility assets (bonds) vs high-volatility (equities)

Reference: Maillard, Roncalli & Teïletche (2010), Journal of Portfolio Management
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .markowitz import PortfolioResult


def _risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Compute marginal risk contributions for each asset."""
    port_vol = np.sqrt(w @ Sigma @ w)
    marginal = Sigma @ w
    return w * marginal / port_vol


def _erc_objective(w: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Minimize sum of squared differences between risk contributions.
    At the optimum, all RC_i are equal → ERC portfolio.
    """
    rc = _risk_contributions(w, Sigma)
    n = len(w)
    target = np.sum(rc) / n  # Each asset should contribute 1/n of total risk
    return float(np.sum((rc - target) ** 2))


def risk_parity(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    target_contributions: np.ndarray | None = None,
) -> PortfolioResult:
    """
    Equal (or custom) Risk Contribution portfolio.

    Parameters
    ----------
    target_contributions : if None, equal risk parity (1/N each).
                           Pass a custom array (summing to 1) for
                           budget risk parity (e.g., 60% risk equity, 40% bonds).

    Algorithm: Sequential Quadratic Programming (scipy L-BFGS-B)
    """
    n = len(mu)
    Sigma = cov.values

    if target_contributions is not None:
        budget = np.array(target_contributions)
        assert len(budget) == n, "target_contributions must match number of assets"

        def objective(w):
            rc = _risk_contributions(w, Sigma)
            port_risk = np.sqrt(w @ Sigma @ w)
            diff = rc / port_risk - budget
            return float(np.sum(diff ** 2))
    else:
        objective = lambda w: _erc_objective(w, Sigma)

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-6, 1.0)] * n  # Long-only with tiny lower bound for numerics

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 5000},
    )

    if not result.success:
        # Fallback: try different solver
        result = minimize(
            objective,
            w0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 5000},
        )

    w = result.x
    w = np.maximum(w, 0)
    w /= w.sum()  # Renormalize

    weights = pd.Series(w, index=mu.index)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol

    return PortfolioResult(weights, ret, vol, sharpe, "Risk Parity (ERC)")


def risk_contribution_report(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and display risk contribution of each asset in a portfolio.

    Returns a DataFrame with absolute and percentage risk contributions.
    """
    w = weights.values
    Sigma = cov.values
    rc = _risk_contributions(w, Sigma)
    port_vol = np.sqrt(w @ Sigma @ w)

    return pd.DataFrame({
        "Weight (%)":     (w * 100).round(2),
        "Risk Contrib (abs)": rc.round(6),
        "Risk Contrib (%)":   (rc / port_vol * 100).round(2),
    }, index=weights.index)
