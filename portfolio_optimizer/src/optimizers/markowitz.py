# src/optimizers/markowitz.py
"""
Markowitz Mean-Variance Optimization.

Implements three classic portfolios via Quadratic Programming (CVXPY):
1. Minimum Variance Portfolio
2. Maximum Sharpe Ratio (Tangency Portfolio)
3. Mean-Variance Efficient Frontier
4. Maximum Utility Portfolio (with investor risk aversion A)

All optimizations are long-only by default (w_i >= 0).
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    weights: pd.Series
    expected_return: float    # Annualized
    volatility: float         # Annualized
    sharpe_ratio: float
    method: str

    def __repr__(self):
        lines = [
            f"\n{'─'*45}",
            f"  Strategy : {self.method}",
            f"  E[r]     : {self.expected_return*100:.2f}%",
            f"  σ        : {self.volatility*100:.2f}%",
            f"  Sharpe   : {self.sharpe_ratio:.3f}",
            f"{'─'*45}",
        ]
        return "\n".join(lines)


def _solve_qp(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    objective: str = "min_variance",
    target_return: float | None = None,
    risk_aversion: float | None = None,
    risk_free_rate: float = 0.0,
    w_prev: np.ndarray | None = None,
    turnover_max: float | None = None,
) -> np.ndarray:
    """
    Core QP solver using CVXPY.

    CVXPY formulates the problem symbolically and dispatches to
    an appropriate solver (OSQP by default for QP problems).
    """
    n = len(mu)
    w = cp.Variable(n)

    # Portfolio return and variance expressions
    port_return   = mu @ w
    port_variance = cp.quad_form(w, Sigma)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    if target_return is not None:
        constraints.append(port_return >= target_return)

    if turnover_max is not None and w_prev is not None:
        constraints.append(cp.norm(w - w_prev, 1) <= turnover_max)

    # Objective
    if objective == "min_variance":
        obj = cp.Minimize(port_variance)

    elif objective == "max_sharpe":
        # Trick: substitute y = w / (mu - rf)'w, then maximize Sharpe
        # equivalent to minimizing variance for unit excess return
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        excess_return = (mu - risk_free_rate) @ y
        constraints_sharpe = [
            excess_return == 1,
            cp.sum(y) == kappa,
            y >= min_weight * kappa,
            y <= max_weight * kappa,
            kappa >= 0,
        ]
        obj = cp.Minimize(cp.quad_form(y, Sigma))
        prob = cp.Problem(obj, constraints_sharpe)
        prob.solve(solver=cp.OSQP, warm_starting=True, max_iter=10000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Max Sharpe QP failed: {prob.status}")
        return y.value / kappa.value

    elif objective == "max_utility":
        # U = E[r] - (A/2)*Var
        obj = cp.Maximize(port_return - (risk_aversion / 2) * port_variance)

    else:
        raise ValueError(f"Unknown objective '{objective}'")

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, warm_starting=True, max_iter=10000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"QP optimization failed: {prob.status}")

    return w.value


def minimum_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Minimum Variance Portfolio — ignores expected returns entirely.
    Preferred when expected returns are noisy/unreliable.
    """
    w = _solve_qp(
        mu.values, cov.values, mu.index.tolist(),
        min_weight=min_weight, max_weight=max_weight,
        objective="min_variance",
    )
    weights = pd.Series(w, index=mu.index)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol
    return PortfolioResult(weights, ret, vol, sharpe, "Minimum Variance")


def maximum_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Tangency Portfolio — maximizes the Sharpe Ratio.
    This is the risky portfolio used for capital allocation.
    """
    w = _solve_qp(
        mu.values, cov.values, mu.index.tolist(),
        min_weight=min_weight, max_weight=max_weight,
        objective="max_sharpe",
        risk_free_rate=risk_free_rate,
    )
    weights = pd.Series(w, index=mu.index)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol
    return PortfolioResult(weights, ret, vol, sharpe, "Maximum Sharpe (Tangency)")


def maximum_utility(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 4.0,
    risk_free_rate: float = 0.043,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Maximum Utility Portfolio for a given risk aversion A.

    Maximizes: U = E[r] - (A/2) * σ²

    This directly incorporates the investor's preference — the result
    is a specific point on the efficient frontier determined by A.
    """
    w = _solve_qp(
        mu.values, cov.values, mu.index.tolist(),
        min_weight=min_weight, max_weight=max_weight,
        objective="max_utility",
        risk_aversion=risk_aversion,
    )
    weights = pd.Series(w, index=mu.index)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol
    return PortfolioResult(weights, ret, vol, sharpe, f"Max Utility (A={risk_aversion})")


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    n_points: int = 100,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by sweeping over target returns.

    Returns a DataFrame with columns: [return, volatility, sharpe, w1, w2, ...]
    for n_points portfolios along the frontier.
    """
    # Bounds: from min-variance return to max-return asset
    mv = minimum_variance(mu, cov, risk_free_rate, min_weight, max_weight)
    r_min = mv.expected_return
    r_max = float(mu.max()) * 0.99

    target_returns = np.linspace(r_min, r_max, n_points)
    rows = []

    for r_target in target_returns:
        try:
            w = _solve_qp(
                mu.values, cov.values, mu.index.tolist(),
                min_weight=min_weight, max_weight=max_weight,
                objective="min_variance",
                target_return=r_target,
            )
            weights = pd.Series(w, index=mu.index)
            vol = float(np.sqrt(weights @ cov @ weights))
            sharpe = (r_target - risk_free_rate) / vol
            row = {"return": r_target, "volatility": vol, "sharpe": sharpe}
            row.update(weights.to_dict())
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)
