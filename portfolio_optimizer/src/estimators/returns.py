# src/estimators/returns.py
"""
Expected return estimators.

The choice of expected returns has the largest impact on portfolio weights
(much more than covariance estimation). Three approaches are implemented:

1. Historical mean (simple, but noisy)
2. CAPM-implied (uses market equilibrium, more stable)
3. Black-Litterman prior (market equilibrium + optional investor views)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 1. Historical Mean
# ─────────────────────────────────────────────────────────────────

def historical_mean(
    returns: pd.DataFrame,
    annualize: bool = True,
    freq: int = 12,
) -> pd.Series:
    """
    Sample mean of historical returns, optionally annualized.

    Warning: highly sensitive to the sample period chosen.
    A single extreme year can dominate the estimate.
    """
    mu = returns.mean()
    if annualize:
        mu = mu * freq
    return mu


# ─────────────────────────────────────────────────────────────────
# 2. CAPM Equilibrium Returns
# ─────────────────────────────────────────────────────────────────

def capm_returns(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.043,
) -> pd.Series:
    """
    CAPM-implied equilibrium returns (Black-Litterman prior).

    Inverts the CAPM relationship to infer expected returns consistent
    with the market portfolio being mean-variance optimal:

        Π = δ · Σ · w_mkt + rf

    where:
        δ  = risk aversion coefficient (typically 2-4 for institutional)
        Σ  = covariance matrix (annualized)
        w_mkt = market-cap weights

    Parameters
    ----------
    market_weights : pd.Series of market-cap weights (sum to 1)
    risk_aversion  : δ, implied risk aversion (use BL_DELTA from config)
    risk_free_rate : annual, added to the excess return
    """
    w = market_weights.reindex(returns.columns).fillna(0)
    w = w / w.sum()

    # Implied excess returns
    pi = risk_aversion * cov_matrix @ w
    return pi + risk_free_rate


# ─────────────────────────────────────────────────────────────────
# 3. Black-Litterman Expected Returns
# ─────────────────────────────────────────────────────────────────

def black_litterman_returns(
    prior_mu: pd.Series,
    cov_matrix: pd.DataFrame,
    views_P: np.ndarray | None = None,
    views_q: np.ndarray | None = None,
    views_omega: np.ndarray | None = None,
    tau: float = 0.05,
) -> pd.Series:
    """
    Black-Litterman posterior expected returns.

    Combines the CAPM prior Π with investor views (P, q) using
    Bayesian updating. Without views, returns the prior.

    Model:
        Prior:    μ ~ N(Π, τ·Σ)
        Views:    P·μ = q + ε,  ε ~ N(0, Ω)
        Posterior: μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}Π + P'Ω^{-1}q]

    Parameters
    ----------
    prior_mu   : Equilibrium expected returns (from capm_returns)
    views_P    : (K x N) pick matrix. Each row is a view.
                 Absolute: P[k,:] has 1 on the asset.
                 Relative: P[k,:] has +1 (outperform) and -1 (underperform).
    views_q    : (K,) vector of view expected returns (annualized)
    views_omega: (K x K) diagonal uncertainty matrix for views.
                 If None, uses proportional uncertainty: Ω = diag(P·τΣ·P')
    tau        : Scales prior uncertainty (typical range 0.01–0.10)

    Example — single absolute view
    --------------------------------
    "I believe SPY will return 12% annually"
    P = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])  # SPY is first column
    q = np.array([0.12])

    Example — relative view
    -------------------------
    "I believe GLD will outperform GSG by 4% annually"
    P = np.array([[0, 0, 0, 0, 0, 0, 1, -1]])  # GLD col +1, GSG col -1
    q = np.array([0.04])
    """
    if views_P is None or views_q is None:
        return prior_mu

    Sigma = cov_matrix.values
    pi = prior_mu.values
    P = np.atleast_2d(views_P)
    q = np.atleast_1d(views_q)
    K = P.shape[0]

    # Prior precision
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)

    # View uncertainty Ω (proportional if not specified)
    if views_omega is None:
        omega_diag = np.diag(P @ (tau * Sigma) @ P.T)
        Omega_inv = np.diag(1.0 / omega_diag)
    else:
        Omega_inv = np.linalg.inv(views_omega)

    # Posterior precision and mean
    posterior_prec = tau_Sigma_inv + P.T @ Omega_inv @ P
    posterior_cov  = np.linalg.inv(posterior_prec)
    posterior_mu   = posterior_cov @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ q)

    return pd.Series(posterior_mu, index=prior_mu.index)


# ─────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────

def estimate_returns(
    returns: pd.DataFrame,
    method: str = "historical",
    **kwargs,
) -> pd.Series:
    """
    Unified dispatcher for expected return estimation.

    Parameters
    ----------
    method : 'historical' | 'capm' | 'black_litterman'
    kwargs : passed to the underlying estimator
    """
    if method == "historical":
        return historical_mean(returns, **kwargs)
    elif method == "capm":
        return capm_returns(returns, **kwargs)
    elif method == "black_litterman":
        return black_litterman_returns(**kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'")
