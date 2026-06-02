# src/estimators/covariance.py
"""
Covariance matrix estimators.

Implements four methods used in institutional practice:
1. Sample covariance (baseline, unstable for large N)
2. Ledoit-Wolf shrinkage (analytical optimal shrinkage)
3. EWMA — exponentially weighted (RiskMetrics approach)
4. Factor model via PCA (dimensionality reduction)
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────
# 1. Sample Covariance
# ─────────────────────────────────────────────────────────────────

def sample_covariance(returns: pd.DataFrame, annualize: bool = True, freq: int = 12) -> pd.DataFrame:
    """
    Standard sample covariance matrix.

    WARNING: Ill-conditioned when N is large relative to T.
    Use Ledoit-Wolf shrinkage for N > T/5.
    """
    cov = returns.cov()
    if annualize:
        cov = cov * freq
    return cov


# ─────────────────────────────────────────────────────────────────
# 2. Ledoit-Wolf Shrinkage
# ─────────────────────────────────────────────────────────────────

def ledoit_wolf_covariance(returns: pd.DataFrame, annualize: bool = True, freq: int = 12) -> pd.DataFrame:
    """
    Ledoit-Wolf analytical shrinkage estimator.

    Shrinks the sample covariance matrix toward a structured target
    (scaled identity) with an analytically optimal shrinkage intensity.

    Reference: Ledoit & Wolf (2004), JMVA.

    The sklearn implementation uses the 'Oracle Approximating Shrinkage' (OAS)
    formula which minimizes the MSE of the covariance estimate.
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    cov_array = lw.covariance_
    if annualize:
        cov_array = cov_array * freq
    return pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)


# ─────────────────────────────────────────────────────────────────
# 3. EWMA (Exponentially Weighted Moving Average)
# ─────────────────────────────────────────────────────────────────

def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = 0.97,
    annualize: bool = True,
    freq: int = 12,
) -> pd.DataFrame:
    """
    EWMA covariance matrix (RiskMetrics methodology).

    More weight is placed on recent observations via exponential decay:
        Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}·r_{t-1}'

    Parameters
    ----------
    lam : decay factor. RiskMetrics recommends 0.94 (daily), 0.97 (monthly).
          Higher λ → longer memory (smoother but slower to react to regime changes).
    """
    T, N = returns.shape
    vals = returns.values - returns.mean().values  # Demeaned

    # Initialize with sample covariance
    cov = np.cov(vals[:min(24, T)].T)

    for t in range(T):
        r = vals[t].reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)

    if annualize:
        cov = cov * freq
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


# ─────────────────────────────────────────────────────────────────
# 4. Factor Model (PCA-based)
# ─────────────────────────────────────────────────────────────────

def factor_covariance(
    returns: pd.DataFrame,
    n_factors: int = 3,
    annualize: bool = True,
    freq: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Factor model covariance via Principal Component Analysis.

    Decomposes returns into systematic (factor) and idiosyncratic components:
        r = B·F + ε
        Σ = B·Σ_F·B' + D

    where:
        B       = (N x k) factor loadings matrix
        Σ_F     = (k x k) factor covariance matrix
        D       = diagonal matrix of idiosyncratic variances

    This dramatically reduces the number of free parameters:
        Full matrix: N(N+1)/2 parameters
        Factor model: Nk + k(k+1)/2 + N parameters

    Returns
    -------
    cov_full     : Reconstructed N×N covariance matrix
    loadings     : Factor loading matrix B (N×k)
    explained_var: Variance explained by each factor
    """
    pca = PCA(n_components=n_factors)
    pca.fit(returns.values)

    # Factor loadings: B = components' (N x k)
    B = pca.components_.T  # Shape: (N, k)
    factors = returns.values @ B  # Factor scores: (T, k)

    # Factor covariance (k x k)
    Sigma_F = np.cov(factors.T)

    # Idiosyncratic residuals
    reconstructed = factors @ B.T
    residuals = returns.values - reconstructed
    D = np.diag(np.var(residuals, axis=0))

    # Full covariance: Σ = B·Σ_F·B' + D
    cov_full = B @ Sigma_F @ B.T + D

    if annualize:
        cov_full = cov_full * freq

    cov_df = pd.DataFrame(cov_full, index=returns.columns, columns=returns.columns)
    loadings_df = pd.DataFrame(
        B, index=returns.columns, columns=[f"PC{i+1}" for i in range(n_factors)]
    )
    explained_df = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(n_factors)],
        name="Explained Variance Ratio",
    )
    return cov_df, loadings_df, explained_df


# ─────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────

def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    **kwargs,
) -> pd.DataFrame:
    """
    Unified interface for covariance estimation.

    Parameters
    ----------
    method : 'sample' | 'ledoit_wolf' | 'ewma' | 'factor'
    """
    dispatch = {
        "sample":       sample_covariance,
        "ledoit_wolf":  ledoit_wolf_covariance,
        "ewma":         ewma_covariance,
        "factor":       lambda r, **kw: factor_covariance(r, **kw)[0],
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(dispatch.keys())}")
    return dispatch[method](returns, **kwargs)


def is_positive_definite(matrix: pd.DataFrame | np.ndarray) -> bool:
    """Check if a matrix is positive definite (all eigenvalues > 0)."""
    arr = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
    try:
        np.linalg.cholesky(arr)
        return True
    except np.linalg.LinAlgError:
        return False
