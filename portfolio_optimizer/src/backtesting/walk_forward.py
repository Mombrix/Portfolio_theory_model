# src/backtesting/walk_forward.py
"""
Walk-Forward Validation Engine.

Simulates real-world portfolio management:
- Train on past data only
- Apply weights to future (unseen) data
- Roll the window forward and repeat

This produces genuinely out-of-sample performance estimates.
"""

import numpy as np
import pandas as pd
from typing import Callable


def walk_forward_backtest(
    returns: pd.DataFrame,
    optimize_fn: Callable,
    train_window: int = 60,    # months of training data
    test_window: int = 12,     # months of out-of-sample testing
    min_train: int = 36,       # minimum months before first optimization
) -> dict:
    """
    Walk-Forward backtesting engine.

    Parameters
    ----------
    returns      : Full return series (T x N)
    optimize_fn  : Function that takes returns DataFrame and returns
                   a pd.Series of weights
    train_window : Number of months used for optimization
    test_window  : Number of months evaluated out-of-sample
    min_train    : Minimum observations before first optimization

    Returns
    -------
    dict with keys:
        'returns'     : pd.Series of out-of-sample portfolio returns
        'weights'     : pd.DataFrame of weights over time
        'rebalance_dates': list of rebalancing dates
    """
    T = len(returns)
    all_returns = []
    all_weights = []
    rebalance_dates = []

    start = min_train

    while start + test_window <= T:
        # Training window
        train_end = start
        train_start = max(0, train_end - train_window)
        train_data = returns.iloc[train_start:train_end]

        # Test window
        test_data = returns.iloc[start:start + test_window]

        try:
            # Optimize on training data
            weights = optimize_fn(train_data)

            # Apply weights to test data (out-of-sample)
            port_returns = test_data @ weights

            all_returns.append(port_returns)
            all_weights.append(
                pd.Series(weights.values, index=weights.index,
                          name=returns.index[start])
            )
            rebalance_dates.append(returns.index[start])

        except Exception as e:
            print(f"Optimization failed at {returns.index[start]}: {e}")

        start += test_window

    # Combine results
    returns_series = pd.concat(all_returns)
    weights_df = pd.DataFrame(all_weights)

    return {
        'returns':          returns_series,
        'weights':          weights_df,
        'rebalance_dates':  rebalance_dates,
    }


def compare_in_vs_out_of_sample(
    in_sample_returns: pd.Series,
    out_of_sample_returns: pd.Series,
    risk_free_rate: float = 0.043,
    freq: int = 12,
) -> pd.DataFrame:
    """
    Compare in-sample vs out-of-sample performance metrics.
    """
    def metrics(r):
        ann_ret = r.mean() * freq
        ann_vol = r.std() * np.sqrt(freq)
        sharpe  = (ann_ret - risk_free_rate) / ann_vol
        max_dd  = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return {
            'Ann. Return (%)':    round(ann_ret * 100, 2),
            'Ann. Vol (%)':       round(ann_vol * 100, 2),
            'Sharpe Ratio':       round(sharpe, 3),
            'Max Drawdown (%)':   round(max_dd * 100, 2),
        }

    return pd.DataFrame({
        'In-Sample':     metrics(in_sample_returns),
        'Out-of-Sample': metrics(out_of_sample_returns),
    })