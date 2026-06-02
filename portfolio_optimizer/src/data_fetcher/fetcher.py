# src/data_fetcher/fetcher.py
"""
Data fetching and cleaning module.
Downloads price data from Yahoo Finance, computes returns,
handles missing values and outliers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1mo",
    cache_dir: str = "data/",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Returns a DataFrame of shape (T, N) with dates as index and tickers as columns.
    Uses a local CSV cache to avoid redundant downloads.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"prices_{interval}_{start}.csv")

    if use_cache and os.path.exists(cache_file):
        print(f"[Cache] Loading prices from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"[Download] Fetching {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # Extract 'Close' prices (adjusted for splits and dividends)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices[tickers]  # Enforce consistent column order

    # Forward-fill up to 3 periods (handles holidays, stale prices)
    prices = prices.ffill(limit=3)

    # Drop rows where more than 50% of assets are missing
    threshold = int(0.5 * len(tickers))
    prices = prices.dropna(thresh=threshold)

    print(f"[Download] Shape: {prices.shape} | Period: {prices.index[0].date()} → {prices.index[-1].date()}")

    prices.to_csv(cache_file)
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
    winsorize_sigma: float | None = 3.0,
) -> pd.DataFrame:
    """
    Compute period returns from price series.

    Parameters
    ----------
    method : 'log' (continuously compounded) or 'simple' (arithmetic)
    winsorize_sigma : if not None, clip returns beyond ±N standard deviations
                      to reduce the impact of outliers (data errors, extreme events)
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

    returns = returns.iloc[1:]  # Drop first NaN row

    if winsorize_sigma is not None:
        mu = returns.mean()
        sigma = returns.std()
        lower = mu - winsorize_sigma * sigma
        upper = mu + winsorize_sigma * sigma
        returns = returns.clip(lower=lower, upper=upper, axis=1)

    # Final check: drop columns with too many NaNs (>10%)
    max_nan_frac = 0.10
    nan_fracs = returns.isna().mean()
    bad_cols = nan_fracs[nan_fracs > max_nan_frac].index.tolist()
    if bad_cols:
        print(f"[Warning] Dropping columns with >10% NaN: {bad_cols}")
        returns = returns.drop(columns=bad_cols)

    returns = returns.dropna()
    return returns


def get_market_caps(tickers: list[str]) -> pd.Series:
    """
    Fetch approximate market capitalizations for computing
    market-cap-weighted equilibrium portfolio (used in Black-Litterman).
    Returns a Series normalized to sum to 1 (portfolio weights).
    """
    caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap") or info.get("totalAssets") or np.nan
            caps[ticker] = cap
        except Exception:
            caps[ticker] = np.nan

    s = pd.Series(caps).dropna()

    # Fill missing with equal weight
    missing = [t for t in tickers if t not in s.index]
    if missing:
        avg = s.mean()
        for t in missing:
            s[t] = avg
        s = s[tickers]

    return s / s.sum()


def describe_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary statistics table: annualized mean, vol, Sharpe proxy,
    skewness, kurtosis, min, max.
    Assumes monthly returns by default (12 periods/year).
    """
    freq = 12  # monthly
    stats = pd.DataFrame({
        "Ann. Return (%)":  (returns.mean() * freq * 100).round(2),
        "Ann. Vol (%)":     (returns.std() * np.sqrt(freq) * 100).round(2),
        "Skewness":         returns.skew().round(3),
        "Excess Kurtosis":  (returns.kurt()).round(3),
        "Min (%)":          (returns.min() * 100).round(2),
        "Max (%)":          (returns.max() * 100).round(2),
        "Obs.":             returns.count(),
    })
    stats["Sharpe (proxy)"] = (
        stats["Ann. Return (%)"] / stats["Ann. Vol (%)"]
    ).round(3)
    return stats
