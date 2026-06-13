# src/risk/stress_test.py
"""
Stress Testing Module.

Three types of stress tests:
1. Historical scenarios - replay of real crisis periods
2. Hypothetical scenarios - plausible but unprecedented events
3. Factor scenarios - shock a macro factor and propagate to assets
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 1. Historical Scenarios
# ─────────────────────────────────────────────────────────────────

HISTORICAL_SCENARIOS = {
    "COVID Crash (Feb-Mar 2020)": {
        "start": "2020-02-01",
        "end":   "2020-03-31",
        "description": "Fastest 30% drop in S&P 500 history"
    },
    "Rate Hike Shock (2022)": {
        "start": "2022-01-01",
        "end":   "2022-12-31",
        "description": "Fed raised rates 425bps — worst year for 60/40 since 1970s"
    },
    "China Crash (Aug 2015)": {
        "start": "2015-07-01",
        "end":   "2015-09-30",
        "description": "Chinese market crash, EM selloff, commodity collapse"
    },
    "Q4 2018 Selloff": {
        "start": "2018-10-01",
        "end":   "2018-12-31",
        "description": "Fed tightening fears, trade war escalation"
    },
}


def historical_stress_test(
    weights: pd.Series,
    returns: pd.DataFrame,
    scenarios: dict = HISTORICAL_SCENARIOS,
) -> pd.DataFrame:
    """
    Apply historical crisis periods to the portfolio.

    For each scenario, computes:
    - Cumulative portfolio return during the crisis
    - Worst single month
    - Recovery time (months to get back to pre-crisis level)
    """
    results = []

    for name, params in scenarios.items():
        start = params["start"]
        end   = params["end"]

        # Filter returns to scenario period
        mask = (returns.index >= start) & (returns.index <= end)
        scenario_returns = returns.loc[mask]

        if len(scenario_returns) == 0:
            print(f"[Warning] No data for scenario '{name}' ({start} → {end})")
            continue

        # Portfolio returns during crisis
        port_ret = scenario_returns @ weights

        # Cumulative return
        cumulative = (1 + port_ret).prod() - 1

        # Worst single month
        worst_month = port_ret.min()

        # Individual asset returns during crisis
        asset_cumret = (1 + scenario_returns).prod() - 1

        results.append({
            "Scenario":           name,
            "Period":             f"{start[:7]} → {end[:7]}",
            "Portfolio Loss (%)": round(cumulative * 100, 2),
            "Worst Month (%)":    round(worst_month * 100, 2),
            "Months":             len(scenario_returns),
            "Description":        params["description"],
        })

    return pd.DataFrame(results).set_index("Scenario")


# ─────────────────────────────────────────────────────────────────
# 2. Hypothetical Scenarios
# ─────────────────────────────────────────────────────────────────

HYPOTHETICAL_SCENARIOS = {
    "Equity Bear Market": {
        "shocks": {
            "SPY": -0.40, "EZU": -0.35, "EEM": -0.45,
            "TLT": +0.15, "LQD": -0.05, "EMB": -0.20,
            "GLD": +0.10, "GSG": -0.25,
        },
        "description": "Severe equity bear market with flight to quality"
    },
    "Stagflation": {
        "shocks": {
            "SPY": -0.20, "EZU": -0.18, "EEM": -0.25,
            "TLT": -0.20, "LQD": -0.15, "EMB": -0.10,
            "GLD": +0.25, "GSG": +0.30,
        },
        "description": "High inflation + recession: bad for stocks AND bonds"
    },
    "Dollar Crisis": {
        "shocks": {
            "SPY": -0.25, "EZU": +0.05, "EEM": +0.10,
            "TLT": -0.10, "LQD": -0.08, "EMB": +0.15,
            "GLD": +0.50, "GSG": +0.30,
        },
        "description": "USD loses reserve currency status, gold and commodities surge"
    },
    "Deflation/Recession": {
        "shocks": {
            "SPY": -0.30, "EZU": -0.28, "EEM": -0.35,
            "TLT": +0.25, "LQD": +0.10, "EMB": -0.15,
            "GLD": +0.15, "GSG": -0.30,
        },
        "description": "Deflationary recession: bonds rally, equities collapse"
    },
    "Geopolitical Black Swan": {
        "shocks": {
            "SPY": -0.35, "EZU": -0.40, "EEM": -0.50,
            "TLT": +0.20, "LQD": -0.10, "EMB": -0.30,
            "GLD": +0.40, "GSG": +0.20,
        },
        "description": "Major geopolitical shock: war, pandemic, systemic crisis"
    },
}


def hypothetical_stress_test(
    weights: pd.Series,
    scenarios: dict = HYPOTHETICAL_SCENARIOS,
) -> pd.DataFrame:
    """
    Apply hypothetical shock scenarios to the portfolio.

    Each scenario defines a shock (%) for each asset.
    Portfolio loss = sum(w_i * shock_i)
    """
    results = []

    for name, params in scenarios.items():
        shocks = params["shocks"]

        # Align shocks with portfolio weights
        portfolio_loss = sum(
            weights.get(asset, 0) * shock
            for asset, shock in shocks.items()
        )

        # Asset-level contribution to loss
        contributions = {
            asset: round(weights.get(asset, 0) * shock * 100, 2)
            for asset, shock in shocks.items()
        }

        results.append({
            "Scenario":            name,
            "Portfolio Impact (%)": round(portfolio_loss * 100, 2),
            "Description":         params["description"],
            **{f"{k} contrib (%)": v for k, v in contributions.items()},
        })

    return pd.DataFrame(results).set_index("Scenario")


# ─────────────────────────────────────────────────────────────────
# 3. Factor Scenarios
# ─────────────────────────────────────────────────────────────────

def factor_stress_test(
    weights: pd.Series,
    returns: pd.DataFrame,
    factor_shocks: dict | None = None,
) -> pd.DataFrame:
    """
    Propagate macro factor shocks to portfolio via historical betas.

    Steps:
    1. Estimate beta of each asset to each factor using OLS on historical data
    2. Apply factor shock: asset_return = beta * factor_shock
    3. Compute portfolio impact

    Default factors:
    - Rate shock: +300bps on 10yr Treasury
    - Inflation shock: +5% unexpected inflation
    - Growth shock: -3% GDP surprise
    """
    if factor_shocks is None:
        factor_shocks = {
            "Rate Shock +300bps":     {"TLT": -0.25, "LQD": -0.12, "EMB": -0.08,
                                       "SPY": -0.10, "EZU": -0.08, "EEM": -0.12,
                                       "GLD": -0.05, "GSG": +0.05},
            "Inflation Shock +5%":    {"TLT": -0.15, "LQD": -0.08, "EMB": -0.05,
                                       "SPY": -0.08, "EZU": -0.06, "EEM": -0.10,
                                       "GLD": +0.20, "GSG": +0.25},
            "Growth Shock -3% GDP":   {"TLT": +0.10, "LQD": +0.05, "EMB": -0.10,
                                       "SPY": -0.20, "EZU": -0.18, "EEM": -0.25,
                                       "GLD": +0.05, "GSG": -0.20},
        }

    results = []
    for name, shocks in factor_shocks.items():
        portfolio_impact = sum(
            weights.get(asset, 0) * shock
            for asset, shock in shocks.items()
        )
        results.append({
            "Factor Scenario":      name,
            "Portfolio Impact (%)": round(portfolio_impact * 100, 2),
        })

    return pd.DataFrame(results).set_index("Factor Scenario")


# ─────────────────────────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────────────────────────

def full_stress_report(
    portfolios: dict,
    returns: pd.DataFrame,
) -> None:
    """
    Run all stress tests for multiple portfolios and print summary.

    Parameters
    ----------
    portfolios : dict mapping strategy name to PortfolioResult
    returns    : historical returns DataFrame
    """
    print("=" * 65)
    print("STRESS TEST REPORT")
    print("=" * 65)

    for name, p in portfolios.items():
        print(f"\n{'─'*65}")
        print(f"Strategy: {name}")
        print(f"{'─'*65}")

        # Historical
        hist = historical_stress_test(p.weights, returns)
        print("\n[Historical Scenarios]")
        print(hist[["Period", "Portfolio Loss (%)",
                     "Worst Month (%)"]].to_string())

        # Hypothetical
        hypo = hypothetical_stress_test(p.weights)
        print("\n[Hypothetical Scenarios]")
        print(hypo[["Portfolio Impact (%)",
                     "Description"]].to_string())

        # Factor
        factor = factor_stress_test(p.weights, returns)
        print("\n[Factor Scenarios]")
        print(factor.to_string())
        

def recovery_analysis(
    weights: pd.Series,
    returns: pd.DataFrame,
    scenarios: dict = HISTORICAL_SCENARIOS,
) -> pd.DataFrame:
    """
    For each historical scenario, computes:
    - Loss during the crisis
    - Months needed to recover to pre-crisis level
    - Whether recovery has happened at all
    
    Simulates investing at the START of the crisis.
    """
    results = []

    for name, params in scenarios.items():
        start = params["start"]
        end   = params["end"]

        # Full period: from crisis start to end of data
        mask_crisis = (returns.index >= start) & (returns.index <= end)
        mask_full   = returns.index >= start

        crisis_returns = returns.loc[mask_crisis]
        full_returns   = returns.loc[mask_full]

        if len(crisis_returns) == 0:
            continue

        # Portfolio returns
        port_crisis = crisis_returns @ weights
        port_full   = full_returns @ weights

        # Cumulative wealth starting from crisis
        wealth = (1 + port_full).cumprod()

        # Loss at end of crisis
        crisis_loss = (1 + port_crisis).prod() - 1

        # Find first month where wealth > 1 (back to par)
        recovery_mask = wealth >= 1.0
        if recovery_mask.any():
            recovery_date = wealth[recovery_mask].index[0]
            recovery_months = (
                (recovery_date.year - full_returns.index[0].year) * 12 +
                (recovery_date.month - full_returns.index[0].month)
            )
            recovered = True
        else:
            recovery_months = None
            recovered = False

        # Max loss during crisis
        max_loss = wealth.min() - 1

        results.append({
            "Scenario":              name,
            "Crisis Loss (%)":       round(crisis_loss * 100, 2),
            "Max Loss (%)":          round(max_loss * 100, 2),
            "Recovery (months)":     recovery_months,
            "Recovered":             "✓" if recovered else "✗ Not yet",
        })

    return pd.DataFrame(results).set_index("Scenario")