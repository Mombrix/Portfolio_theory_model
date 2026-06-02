# src/utils/plotting.py
"""
Visualization utilities for portfolio analysis.

Functions:
- plot_efficient_frontier  : Frontier + special portfolios + CAL
- plot_weights_bar         : Portfolio weights comparison
- plot_risk_contributions  : Risk budget chart
- plot_drawdown            : Underwater chart
- plot_cumulative_returns  : Wealth index comparison
- plot_correlation_heatmap : Asset correlation matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Professional color palette
COLORS = {
    "frontier":   "#2196F3",
    "min_var":    "#4CAF50",
    "max_sharpe": "#FF9800",
    "risk_parity":"#9C27B0",
    "cvar":       "#F44336",
    "max_util":   "#00BCD4",
    "cal":        "#607D8B",
    "rf":         "#9E9E9E",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    portfolios: dict,          # {"label": PortfolioResult, ...}
    risk_free_rate: float = 0.043,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the efficient frontier with special portfolios and CAL.

    Parameters
    ----------
    frontier_df : Output of markowitz.efficient_frontier()
    portfolios  : Dict mapping strategy name to PortfolioResult
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Efficient frontier
    ax.plot(
        frontier_df["volatility"] * 100,
        frontier_df["return"] * 100,
        lw=2.5, color=COLORS["frontier"], label="Efficient Frontier", zorder=2
    )

    # Color frontier by Sharpe ratio
    sc = ax.scatter(
        frontier_df["volatility"] * 100,
        frontier_df["return"] * 100,
        c=frontier_df["sharpe"], cmap="YlOrRd", s=8, zorder=3, alpha=0.7
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Special portfolios
    portfolio_colors = [COLORS["min_var"], COLORS["max_sharpe"],
                        COLORS["risk_parity"], COLORS["cvar"], COLORS["max_util"]]
    markers = ["^", "*", "D", "s", "P"]

    for i, (name, pr) in enumerate(portfolios.items()):
        c = portfolio_colors[i % len(portfolio_colors)]
        ax.scatter(
            pr.volatility * 100, pr.expected_return * 100,
            color=c, s=200, marker=markers[i % len(markers)],
            zorder=5, label=f"{name}\n(SR={pr.sharpe_ratio:.2f})",
            edgecolors="black", linewidths=0.8
        )

    # Capital Allocation Line (from max Sharpe)
    if "Max Sharpe" in portfolios or "Maximum Sharpe (Tangency)" in portfolios:
        key = "Max Sharpe" if "Max Sharpe" in portfolios else "Maximum Sharpe (Tangency)"
        tp = portfolios[key]
        cal_x = np.linspace(0, tp.volatility * 1.5, 100)
        slope = (tp.expected_return - risk_free_rate) / tp.volatility
        cal_y = risk_free_rate + slope * cal_x
        ax.plot(
            cal_x * 100, cal_y * 100,
            "--", color=COLORS["cal"], lw=1.5, label="Capital Allocation Line (CAL)", zorder=1
        )
        ax.scatter([0], [risk_free_rate * 100], color=COLORS["rf"], s=100,
                   marker="o", zorder=5, label=f"Risk-Free ({risk_free_rate*100:.1f}%)")

    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualized Expected Return (%)", fontsize=12)
    ax.set_title("Efficient Frontier & Portfolio Strategies", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_weights_bar(
    portfolios: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing weights across portfolio strategies."""
    weights_df = pd.DataFrame({name: pr.weights for name, pr in portfolios.items()})
    weights_df = weights_df * 100  # percent

    fig, ax = plt.subplots(figsize=(12, 5))
    weights_df.plot(kind="bar", ax=ax, width=0.7, colormap="tab10")

    ax.axhline(y=0, color="black", lw=0.8)
    ax.set_xlabel("Asset", fontsize=11)
    ax.set_ylabel("Weight (%)", fontsize=11)
    ax.set_title("Portfolio Weights by Strategy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_contributions(
    risk_contrib_df: pd.DataFrame,
    title: str = "Risk Contribution by Asset",
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of risk contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Weight
    risk_contrib_df["Weight (%)"].sort_values().plot(
        kind="barh", ax=axes[0], color="#2196F3", edgecolor="white"
    )
    axes[0].set_title("Portfolio Weight (%)", fontsize=11, fontweight="bold")
    axes[0].axvline(x=100/len(risk_contrib_df), color="red", ls="--", lw=1, label="Equal weight")
    axes[0].legend()

    # Risk contribution
    risk_contrib_df["Risk Contrib (%)"].sort_values().plot(
        kind="barh", ax=axes[1], color="#FF9800", edgecolor="white"
    )
    axes[1].set_title("Risk Contribution (%)", fontsize=11, fontweight="bold")
    axes[1].axvline(x=100/len(risk_contrib_df), color="red", ls="--", lw=1, label="Equal risk (ERC)")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_returns(
    returns: pd.DataFrame,
    weights_dict: dict,
    risk_free_rate_monthly: float = 0.043 / 12,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot wealth index (cumulative returns) for multiple strategies.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    colors = list(COLORS.values())
    for i, (name, weights) in enumerate(weights_dict.items()):
        port_returns = returns @ weights
        wealth = (1 + port_returns).cumprod()
        axes[0].plot(wealth.index, wealth.values, lw=1.8, color=colors[i], label=name)

    # Risk-free baseline
    rf_wealth = (1 + risk_free_rate_monthly) ** np.arange(len(returns) + 1)
    axes[0].plot(
        returns.index, rf_wealth[1:], "--", lw=1.2, color=COLORS["rf"], label="Risk-Free"
    )

    axes[0].set_ylabel("Wealth Index (start=1)", fontsize=11)
    axes[0].set_title("Cumulative Portfolio Returns", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Drawdown for first strategy
    first_name, first_weights = next(iter(weights_dict.items()))
    port_returns_first = returns @ first_weights
    dd = (1 + port_returns_first).cumprod()
    dd = (dd / dd.cummax() - 1) * 100
    axes[1].fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4, label=f"Drawdown: {first_name}")
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Correlation heatmap with hierarchical clustering."""
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Asset Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
