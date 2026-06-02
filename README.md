# Portfolio_theory_model

Dati → Pulizia → Stima Ritorni Attesi → Stima Covarianza → Ottimizzazione → Risk Check → Esecuzione
  ↑                                              ↑                  ↑               ↑
  |                  Factor Model / Shrinkage ───┘     QP Solver    |     VaR/CVaR  |
  |                  BARRA / Ledoit-Wolf                (cvxpy)      |    Drawdown   |
  └──────────────────── Ribilanciamento mensile/trimestrale ─────────┴───────────────┘


  config.py              ← Tutto customizzabile qui (asset, A, vincoli, metodi)
src/
  data_fetcher/
    fetcher.py         ← Download Yahoo Finance, pulizia, winsorization
  estimators/
    covariance.py      ← Sample, Ledoit-Wolf, EWMA, Factor/PCA
    returns.py         ← Historical, CAPM equilibrium, Black-Litterman
  optimizers/
    markowitz.py       ← Min-Var, Max-Sharpe, Max-Utility, Frontier (CVXPY)
    risk_parity.py     ← Equal Risk Contribution (ERC)
    cvar_optimizer.py  ← CVaR minimization (LP di Rockafellar-Uryasev)
  risk/
    metrics.py         ← Sharpe, Sortino, Calmar, VaR, CVaR, y*, CAL
  utils/
    plotting.py        ← Tutte le visualizzazioni
notebooks/
  01_data_exploration  ← Prezzi, ritorni, test di normalità
  02_covariance        ← Confronto stimatori, spettro eigenvalori
  03_optimization      ← Tutti i portafogli, frontiera, backtest
  04_capital_alloc     ← y*, CAL, profili investitori, Black-Litterman
  05_full_pipeline     ← Master notebook end-to-end


  
