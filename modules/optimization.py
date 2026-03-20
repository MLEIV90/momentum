from __future__ import annotations

import numpy as np
import pandas as pd

"""
Optimización media-varianza (Markowitz) – versión CONSISTENTE y defendible (TESIS).

Convenciones:
- Retornos y covarianzas ANUALIZADOS.
- TODO se calcula con la MISMA matriz de covarianzas (shrinked).
- Uso estrictamente IN-SAMPLE (ejercicio ilustrativo).
"""

# ======================================================
# Estadísticos básicos
# ======================================================

def mean_variance_stats(
    returns: pd.DataFrame,
    periods_per_year: int = 12
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calcula retornos esperados y matriz de covarianzas anualizadas.
    """
    r = returns.dropna(how="any")
    mu = r.mean() * periods_per_year
    cov = r.cov() * periods_per_year
    return mu, cov

# ======================================================
# Shrinkage
# ======================================================

def shrink_cov_to_diag(cov: pd.DataFrame, shrink: float = 0.1) -> pd.DataFrame:
    """
    Shrinkage lineal hacia la matriz diagonal.
    """
    lam = float(np.clip(shrink, 0.0, 1.0))
    diag = np.diag(np.diag(cov.values))
    sh = (1.0 - lam) * cov.values + lam * diag
    return pd.DataFrame(sh, index=cov.index, columns=cov.columns)

def _pinv(x: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(x)

# ======================================================
# Portafolios óptimos (Σ_sh)
# ======================================================

def min_variance_portfolio(cov: pd.DataFrame, shrink: float = 0.1) -> pd.Series:
    cov_sh = shrink_cov_to_diag(cov, shrink)
    inv = _pinv(cov_sh.values)
    ones = np.ones(len(cov_sh))
    w = inv @ ones
    w = w / w.sum()
    return pd.Series(w, index=cov_sh.columns, name="MinVar")

def tangency_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    shrink: float = 0.1
) -> pd.Series:
    cov_sh = shrink_cov_to_diag(cov, shrink)
    inv = _pinv(cov_sh.values)
    excess = mu - rf
    w = inv @ excess.values
    if np.allclose(w.sum(), 0):
        w = np.ones_like(w)
    w = w / w.sum()
    return pd.Series(w, index=mu.index, name="Tangency")

# ======================================================
# Performance teórica (consistente con Σ_sh)
# ======================================================

def portfolio_performance(
    w: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    shrink: float = 0.1
) -> dict:
    """
    Retorno esperado y volatilidad anual del portafolio.
    """
    cov_sh = shrink_cov_to_diag(cov, shrink)
    ret = float(w @ mu)
    vol = float(np.sqrt(w.values @ cov_sh.values @ w.values))
    return {"Return": ret, "Volatility": vol}

def sharpe_theoretical(ret: float, vol: float, rf: float = 0.0) -> float:
    """
    Sharpe clásico teórico (referencia geométrica).
    """
    return np.nan if vol == 0 else (ret - rf) / vol

# ======================================================
# Frontera eficiente (Σ_sh)
# ======================================================

def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 40,
    shrink: float = 0.1
) -> pd.DataFrame:
    cov_sh = shrink_cov_to_diag(cov, shrink)
    inv = _pinv(cov_sh.values)
    ones = np.ones(len(mu))

    A = ones @ inv @ ones
    B = ones @ inv @ mu.values
    C = mu.values @ inv @ mu.values
    D = A * C - B**2

    if abs(D) < 1e-10:
        raise RuntimeError("Matriz casi singular: frontera no identificable.")

    targets = np.linspace(mu.min(), mu.max(), n_points)
    mus, vols = [], []

    for m in targets:
        l1 = (C - B * m) / D
        l2 = (A * m - B) / D
        w = inv @ (l1 * ones + l2 * mu.values)
        w = w / w.sum()
        mus.append(float(w @ mu.values))
        vols.append(float(np.sqrt(w @ cov_sh.values @ w)))

    return pd.DataFrame({"Return": mus, "Volatility": vols})
