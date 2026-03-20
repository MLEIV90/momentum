from __future__ import annotations
import numpy as np
import pandas as pd

"""
Métricas de desempeño y riesgo (TESIS – versión parsimoniosa y reproducible).

Convenciones:
- Retornos mensuales en escala DECIMAL (0.01 = 1%).
- Annualización consistente (periods_per_year).
- Diseñado para estrategias Long-Only.
"""

# ======================================================
# Performance básica
# ======================================================

def annualized_return(r: pd.Series, periods_per_year: int = 12) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    total = (1.0 + r).prod()
    years = len(r) / periods_per_year
    return total ** (1.0 / years) - 1.0


def annualized_volatility(r: pd.Series, periods_per_year: int = 12) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    r: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 12
) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan

    rf_per = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per

    mu = excess.mean() * periods_per_year
    vol = excess.std(ddof=1) * np.sqrt(periods_per_year)

    return np.nan if vol == 0 else mu / vol


def sharpe_lo_adjusted(
    r: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 12,
    max_lag: int = 6
) -> float:
    """
    Sharpe ajustado por autocorrelación (Lo, 2002).
    """
    r = r.dropna()
    if len(r) < 12:
        return np.nan

    sr = sharpe_ratio(r, rf_annual, periods_per_year)
    if not np.isfinite(sr):
        return sr

    T = len(r)
    m = min(max_lag, T - 2)

    denom = 1.0
    for k in range(1, m + 1):
        rho = r.autocorr(lag=k)
        if np.isfinite(rho):
            denom += 2.0 * rho * (1.0 - k / T)

    return np.nan if denom <= 0 else sr / np.sqrt(denom)


def equity_curve(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1.0 + r).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()

# ======================================================
# Riesgo extremo histórico
# ======================================================

def historical_var(r: pd.Series, alpha: float = 0.95) -> float:
    r = r.dropna()
    return np.nan if r.empty else np.quantile(r, 1.0 - alpha)


def historical_es(r: pd.Series, alpha: float = 0.95) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return np.nan if tail.empty else tail.mean()

# ======================================================
# Backtesting VaR (Kupiec)
# ======================================================

def rolling_historical_var(
    r: pd.Series,
    window: int = 60,
    alpha: float = 0.95
) -> pd.Series:
    r = r.dropna()
    return r.rolling(window).quantile(1.0 - alpha)


def kupiec_pof(exceptions: np.ndarray, alpha: float) -> dict:
    x = int(exceptions.sum())
    n = int(len(exceptions))
    p = 1.0 - alpha

    if n == 0:
        return {"N": 0, "Violaciones": 0, "p_hat": np.nan, "LR": np.nan, "pvalue": np.nan}

    p_hat = x / n

    def slog(a):
        return np.log(a) if a > 0 else -1e12

    ll_null = (n - x) * slog(1 - p) + x * slog(p)
    ll_alt = (
        (n - x) * slog(1 - p_hat) + x * slog(p_hat)
        if 0 < p_hat < 1 else ll_null
    )

    LR = -2.0 * (ll_null - ll_alt)

    try:
        from scipy.stats import chi2
        pval = 1.0 - chi2.cdf(LR, df=1)
    except Exception:
        pval = np.nan

    return {"N": n, "Violaciones": x, "p_hat": p_hat, "LR": LR, "pvalue": pval}


def backtest_var_kupiec(
    r: pd.Series,
    window: int = 60,
    alpha: float = 0.95
) -> dict:
    r = r.dropna()
    var = rolling_historical_var(r, window, alpha)
    df = pd.concat([r, var], axis=1).dropna()

    if df.empty:
        return {"N": 0, "Violaciones": 0, "p_hat": np.nan, "LR": np.nan, "pvalue": np.nan}

    exceptions = df.iloc[:, 0] < df.iloc[:, 1]
    return kupiec_pof(exceptions.values, alpha)


# ======================================================
# Resúmenes
# ======================================================

def summarize_strategy(
    r: pd.Series,
    name: str,
    rf_annual: float = 0.0,
    periods_per_year: int = 12
) -> dict:
    return {
        "Estrategia": name,
        "CAGR": annualized_return(r, periods_per_year),
        "Volatilidad": annualized_volatility(r, periods_per_year),
        "Sharpe": sharpe_ratio(r, rf_annual, periods_per_year),
        "Sharpe_Lo": sharpe_lo_adjusted(r, rf_annual, periods_per_year),
        "MaxDrawdown": max_drawdown(r),
        "VaR_95": historical_var(r, 0.95),
        "ES_95": historical_es(r, 0.95),
    }


def summarize_multiple(
    strategies: dict[str, pd.Series],
    rf_annual: float = 0.0,
    periods_per_year: int = 12
) -> pd.DataFrame:
    rows = [
        summarize_strategy(r, name, rf_annual, periods_per_year)
        for name, r in strategies.items()
    ]
    return pd.DataFrame(rows)
