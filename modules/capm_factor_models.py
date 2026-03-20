from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

"""
CAPM para estrategias Long-Only (TESIS – versión definitiva).

Objetivo:
- Estimar alpha y beta de cada estrategia vs el mercado (Mkt - Rf),
  usando retornos mensuales NETOS y una tasa libre de riesgo DINÁMICA.
- Inferencia robusta a heterocedasticidad y autocorrelación mediante
  errores estándar HAC (Newey–West).

Notas metodológicas (defendibles):
- CAPM se utiliza como benchmark interpretativo, no como modelo exhaustivo.
- La tasa libre de riesgo es consistente con la utilizada en métricas de performance.
- Annualización del alpha mediante capitalización compuesta.
"""

# ======================================================
# Helpers de alineación temporal
# ======================================================

def to_month_end_index(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.to_datetime(out.index) + pd.offsets.MonthEnd(0)
    return out

def align_on_common_index(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.intersection(b.index)
    return a.loc[idx].sort_index(), b.loc[idx].sort_index()

# ======================================================
# Construcción del dataframe CAPM (excesos)
# ======================================================

def build_capm_df(strategy_ret: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataframe de retornos en exceso.

    Requiere:
    - factors con columnas:
        * 'RF'     : tasa libre de riesgo mensual (decimal)
        * 'Mkt-RF' : exceso de retorno del mercado (decimal)
    - strategy_ret: retornos mensuales de la estrategia (decimal, NETOS)

    Devuelve:
    - DataFrame con r_e = r_p - RF alineado temporalmente.
    """
    if not {"RF", "Mkt-RF"}.issubset(factors.columns):
        raise ValueError("Factores deben incluir columnas 'RF' y 'Mkt-RF'.")

    r = to_month_end_index(strategy_ret.dropna())
    f = factors.copy()
    f.index = pd.to_datetime(f.index) + pd.offsets.MonthEnd(0)

    rf = f["RF"].dropna()
    mkt = f["Mkt-RF"].dropna()

    r, rf = align_on_common_index(r, rf)
    r, mkt = align_on_common_index(r, mkt)

    df = pd.DataFrame(index=r.index)
    df["r"] = r
    df["RF"] = rf
    df["Mkt-RF"] = mkt
    df["r_e"] = df["r"] - df["RF"]

    return df.dropna()

# ======================================================
# Estimación CAPM con HAC (Newey–West)
# ======================================================

def run_capm(
    strategy_ret: pd.Series,
    factors: pd.DataFrame,
    hac_lags: int = 3
) -> dict:
    """
    Estima el modelo CAPM:

        r_e = alpha + beta * (Mkt - RF) + epsilon

    utilizando errores estándar HAC (Newey–West).

    Devuelve un diccionario con resultados clave para reporte académico.
    """
    df = build_capm_df(strategy_ret, factors)

    y = df["r_e"]
    X = sm.add_constant(df["Mkt-RF"])

    model = sm.OLS(y, X)
    res = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(hac_lags)}
    )

    alpha_m = float(res.params.get("const", np.nan))
    beta = float(res.params.get("Mkt-RF", np.nan))

    # Annualización correcta (capitalización compuesta)
    alpha_a = float((1.0 + alpha_m) ** 12 - 1.0)

    return {
        "alpha_m": alpha_m,
        "alpha_a": alpha_a,
        "t_alpha": float(res.tvalues.get("const", np.nan)),
        "p_alpha": float(res.pvalues.get("const", np.nan)),
        "beta": beta,
        "t_beta": float(res.tvalues.get("Mkt-RF", np.nan)),
        "p_beta": float(res.pvalues.get("Mkt-RF", np.nan)),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "hac_lags": int(hac_lags),
        # Trazabilidad (útil para anexo o defensa)
        "data": df,
        "res": res,
    }

# ======================================================
# Tabla resumen CAPM (para el cuerpo de la tesis)
# ======================================================

def capm_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Construye la tabla CAPM estándar para el cuerpo del Capítulo 4.6.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Estrategia": name,
            "Alpha_mensual": r["alpha_m"],
            "Alpha_anual": r["alpha_a"],
            "t(Alpha)": r["t_alpha"],
            "Beta": r["beta"],
            "t(Beta)": r["t_beta"],
            "R2": r["r2"],
            "N": r["n"],
            "HAC_lags": r["hac_lags"],
        })
    return pd.DataFrame(rows)
