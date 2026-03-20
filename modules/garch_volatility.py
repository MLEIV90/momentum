from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from arch import arch_model
except ImportError as e:
    raise ImportError("Falta el paquete 'arch'. Instalá con: pip install arch") from e

from scipy.stats import t as student_t


"""
Módulo GARCH(1,1) para retornos mensuales (tesis, versión parsimoniosa y defendible).

Convenciones:
- Entradas de retornos: escala decimal (ej. 0.01 = 1%).
- El paquete `arch` suele ser numéricamente más estable en escala %; por eso se ajusta el modelo sobre (retornos * 100).
- La volatilidad condicional σ_t se reporta en escala decimal.
- VaR y ES se reportan como PÉRDIDA POSITIVA (por ejemplo, 0.05 = 5% de pérdida) al nivel de confianza 95% (alpha=0.05).

Nota sobre la distribución t:
- En `arch`, con dist='t', las innovaciones están estandarizadas a varianza 1.
- Para calcular cuantiles/ES consistentes se utiliza una t-Student reescalada a var=1:
  scale = sqrt((nu-2)/nu), para nu>2.
"""


# ======================================================
# Ajuste GARCH(1,1)
# ======================================================

def fit_garch_11(
    returns: pd.Series,
    dist: str = "t",
    mean: str = "zero",
):
    """
    Ajusta GARCH(1,1) sobre retornos mensuales en DECIMAL.
    """
    r = returns.dropna()
    if len(r) < 36:
        raise ValueError("Serie demasiado corta para GARCH(1,1). Requiere al menos ~36 observaciones.")

    model = arch_model(
        r * 100.0,                 # escala %
        vol="GARCH",
        p=1, q=1,
        mean=mean,
        dist=dist
    )

    res = model.fit(disp="off")
    return res


def compare_distributions(
    returns: pd.Series,
    mean: str = "zero"
) -> pd.DataFrame:
    """
    Compara Normal vs t-Student (GARCH(1,1)).
    Devuelve tabla con AIC/BIC y loglik.
    """
    rows = []
    for dist in ["normal", "t"]:
        try:
            res = fit_garch_11(returns, dist=dist, mean=mean)
            rows.append({
                "dist": dist,
                "loglik": float(getattr(res, "loglikelihood", np.nan)),
                "aic": float(getattr(res, "aic", np.nan)),
                "bic": float(getattr(res, "bic", np.nan)),
            })
        except Exception as e:
            rows.append({"dist": dist, "loglik": np.nan, "aic": np.nan, "bic": np.nan})

    df = pd.DataFrame(rows)
    if "aic" in df.columns:
        df = df.sort_values("aic", na_position="last")
    return df


# ======================================================
# Outputs: sigma_t, parámetros, VaR/ES
# ======================================================

def conditional_volatility(res) -> pd.Series:
    """
    Devuelve sigma_t condicional en escala DECIMAL (no %).
    """
    sigma = (res.conditional_volatility / 100.0).copy()
    sigma.name = "sigma_t"
    return sigma


def params_table(res, strategy_name: str) -> pd.DataFrame:
    """
    Tabla de parámetros (omega, alpha[1], beta[1], nu si dist=t) con t-stats.
    """
    p = res.params
    tvals = res.tvalues
    out = pd.DataFrame({
        "Estrategia": strategy_name,
        "Parametro": p.index,
        "Estimacion": p.values,
        "tstat": tvals.values
    })
    return out


def garch_var_es(
    res,
    alpha: float = 0.05,
    as_loss: bool = True
) -> pd.DataFrame:
    """
    Calcula VaR y ES condicionales a partir de un ajuste GARCH (arch).

    Retorna un DataFrame indexado por fecha con:
    - sigma_t (decimal)
    - VaR_t (decimal, pérdida positiva si as_loss=True)
    - ES_t (decimal, pérdida positiva si as_loss=True)

    Implementación:
    r_t = mu + sigma_t * z_t, con z_t ~ D(0,1) estandarizada.
    Si dist='t', se usa t-Student con df=nu y escala ajustada a var=1.
    """
    sigma = conditional_volatility(res)

    # media condicional
    mu = 0.0
    if "mu" in res.params.index:
        mu = float(res.params["mu"]) / 100.0  # de % a decimal

    dist_name = getattr(getattr(res.model, "distribution", None), "name", "").lower()

    # Cuantil y ES de la innovación estandarizada z_t
    if "t" in dist_name:
        if "nu" not in res.params.index:
            raise ValueError("El modelo GARCH dist='t' no devolvió parámetro 'nu'.")
        nu = float(res.params["nu"])
        if nu <= 2:
            raise ValueError(f"nu={nu:.3f} no permite varianza finita. No se puede estandarizar a var=1.")

        # escala para que Var(z)=1
        scale = np.sqrt((nu - 2.0) / nu)

        z_alpha = student_t.ppf(alpha, df=nu)            # cuantil (escala 1)
        q = scale * z_alpha                              # cuantil estandarizado (var=1)

        # ES para cola izquierda:
        # E[T | T < z] = - ((nu + z^2)/(nu-1)) * f(z) / alpha
        fz = student_t.pdf(z_alpha, df=nu)
        tail_mean_T = - ( (nu + z_alpha**2) / (nu - 1.0) ) * (fz / alpha)
        es_z = scale * tail_mean_T  # estandarizado (var=1), negativo

    else:
        # Normal estándar (var=1)
        from scipy.stats import norm
        q = norm.ppf(alpha)
        es_z = - norm.pdf(norm.ppf(alpha)) / alpha  # E[Z | Z<q], negativo

    # VaR/ES sobre retornos
    var_ret = mu + sigma * q          # típicamente negativo
    es_ret  = mu + sigma * es_z       # típicamente negativo

    if as_loss:
        var_out = -var_ret
        es_out = -es_ret
        var_out.name = "VaR_t"
        es_out.name = "ES_t"
    else:
        var_out = var_ret.rename("VaR_t")
        es_out = es_ret.rename("ES_t")

    out = pd.concat([sigma, var_out, es_out], axis=1).dropna()
    return out


# ======================================================
# Gráficos
# ======================================================

def plot_conditional_vol(
    sigma: pd.Series,
    title: str,
    outpath: str
) -> None:
    """
    Gráfico simple de sigma_t (parsimonia).
    """
    s = sigma.dropna()
    plt.figure(figsize=(11, 5))
    plt.plot(s.index, s.values, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad condicional (sigma_t)")
    plt.grid(True, alpha=0.3)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_returns_and_volatility(
    returns: pd.Series,
    sigma: pd.Series,
    title: str,
    outpath: str
) -> None:
    """
    Gráfico de 'clustering' (2 paneles):
    - Arriba: retornos mensuales (en % para lectura)
    - Abajo: volatilidad condicional sigma_t (en % para lectura)
    """
    r = returns.dropna()
    s = sigma.dropna()

    # alinear índices
    idx = r.index.intersection(s.index)
    r = r.loc[idx]
    s = s.loc[idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Retornos (en %)
    axes[0].bar(r.index, r.values * 100.0, width=20)
    axes[0].axhline(0, linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_ylabel("Retorno mensual (%)")
    axes[0].grid(True, alpha=0.2)

    # Volatilidad (en %)
    axes[1].plot(s.index, s.values * 100.0, linewidth=1.8)
    axes[1].set_ylabel("σ_t condicional (%)")
    axes[1].set_xlabel("Fecha")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
