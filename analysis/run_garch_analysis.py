from __future__ import annotations

# --- bootstrap de imports: permite ejecutar este script desde cualquier directorio ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
import numpy as np
import pandas as pd

from config import DATA_DIR, FIG_DIR
from modules.garch_volatility import (
    fit_garch_11,
    compare_distributions,
    conditional_volatility,
    params_table,
    garch_var_es,
    plot_returns_and_volatility,
)

# ======================================================
# CLI
# ======================================================

def _parse_args():
    p = argparse.ArgumentParser(description="GARCH(1,1) + VaR/ES condicional (t-Student) – Estrategias Long-Only.")
    p.add_argument(
        "--audit_file",
        default=str(DATA_DIR / "momentum_backtest_audit.xlsx"),
        help="Excel del backtest auditado con retornos netos (sheet: returns_net)."
    )
    p.add_argument(
        "--sheet",
        default="returns_net",
        help="Nombre de la hoja con retornos netos (default: returns_net)."
    )
    p.add_argument(
        "--dist",
        default="t",
        choices=["normal", "t"],
        help="Distribución para el ajuste final (por defecto t para colas)."
    )
    p.add_argument(
        "--mean",
        default="zero",
        choices=["zero", "constant"],
        help="Media en el modelo (por defecto zero)."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nivel de cola para VaR/ES (default 0.05 => 95% de confianza)."
    )
    p.add_argument(
        "--only_long_only",
        action="store_true",
        default=True,
        help="Filtra y deja SOLO SPY y estrategias Long-Only."
    )
    p.add_argument(
        "--keep_regex",
        default=r"^(SPY|LO_\d+)$",
        help="Regex a conservar si only_long_only=True. Default: SPY y LO_#."
    )
    p.add_argument(
        "--main_strategy",
        default="LO_3",
        help="Estrategia principal para 1 único gráfico de clustering (retornos + sigma_t)."
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Si se activa, no genera gráfico."
    )
    return p.parse_args()


# ======================================================
# MAIN
# ======================================================

def main():
    args = _parse_args()

    # 1) Retornos NETOS del backtest auditado (misma 'fuente de verdad' que CAPM)
    rets = pd.read_excel(
        args.audit_file,
        sheet_name=args.sheet,
        index_col=0,
        parse_dates=True
    ).sort_index()
    
    # 🔒 Blindaje temporal: usar solo el período común a todas las estrategias
    rets = rets.dropna(how="any")

    # Sanity check de unidades (retornos en decimales)
    mean_abs_return = rets.abs().mean().mean()
    if mean_abs_return > 0.5:
        raise ValueError(
            f"[ERROR DE UNIDADES] El retorno medio absoluto es {mean_abs_return:.2f}. "
            "Esto sugiere que los retornos están en porcentaje (ej. 5 = 5%) "
            "y no en formato decimal (0.05). Verifique la hoja de retornos."
        )
    elif mean_abs_return > 0.2:
        print(
            f"[ADVERTENCIA] El retorno medio absoluto es {mean_abs_return:.2f}. "
            "Verifique que los retornos estén efectivamente en formato decimal."
        )

    # Filtrado (parsimonia)
    if args.only_long_only:
        pat = re.compile(args.keep_regex)
        cols = [c for c in rets.columns if pat.match(str(c))]
        rets = rets[cols]

    strategies = list(rets.columns)
    if len(strategies) == 0:
        raise ValueError("No se encontraron estrategias tras el filtrado. Revisá keep_regex/only_long_only.")

    rows_cmp = []
    all_params = []
    rows_metrics = []

    for s in strategies:
        r = rets[s].dropna()

        # comparación Normal vs t (para justificar colas)
        cmp_df = compare_distributions(r, mean=args.mean)

        if cmp_df is not None and not cmp_df.empty:
            cmp_df = cmp_df.copy()
            cmp_df["Estrategia"] = s
            rows_cmp.append(cmp_df)


        # ajuste final con dist elegida (default t)
        try:
            res = fit_garch_11(r, dist=args.dist, mean=args.mean)
            all_params.append(params_table(res, s))

            sigma = conditional_volatility(res)
            risk_df = garch_var_es(res, alpha=args.alpha, as_loss=True)

            # Métricas resumen (promedios en escala mensual)
            mean_sigma_m = float(risk_df["sigma_t"].mean())
            mean_sigma_a = float(mean_sigma_m * np.sqrt(12.0))

            mean_var = float(risk_df["VaR_t"].mean())
            mean_es = float(risk_df["ES_t"].mean())

            rows_metrics.append({
                "Estrategia": s,
                "Media sigma mensual": mean_sigma_m,
                "Media sigma anualizada": mean_sigma_a,
                f"VaR {int((1-args.alpha)*100)}% (pérdida) - promedio": mean_var,
                f"ES {int((1-args.alpha)*100)}% (pérdida) - promedio": mean_es,
                "N": int(risk_df.shape[0]),
                "dist": args.dist,
                "mean": args.mean
            })

            # Gráfico SOLO para main_strategy (parsimonia): retornos + sigma_t
            if (not args.no_plot) and (s == args.main_strategy):
                outp = FIG_DIR / f"garch_clustering_{s}.png"
                plot_returns_and_volatility(
                    returns=r,
                    sigma=sigma,
                    title=f"Clustering de volatilidad – GARCH(1,1) ({args.dist}) – {s}",
                    outpath=str(outp)
                )
                print(f"✔ Gráfico clustering guardado en: {outp}")

        except Exception as e:
            print(f"⚠ GARCH falló para {s}: {e}")

    # 2) Guardar comparación de distribuciones
    cmp_out = pd.concat(rows_cmp, ignore_index=True)

    cmp_path = DATA_DIR / "garch_compare_distributions.xlsx"
    cmp_out.to_excel(cmp_path, index=False)
    print(f"✔ Comparación Normal vs t guardada en: {cmp_path}")

    # 3) Guardar parámetros
    if all_params:
        params_out = pd.concat(all_params, ignore_index=True)
        params_path = DATA_DIR / "garch_params_summary.xlsx"
        params_out.to_excel(params_path, index=False)
        print(f"✔ Parámetros GARCH guardados en: {params_path}")
    else:
        print("⚠ No se generaron parámetros GARCH (todas las estrategias fallaron o series muy cortas).")

    # 4) Guardar métricas VaR/ES (GARCH) resumen
    if rows_metrics:
        metrics_out = pd.DataFrame(rows_metrics)
        metrics_path = DATA_DIR / "garch_risk_metrics.xlsx"
        metrics_out.to_excel(metrics_path, index=False)
        print(f"✔ Métricas VaR/ES (GARCH) guardadas en: {metrics_path}")
    else:
        print("⚠ No se generaron métricas VaR/ES. Revisá errores anteriores.")

    print("✔ GARCH + VaR/ES completado correctamente.")


if __name__ == "__main__":
    main()
