from __future__ import annotations

# --- bootstrap imports ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, FIG_DIR
from modules.optimization import (
    mean_variance_stats,
    min_variance_portfolio,
    tangency_portfolio,
    portfolio_performance,
    efficient_frontier,
    shrink_cov_to_diag,
    sharpe_theoretical
)

# ======================================================
# CLI
# ======================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Optimización Markowitz (IN-SAMPLE, ILUSTRATIVA – TESIS)"
    )
    p.add_argument("--rf_annual", type=float, default=0.0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--keep_regex", default=r"^(SPY|LO_\d+)$")
    p.add_argument(
        "--audit_file",
        default=str(DATA_DIR / "momentum_backtest_audit.xlsx"),
        help="Excel auditado con retornos netos (sheet: returns_net)."
    )
    p.add_argument(
        "--sheet",
        default="returns_net",
        help="Hoja del Excel con retornos netos."
    )
    return p.parse_args()

# ======================================================
# MAIN
# ======================================================

def main() -> None:
    args = _parse_args()

    # 1) Cargar retornos NETOS (misma fuente que CAPM y GARCH)
    df = pd.read_excel(
        args.audit_file,
        sheet_name=args.sheet,
        index_col=0,
        parse_dates=True
    ).sort_index()

    # Sanity check de unidades (decimal)
    mean_abs_return = df.abs().mean().mean()
    if mean_abs_return > 0.5:
        raise ValueError(
            "[ERROR DE UNIDADES] Los retornos parecen estar en porcentaje "
            "(ej. 5 = 5%) y no en formato decimal (0.05)."
        )

    # Filtrado parsimonioso
    pat = re.compile(args.keep_regex)
    df = df[[c for c in df.columns if pat.match(c)]]

    if df.empty:
        raise ValueError("No quedaron columnas tras el filtrado keep_regex.")

    # 2) Estadísticos media–varianza
    mu, cov = mean_variance_stats(df, periods_per_year=12)
    cov_sh = shrink_cov_to_diag(cov, args.shrink)

    # 3) Portafolios óptimos (in-sample)
    w_min = min_variance_portfolio(cov, shrink=args.shrink)
    w_tan = tangency_portfolio(mu, cov, rf=args.rf_annual, shrink=args.shrink)

    # 4) Performance teórica (referencia)
    rows = []
    for name, w in [("MinVar", w_min), ("Tangency", w_tan)]:
        perf = portfolio_performance(w, mu, cov, args.shrink)
        sharpe = sharpe_theoretical(
            perf["Return"], perf["Volatility"], args.rf_annual
        )
        rows.append({
            "Portfolio": name,
            "Return": perf["Return"],
            "Volatility": perf["Volatility"],
            "Sharpe_classic": sharpe
        })

    stats = pd.DataFrame(rows)

    # 5) Exportar tablas (ANEXO)
    pd.concat([w_min, w_tan], axis=1).to_excel(
        DATA_DIR / "optimization_weights.xlsx"
    )
    stats.to_excel(
        DATA_DIR / "optimization_stats.xlsx", index=False
    )
    cov_sh.to_excel(
        DATA_DIR / "optimization_covariance_shrinkage.xlsx"
    )

    std = np.sqrt(np.diag(cov_sh))
    corr = cov_sh / np.outer(std, std)
    corr = pd.DataFrame(corr, index=cov_sh.index, columns=cov_sh.columns)

    corr.to_excel(DATA_DIR / "optimization_correlations_correct.xlsx")

    # 6) Frontera eficiente
    ef = efficient_frontier(mu, cov, shrink=args.shrink)

    # 7) Gráfico ilustrativo
    plt.figure(figsize=(9, 6))

    plt.plot(
        ef["Volatility"],
        ef["Return"],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Frontera eficiente (teórica)"
    )

    asset_vol = np.sqrt(np.diag(cov_sh.values))
    asset_ret = mu.values

    plt.scatter(
        asset_vol,
        asset_ret,
        s=70,
        label="SPY y estrategias Long-Only"
    )

    for i, name in enumerate(mu.index):
        if name in ["SPY", "LO_3"]:
            plt.annotate(
                name,
                (asset_vol[i], asset_ret[i]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9
            )

    tan = stats.loc[stats["Portfolio"] == "Tangency"].iloc[0]
    plt.scatter(
        tan["Volatility"],
        tan["Return"],
        s=120,
        marker="D",
        color="black",
        label="Tangency portfolio"
    )

    plt.xlabel("Volatilidad anual")
    plt.ylabel("Retorno esperado anual")
    plt.title("Optimización media–varianza (in-sample, ilustrativa)")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        FIG_DIR / "efficient_frontier.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("✔ Optimización completada correctamente (versión tesis).")

if __name__ == "__main__":
    main()
