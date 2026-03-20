from __future__ import annotations

# --- bootstrap de imports ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, FIG_DIR, DOWNLOADS_DIR
from modules.capm_factor_models import run_capm, capm_table

# ======================================================
# CLI
# ======================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAPM (HAC Newey–West) – Estrategias Long-Only"
    )
    p.add_argument(
        "--audit_file",
        default=str(DATA_DIR / "momentum_backtest_audit.xlsx"),
        help="Excel del backtest con retornos netos (sheet: returns_net)."
    )
    p.add_argument(
        "--ff_file",
        default=str(DOWNLOADS_DIR / "F-F_Research_Data_Factors.xlsx"),
        help="Excel Fama-French mensual (RF y Mkt-RF)."
    )
    p.add_argument("--sheet", default=0)
    p.add_argument("--hac_lags", type=int, default=3)
    p.add_argument("--only_long_only", action="store_true", default=True)
    p.add_argument("--keep_regex", default=r"^(SPY|LO_\d+)$")
    p.add_argument("--main_strategy", default="LO_3")
    p.add_argument("--no_scatter", action="store_true")
    return p.parse_args()

# ======================================================
# Loader robusto de factores
# ======================================================

def load_factors(ff_path: str, sheet) -> pd.DataFrame:
    ff_raw = pd.read_excel(ff_path, sheet_name=sheet)

    date_col = next(
        (c for c in ff_raw.columns if ff_raw[c].dtype != "float64"),
        None
    )
    if date_col is None:
        raise ValueError("No se encontró columna de fecha en el archivo FF.")

    ff_raw[date_col] = ff_raw[date_col].astype(str).str.replace(r"\D+", "", regex=True)
    ff_raw["Date"] = pd.to_datetime(
        ff_raw[date_col].str[:4] + ff_raw[date_col].str[4:] + "01",
        errors="coerce"
    ) + pd.offsets.MonthEnd(0)

    ff = ff_raw.dropna(subset=["Date"]).set_index("Date").copy()

    for col in ["Mkt-RF", "RF"]:
        if col not in ff.columns:
            raise ValueError(f"Falta columna '{col}' en factores.")
        ff[col] = ff[col] / 100.0

    return ff[["Mkt-RF", "RF"]].dropna()

# ======================================================
# Scatter CAPM (parsimonioso)
# ======================================================

def plot_scatter_capm(df_excess: pd.DataFrame, res, name: str, outpath: Path) -> None:
    x = df_excess["Mkt-RF"]
    y = df_excess["r_e"]

    alpha = res.params["const"]
    beta = res.params["Mkt-RF"]
    r2 = res.rsquared

    # Recta CAPM
    x_line = pd.Series([x.min(), x.max()])
    y_line = alpha + beta * x_line

    plt.figure(figsize=(8, 6))

    # Scatter
    plt.scatter(
        x,
        y,
        alpha=0.45,
        s=18,
        edgecolor="none"
    )

    # Recta CAPM
    plt.plot(
        x_line,
        y_line,
        linewidth=2.5,
        label="Recta de regresión CAPM"
    )

    # Ejes en cero
    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(0, linestyle="--", linewidth=1, alpha=0.7)

    # Etiquetas
    plt.xlabel("Retorno del mercado en exceso (Mkt − RF)")
    plt.ylabel("Retorno de la estrategia en exceso (Estrategia − RF)")

    # Título académico
    plt.title(f"Modelo CAPM – Estrategia {name}", fontsize=12)

    # Cuadro informativo
    textstr = (
        f"α mensual = {alpha*100:.2f}%\n"
        f"β = {beta:.3f}\n"
        f"R² = {r2:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.grid(alpha=0.3)
    plt.legend(frameon=False)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================
# MAIN
# ======================================================

def main() -> None:
    args = _parse_args()

    # 1) Retornos NETOS del backtest
    df = pd.read_excel(
        args.audit_file,
        sheet_name="returns_net",
        index_col=0,
        parse_dates=True
    ).sort_index()
    # 🔒 Blindaje temporal: usar solo el período común a todas las estrategias
    df = df.dropna(how="any")

    
    # ======================================================
    # Sanity check de unidades (retornos en decimales)
    # ======================================================

    mean_abs_return = df.abs().mean().mean()

    if mean_abs_return > 0.5:
        raise ValueError(
            f"[ERROR DE UNIDADES] El retorno medio absoluto es {mean_abs_return:.2f}. "
            "Esto sugiere que los retornos están en porcentaje (ej. 5 = 5%) "
            "y no en formato decimal (0.05). Verifique el archivo 'returns_net'."
        )

    elif mean_abs_return > 0.2:
        print(
            f"[ADVERTENCIA] El retorno medio absoluto es {mean_abs_return:.2f}. "
            "Verifique que los retornos estén efectivamente en formato decimal."
        )


    if args.only_long_only:
        pat = re.compile(args.keep_regex)
        df = df[[c for c in df.columns if pat.match(c)]]

    # 2) Factores
    ff = load_factors(args.ff_file, args.sheet)
    ff = ff.loc[df.index.min():df.index.max()]

    # 3) Estimar CAPM
    results = {}
    for col in df.columns:
        results[col] = run_capm(df[col], ff, hac_lags=args.hac_lags)

    # 4) Tabla CAPM
    table = capm_table(results)
    out_table = DATA_DIR / "capm_summary.xlsx"
    table.to_excel(out_table, index=False)
    print(f"✔ Tabla CAPM guardada en: {out_table}")

    # 5) Scatter (una estrategia)
    if not args.no_scatter and args.main_strategy in results:
        out_fig = FIG_DIR / f"scatter_capm_{args.main_strategy}.png"
        plot_scatter_capm(
            results[args.main_strategy]["data"],
            results[args.main_strategy]["res"],
            args.main_strategy,
            out_fig
        )
        print(f"✔ Scatter CAPM guardado en: {out_fig}")

    print("✔ CAPM completado correctamente.")

if __name__ == "__main__":
    main()
