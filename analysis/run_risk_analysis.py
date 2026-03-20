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

from config import DATA_DIR
from modules.risk_measures import summarize_multiple, backtest_var_kupiec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Métricas de riesgo y desempeño complementarias (ANEXO B – TESIS)."
    )
    p.add_argument(
        "--audit_file",
        default=str(DATA_DIR / "momentum_backtest_audit.xlsx"),
        help="Excel auditado con retornos netos (sheet: returns_net)."
    )
    p.add_argument(
        "--sheet",
        default="returns_net",
        help="Hoja del Excel con retornos mensuales netos."
    )
    p.add_argument(
        "--rf_annual",
        type=float,
        default=0.0,
        help="Tasa libre de riesgo anual."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Nivel de confianza para VaR/ES."
    )
    p.add_argument(
        "--var_window",
        type=int,
        default=60,
        help="Ventana rolling (meses) para VaR histórico."
    )
    # ✅ FIX: \d debe ser un dígito real en regex (no literal)
    p.add_argument(
        "--keep_regex",
        default=r"^(SPY|LO_\d+)$",
        help="Columnas a conservar (parsimonia)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # 1) Cargar retornos NETOS (misma fuente que CAPM, GARCH y optimización)
    df = pd.read_excel(
        args.audit_file,
        sheet_name=args.sheet,
        index_col=0,
        parse_dates=True
    ).sort_index()

    # Limpieza defensiva: nombres de columnas sin espacios raros
    df.columns = [str(c).strip() for c in df.columns]

    # Sanity check de unidades (retornos en decimal)
    mean_abs = df.abs().mean().mean()
    if mean_abs > 0.5:
        raise ValueError(
            "[ERROR DE UNIDADES] Los retornos parecen estar en porcentaje "
            "y no en formato decimal."
        )

    # 2) Filtrado parsimonioso
    pat = re.compile(args.keep_regex)
    keep_cols = [c for c in df.columns if pat.match(c)]
    df = df[keep_cols]

    if df.empty:
        raise ValueError(
            "No quedaron estrategias tras el filtrado. "
            f"Regex usado: {args.keep_regex}. Columnas disponibles: {list(df.columns)}"
        )

    strategies = {c: df[c].dropna() for c in df.columns}
    print("✔ Estrategias analizadas:", list(strategies.keys()))

    # 3) Tabla de métricas complementarias (ANEXO B)
    summary = summarize_multiple(
        strategies,
        rf_annual=args.rf_annual,
        periods_per_year=12
    )

    out_perf = DATA_DIR / "annexB_risk_performance_summary.xlsx"
    summary.to_excel(out_perf, index=False)
    print(f"✔ Tabla Anexo B guardada en: {out_perf}")

    # 4) Backtesting VaR (Kupiec)
    rows_bt = []
    for name, r in strategies.items():
        bt = backtest_var_kupiec(
            r,
            window=args.var_window,
            alpha=args.alpha
        )
        rows_bt.append({"Estrategia": name, **bt})

    bt_df = pd.DataFrame(rows_bt)
    out_bt = DATA_DIR / "annexB_var_backtest_kupiec.xlsx"
    bt_df.to_excel(out_bt, index=False)
    print(f"✔ Backtesting VaR (Anexo B) guardado en: {out_bt}")

    print("✔ Análisis de riesgo complementario completado.")


if __name__ == "__main__":
    main()
