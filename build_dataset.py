from __future__ import annotations

import argparse
from datetime import timedelta
import pandas as pd
import yfinance as yf

from config import (
    DATA_DIR,
    RAW_START_DATE,
    END_DATE,
    TICKERS_DEFAULT
)

# =====================================================
# CLI
# =====================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descarga precios y construye dataset mensual (fin de mes, reproducible)."
    )
    p.add_argument(
        "--start",
        default=RAW_START_DATE,
        help="Fecha inicio descarga (YYYY-MM-DD)."
    )
    p.add_argument(
        "--end",
        default=END_DATE,
        help="Fecha fin del dataset (YYYY-MM-DD, mes completo)."
    )
    p.add_argument(
        "--tickers",
        nargs="*",
        default=TICKERS_DEFAULT,
        help="Lista de tickers (ETFs sectoriales + benchmark)."
    )
    return p.parse_args()

# =====================================================
# MAIN
# =====================================================

def main() -> None:
    args = _parse_args()
    tickers = list(dict.fromkeys(args.tickers))  # elimina duplicados, preserva orden

    # --- asegurar inclusión del último día ---
    end_date = pd.to_datetime(args.end)
    end_inclusive = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    print("=================================================")
    print(" Construcción del dataset mensual (AUDITABLE)")
    print("=================================================")
    print(f"Tickers: {tickers}")
    print(f"Descarga cruda desde: {args.start}")
    print(f"Fecha final declarada: {args.end}")
    print(f"Fecha final efectiva (inclusiva): {end_inclusive}")
    print("-------------------------------------------------")

    # =================================================
    # Descarga de datos diarios (ajustados)
    # =================================================
    data = yf.download(
        tickers,
        start=args.start,
        end=end_inclusive,
        auto_adjust=True,
        progress=True
    )

    if data.empty:
        raise RuntimeError("Descarga vacía. Revisar conexión o tickers.")

    # Selección robusta de precios de cierre
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            prices_daily = data["Close"]
        else:
            prices_daily = data["Adj Close"]
    else:
        prices_daily = data.copy()

    prices_daily = (
        prices_daily
        .sort_index()
        .ffill()
        .dropna(how="all")
    )

    # =================================================
    # Conversión a frecuencia mensual (fin de mes)
    # =================================================
    prices_monthly = prices_daily.resample("ME").last()

    # --- chequeo crítico: último mes completo ---
    last_month = prices_monthly.index.max()
    if last_month != end_date + pd.offsets.MonthEnd(0):
        raise RuntimeError(
            f"Último mes mensual ({last_month.date()}) "
            f"NO coincide con fin de mes esperado ({end_date.date()})."
        )

    returns_monthly = prices_monthly.pct_change().dropna(how="all")

    # =================================================
    # Guardado de datasets
    # =================================================
    prices_daily.to_csv(DATA_DIR / "prices_daily.csv")
    prices_monthly.to_csv(DATA_DIR / "prices_monthly.csv")
    returns_monthly.to_csv(DATA_DIR / "returns_monthly.csv")

    # =================================================
    # Metadata de auditoría
    # =================================================
    meta = pd.DataFrame({
        "Ticker": prices_monthly.columns,
        "Inicio diario": [prices_daily[c].first_valid_index() for c in prices_monthly.columns],
        "Fin diario": [prices_daily[c].last_valid_index() for c in prices_monthly.columns],
        "Inicio mensual": [prices_monthly[c].first_valid_index() for c in prices_monthly.columns],
        "Fin mensual": [prices_monthly[c].last_valid_index() for c in prices_monthly.columns],
    })

    meta.to_csv(DATA_DIR / "dataset_metadata.csv", index=False)

    # =================================================
    # Resumen final
    # =================================================
    print("✔ Dataset construido correctamente.")
    print(f"✔ Observaciones mensuales: {len(prices_monthly)}")
    print(f"✔ Rango mensual: {prices_monthly.index.min().date()} – {prices_monthly.index.max().date()}")
    print("✔ Archivos generados:")
    print("   - prices_daily.csv")
    print("   - prices_monthly.csv")
    print("   - returns_monthly.csv")
    print("   - dataset_metadata.csv")
    print(f"✔ Salida en: {DATA_DIR}")

if __name__ == "__main__":
    main()
