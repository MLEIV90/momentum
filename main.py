from __future__ import annotations

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DATA_DIR,
    FIG_DIR,
    BACKTEST_START,
    BENCHMARK,
    LOOKBACK_WINDOWS_DEFAULT,
    TOP_PCT_DEFAULT,
    TRANSACTION_COST_BPS_DEFAULT,
)

RISK_FREE_TICKER = "^IRX"  # 3-Month T-Bill yield index (annualized %, level)

# ======================================================
# CLI
# ======================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest Momentum Sectorial LONG-ONLY (auditado, sin look-ahead, drift, warm-up correcto y Rf = ^IRX)."
    )
    p.add_argument("--backtest_start", default=BACKTEST_START)
    return p.parse_args()

# ======================================================
# Utilidades
# ======================================================

def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()

def _serialize(lst: list[str]) -> str:
    return ",".join(lst)

# ======================================================
# MAIN
# ======================================================

def main() -> None:
    args = _parse_args()
    backtest_start = pd.to_datetime(args.backtest_start)

    # Parámetros estructurales
    lookbacks = list(LOOKBACK_WINDOWS_DEFAULT)
    top_pct = float(TOP_PCT_DEFAULT)
    tcost_rate = TRANSACTION_COST_BPS_DEFAULT / 10_000.0

    # ======================================================
    # Datos (NO se cortan todavía: warm-up correcto)
    # ======================================================
    prices = pd.read_csv(DATA_DIR / "prices_monthly.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(DATA_DIR / "returns_monthly.csv", index_col=0, parse_dates=True)

    # Universo invertible: excluye benchmark y la tasa libre de riesgo
    universe = [c for c in prices.columns if c not in {BENCHMARK, RISK_FREE_TICKER}]

    # ======================================================
    # Tasa libre de riesgo (^IRX): convertir a tasa mensual
    # ======================================================
    if RISK_FREE_TICKER not in prices.columns:
        raise RuntimeError(
            f"No se encontró {RISK_FREE_TICKER} en prices_monthly.csv. "
            "Asegurate de incluirlo en TICKERS_DEFAULT y reconstruir el dataset."
        )

    # ^IRX suele venir como nivel de yield anualizado en %, p.ej. 5.25
    irx_level = prices[RISK_FREE_TICKER].loc[backtest_start:].dropna()

    # Convertimos a tasa mensual geom
    rf_monthly = (1.0 + irx_level / 100.0) ** (1.0 / 12.0) - 1.0
    rf_monthly.name = "rf_monthly"

    # ======================================================
    # Benchmark
    # ======================================================
    ret_bh = returns.loc[backtest_start:, BENCHMARK].dropna()
    results: dict[str, pd.Series] = {"SPY": ret_bh}

    transactions, weights = [], []

    # ======================================================
    # Estrategias Momentum (calcular retornos y logs)
    # ======================================================
    for L in lookbacks:
        # === Señal de momentum  : retorno acumulado compuesto  ===
        # Momentum_{i,t}^{(L)} = Π_{k=1..L} (1 + r_{i,t-k}) - 1
        mom_full = (
            (1.0 + returns[universe])
            .rolling(window=L, min_periods=L)
            .apply(np.prod, raw=True)
            - 1.0
        ).shift(1)

        mom = mom_full.loc[backtest_start:]
        ret = returns.loc[backtest_start:]

        prev_w = pd.Series(0.0, index=universe)
        has_prev = False   # <- NUEVO: todavía no existe cartera previa invertida
        rets, dates = [], []


        for date in mom.index:
            if date not in ret.index:
                continue

            signal = mom.loc[date, universe].dropna()
            r_t = ret.loc[date, universe].dropna()
            common = signal.index.intersection(r_t.index)

            if len(common) < 2:
                continue

            # === Drift de pesos ANTES de rebalancear (usa retorno t-1) ===
            w_drift = prev_w.copy()

            # === Selección de ganadores (ranking cross-sectional) ===
            ranking = signal.loc[common].sort_values(ascending=False)
            winners = list(ranking.index[:max(1, int(len(ranking) * top_pct))])

            w_new = pd.Series(0.0, index=universe)
            w_new.loc[winners] = 1.0 / len(winners)

            # === Turnover + costo transaccional (one-way) ===
            if not has_prev:
                turnover = 0.0
                tc = 0.0
            else:
                delta = (w_new - w_drift).abs()
                turnover = 0.5 * delta.sum()
                tc = tcost_rate * turnover


            # === Retorno del período t ===
            gross = r_t.loc[winners].mean()
            net = gross - tc

            rets.append(net)
            dates.append(date)

            # === Log de transacciones ===
            transactions.append({
                "Date": date,
                "Strategy": f"LO_{L}",
                "Lookback": L,
                "Winners": _serialize(winners),
                "Turnover": turnover,
                "TCostRate": tcost_rate,
                "TransactionCost": tc,
                "GrossReturn": gross,
                "NetReturn": net,
            })

            # === Log de trades por sector (one-way) ===
            if has_prev:
                for s in universe:
                    d_abs = abs(w_new[s] - w_drift[s])
                    if d_abs > 0:
                        trade_one_way = 0.5 * d_abs
                        weights.append({
                            "Date": date,
                            "Strategy": f"LO_{L}",
                            "Lookback": L,
                            "Sector": s,
                            "WeightDrift": w_drift[s],
                            "WeightNew": w_new[s],
                            "DeltaAbs": d_abs,
                            "TradeOneWay": trade_one_way,
                            "TradeCost": trade_one_way * tcost_rate,
                        })


            # === Actualización de pesos para el próximo período (drift correcto) ===
            port_ret = (w_new.loc[common] * r_t.loc[common]).sum()
            prev_w = w_new * (1.0 + r_t)
            prev_w /= (1.0 + port_ret)
            has_prev = True


                
        ret_s = pd.Series(rets, index=pd.to_datetime(dates)).sort_index()
        results[f"LO_{L}"] = ret_s


    # ======================================================
    # Outputs (retornos)
    # ======================================================
    df_returns = pd.DataFrame(results).sort_index()
    tx = pd.DataFrame(transactions)
    wlog = pd.DataFrame(weights)

    # Período común para comparabilidad (y para la tabla 4.1)
    df_common = df_returns.dropna()
    T = len(df_common) / 12

    # ======================================================
    # Tabla 4.1: retorno + riesgo básico + MDD + Sharpe con Rf (^IRX)
    # ======================================================
    # Alinear rf a las fechas del período común
    rf_common = rf_monthly.reindex(df_common.index).ffill()

    # Retorno mensual promedio (simple)
    mean_monthly = df_common.mean()

    # Volatilidad anualizada (retornos totales, medida de riesgo)
    ann_vol = df_common.std() * (12 ** 0.5)
 
    # Sharpe con Rf variable: exceso mensual, vol de excesos en denominador
    excess = df_common.sub(rf_common, axis=0)
    mean_excess_monthly = excess.mean()
    ann_vol_excess = excess.std() * (12 ** 0.5)
    sharpe = (mean_excess_monthly * 12) / ann_vol_excess
  


    # Max Drawdown (%) sobre el período común
    cum_ret = (1.0 + df_common).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1.0
    max_dd = drawdown.min() * 100  # negativo

    table_4_1 = pd.DataFrame({
        "Retorno mensual promedio (%)": mean_monthly * 100,
        "CAGR (%)": ((1 + df_common).prod() ** (1 / T) - 1) * 100,
        "Volatilidad anual (%)": ann_vol * 100,
        "Sharpe Ratio (Rf=^IRX)": sharpe,
        "Max Drawdown (%)": max_dd,
    })

    # Resumen de costos
    summary_costs = (
        tx.groupby("Strategy")
        .agg(
            TurnoverProm=("Turnover", "mean"),
            CostoMensualProm=("TransactionCost", "mean"),
        )
        .reset_index()
    )
    summary_costs["CostoAnualAprox"] = 12 * summary_costs["CostoMensualProm"]

    # ======================================================
    # Figura: Equity curves correctas (capital acumulado) en período común
    # ======================================================
    plt.figure(figsize=(11, 6))
    for col in df_common.columns:
        eq = equity_curve(df_common[col])
        eq = eq / eq.iloc[0]  # base = 1 común

        if col == "SPY":
            plt.plot(eq, label="SPY (Buy-and-Hold)", color="black", linewidth=2.2)
        else:
            plt.plot(eq, label=col)

    plt.title("Curvas de capital (base = 1)\nMomentum Sectorial Long-Only y Benchmark")
    plt.xlabel("Fecha")
    plt.ylabel("Valor acumulado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "equity_curves_long_only.png", dpi=300)
    plt.close()

    # ======================================================
    # Exportar Excel (mantener TODAS las pestañas)
    # ======================================================
    out = DATA_DIR / "momentum_backtest_audit.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_returns.to_excel(writer, sheet_name="returns_net")
        table_4_1.to_excel(writer, sheet_name="Table_4_1_Rentabilidad")
        tx.to_excel(writer, sheet_name="transactions", index=False)
        wlog.to_excel(writer, sheet_name="weights", index=False)
        summary_costs.to_excel(writer, sheet_name="summary_costs", index=False)

    print("✔ Backtest finalizado (Rf=^IRX, Tabla 4.1 completa, Excel con todas las pestañas).")
    print("✔ Gráfico generado: equity_curves_long_only.png")
    print("✔ Archivo generado:", out.name)

if __name__ == "__main__":
    main()
