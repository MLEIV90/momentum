# Estrategia de Momentum sobre ETFs — Backtesting y Análisis de Riesgo

**🇪🇸 Español** · [🇬🇧 English](README.en.md)

Backtesting (2005–2025) de estrategias de momentum sectorial sobre ETFs del mercado
estadounidense, con análisis de performance, modelización de riesgo (VaR, Expected
Shortfall, GARCH) y optimización de portafolios (Markowitz). Trabajo de tesis (UBA).

## Resultados

| Estrategia | CAGR | Vol. anual | Sharpe (Lo) | Max Drawdown |
|---|---|---|---|---|
| SPY (benchmark) | 10,61% | 14,81% | 0,79 | −50,78% |
| LO_3 | 10,42% | 14,94% | **0,90** | −45,16% |
| LO_6 | 8,96% | 14,38% | 0,77 | −44,47% |
| LO_12 | 9,75% | 14,44% | 0,87 | **−39,07%** |

> **Hallazgo principal — perfil riesgo-retorno, no retorno absoluto.** Las estrategias
> de momentum sectorial no superan al S&P 500 en retorno (CAGR comparable o levemente
> inferior), pero **reducen sustancialmente la pérdida máxima**: el drawdown del
> benchmark alcanza −50,78%, mientras que LO_12 lo contiene en −39,07% (casi 12 puntos
> menos) con un retorno casi equivalente. LO_3 mejora además el Sharpe ajustado por
> autocorrelación (Lo, 2002): 0,90 vs. 0,79 del mercado. El momentum sectorial opera
> aquí como herramienta de **rotación defensiva**.

Sharpe ajustado por autocorrelación según Lo (2002); VaR/ES condicionales vía GARCH(1,1)
con innovaciones t-Student; validación del VaR mediante test de Kupiec. Detalle completo
en el documento de tesis.

<!-- Cuando subas el gráfico al repo (p. ej. a una carpeta figures/), descomentá la línea siguiente:
![Curvas de capital — Momentum sectorial vs. SPY](figures/equity_curves_long_only.png)
-->

## Estructura

- `main.py` — pipeline principal: backtest de momentum + outputs base
- `build_dataset.py` — descarga y arma el dataset, guarda los CSV en `data/`
- `modules/` — funciones de riesgo, modelos de factores, GARCH y optimización
- `analysis/` — runners por capítulo (análisis específicos + tablas y gráficos)

## Documentación

- Tesis completa: `docs/PPA022026 Tesina.docx`
- Presentación de defensa: `docs/20260319_Momentum.pptx`

## Ejecución

```bash
pip install -r requirements.txt

python build_dataset.py      # 1. genera los datasets (requerido primero)
python main.py               # 2. backtest + resultados base
python analysis/run_risk_analysis.py
python analysis/run_capm_analysis.py
python analysis/run_garch_analysis.py
python analysis/run_optimization_analysis.py
```

Stack: Python · pandas · numpy · scipy · statsmodels · arch · matplotlib · yfinance
