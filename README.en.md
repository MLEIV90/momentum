# Momentum ETF Strategy — Backtesting & Risk Analysis

[🇪🇸 Español](README.md) · **🇬🇧 English**

Backtesting (2005–2025) of sector-momentum strategies on US ETFs, with performance
analysis, risk modeling (VaR, Expected Shortfall, GARCH) and portfolio optimization
(Markowitz). Degree thesis (UBA).

## Results

| Strategy | CAGR | Annual vol. | Sharpe (Lo) | Max Drawdown |
|---|---|---|---|---|
| SPY (benchmark) | 10.61% | 14.81% | 0.79 | −50.78% |
| LO_3 | 10.42% | 14.94% | **0.90** | −45.16% |
| LO_6 | 8.96% | 14.38% | 0.77 | −44.47% |
| LO_12 | 9.75% | 14.44% | 0.87 | **−39.07%** |

> **Key finding — risk-adjusted profile, not absolute return.** Sector-momentum
> strategies do not beat the S&P 500 on return (comparable or slightly lower CAGR),
> but they **substantially reduce maximum loss**: the benchmark drawdown reaches
> −50.78%, while LO_12 contains it at −39.07% (nearly 12 points less) at near-equivalent
> return. LO_3 also improves the autocorrelation-adjusted Sharpe (Lo, 2002): 0.90 vs.
> 0.79 for the market. Here, sector momentum acts as a **defensive rotation** tool.

Autocorrelation-adjusted Sharpe per Lo (2002); conditional VaR/ES via GARCH(1,1) with
Student-t innovations; VaR validation through the Kupiec test. Full detail in the thesis
document.

<!-- Once you upload the chart to the repo (e.g. to a figures/ folder), uncomment the line below:
![Equity curves — Sector momentum vs. SPY](figures/equity_curves_long_only.png)
-->

## Structure

- `main.py` — main pipeline: momentum backtest + base outputs
- `build_dataset.py` — downloads and builds the dataset, saves CSVs to `data/`
- `modules/` — risk, factor-model, GARCH and optimization functions
- `analysis/` — per-chapter runners (specific analyses + tables and figures)

## Documentation

- Full thesis: `docs/PPA022026 Tesina.docx`
- Defense slides: `docs/20260319_Momentum.pptx`

## Run

```bash
pip install -r requirements.txt

python build_dataset.py      # 1. builds the datasets (run first)
python main.py               # 2. backtest + base outputs
python analysis/run_risk_analysis.py
python analysis/run_capm_analysis.py
python analysis/run_garch_analysis.py
python analysis/run_optimization_analysis.py
```

Stack: Python · pandas · numpy · scipy · statsmodels · arch · matplotlib · yfinance
