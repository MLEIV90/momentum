# Momentum ETF Strategy – Backtesting & Risk Analysis

Proyecto de análisis financiero enfocado en el desarrollo y evaluación de estrategias de inversión basadas en momentum sectorial sobre ETFs del mercado estadounidense.

Incluye backtesting histórico (2005–2025), análisis de performance y modelización de riesgo utilizando métricas como VaR, Expected Shortfall y modelos GARCH.

El objetivo es evaluar la viabilidad de estrategias cuantitativas basadas en momentum y analizar su perfil riesgo-retorno en distintos escenarios de mercado.

## Funcionalidades principales

- Backtesting de estrategias de momentum sobre ETFs
- Evaluación de performance (Sharpe ratio, drawdown, retornos acumulados)
- Análisis de riesgo (VaR / Expected Shortfall)
- Modelización de volatilidad mediante GARCH
- Optimización de portafolios (Markowitz)

## Tecnologías utilizadas

Python (pandas, numpy, matplotlib, scipy, statsmodels, arch, yfinance)

## Documentación

- Tesis completa: docs/PPA022026 Tesina.docx
- Presentación de defensa: docs/20260319_Momentum.pptx
 
## Estructura
- `main.py`: pipeline principal (backtest momentum + outputs base)
- `build_dataset.py`: descarga/arma dataset y guarda CSVs en `data/`
- `modules/`: funciones (riesgo, factores, GARCH, optimización)
- `analysis/`: runners por capítulo (ejecutan análisis específicos y exportan tablas/gráficos)

## Instalación

```bash
python -m pip install -r requirements.txt
```
## Ejecución recomendada
- python build_dataset.py
- python main.py
- python analysis/run_risk_analysis.py
- python analysis/run_capm_analysis.py
- python analysis/run_garch_analysis.py
- python analysis/run_optimization_analysis.py
