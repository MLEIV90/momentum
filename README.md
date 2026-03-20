# tesis_momentum

## Estructura
- `main.py`: pipeline principal (backtest momentum + outputs base)
- `build_dataset.py`: descarga/arma dataset y guarda CSVs en `data/`
- `modules/`: funciones (riesgo, factores, GARCH, optimización)
- `analysis/`: runners por capítulo (ejecutan análisis específicos y exportan tablas/gráficos)

## Instalación
python -m pip install -r requirements.txt

## Ejecución recomendada
1) python build_dataset.py
2) python main.py
3) python analysis/run_risk_analysis.py
4) python analysis/run_capm_analysis.py
5) python analysis/run_garch_analysis.py
6) python analysis/run_optimization_analysis.py
