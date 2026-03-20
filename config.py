"""
config.py

Configuración centralizada de paths y parámetros generales.

Objetivo:
- Eliminar paths hardcodeados y hacer el proyecto portable.
- Fijar explícitamente el período muestral para reproducibilidad.
- Centralizar supuestos estructurales del backtest (costos, ranking, rebalanceo)
  para evitar discrepancias entre ejecuciones y outputs.
"""

from __future__ import annotations

from pathlib import Path

# =====================================================
# DIRECTORIOS DEL PROYECTO
# =====================================================

# Directorio raíz del proyecto (carpeta donde está este archivo)
BASE_DIR: Path = Path(__file__).resolve().parent

# Carpetas de datos y salidas
DATA_DIR: Path = BASE_DIR / "data"
FIG_DIR: Path = BASE_DIR / "graficos"
DOWNLOADS_DIR: Path = DATA_DIR / "descargas"

# Crear carpetas si no existen
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# PARÁMETROS TEMPORALES DEL ANÁLISIS
# =====================================================

# Inicio de descarga de datos crudos (historia larga para señales)
RAW_START_DATE: str = "2000-01-01"

# Inicio del backtest (primer mes considerado para estrategias)
BACKTEST_START: str = "2005-01-01"

# Fecha final fija del dataset (mes completo)
# IMPORTANTE: el pipeline debe asegurar que el último mes esté completo (sin meses parciales).
END_DATE: str = "2025-12-31"

# =====================================================
# BENCHMARK Y UNIVERSO DE ACTIVOS
# =====================================================

# Benchmark de mercado
BENCHMARK: str = "SPY"

# ETFs sectoriales SPDR (9 sectores) + benchmark + LR
TICKERS_DEFAULT: list[str] = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLU",  # Utilities
    "SPY",  # S&P 500 (benchmark)
    "^IRX",  # 3-Month Treasury Bill (^IRX)
]

# =====================================================
# SUPUESTOS ESTRUCTURALES DEL BACKTEST (TESIS)
# =====================================================

# Ventanas de formación (meses) – baseline de la tesis
LOOKBACK_WINDOWS_DEFAULT: list[int] = [3, 6, 12]

# Porcentaje del universo seleccionado como "ganadores"
# Con 9 sectores, 0.34 ~ 3 sectores (tercio superior aproximado).
TOP_PCT_DEFAULT: float = 0.34

# Costos transaccionales proporcionales (baseline tesis)
# 10 bps round-trip (5 bps compra + 5 bps venta).
# Se aplica sobre turnover one-way: tc = 10 bps × one_way_turnover.
TRANSACTION_COST_BPS_DEFAULT: float = 10.0  # 0.10% round-trip

# Convención de turnover:
# - "one_way": 0.5 * sum_i |w_{i,t} - w_{i,t-1}|  (estándar académico)
# - "two_way": sum_i |w_{i,t} - w_{i,t-1}|
TURNOVER_CONVENTION: str = "one_way"

# =====================================================
# PARÁMETROS DE RIESGO (baseline para reportes)
# =====================================================

VAR_ALPHA_DEFAULT: float = 0.95
VAR_WINDOW_DEFAULT: int = 60  # meses para VaR rolling (Kupiec)
