"""
backtest.py
-----------
Análisis detallado de rentabilidad sobre el test set histórico.
Requiere que model.py se haya ejecutado previamente
(lee las señales con outcome_label de la BD).

Estrategias analizadas:
  - Sin filtro (baseline)
  - EV > 0
  - EV > 0.10  (medium+)
  - EV > 0.30  (high)
  - Score calibrado >= umbral óptimo guardado en metadata.json
  - Score calibrado >= 0.70 (alta confianza)

Métricas por estrategia:
  - Precisión (win rate real sobre señales emitidas)
  - P&L medio por unidad apostada (AvgPnL/u)
  - Capital total = sum(pnl_por_unidad) × TRADE_SIZE
  - Sharpe simplificado (dimensionless, calculado sobre valores unitarios)
  - Max drawdown en capital (× TRADE_SIZE)
  - Fractional Kelly recomendado (Kelly/4, dimensionless)

Output adicional:
  - Evolución del P&L acumulado en capital
  - Desglose mensual de la estrategia óptima
  - Análisis real por tier (EV-based)

── Modelo de P&L ──────────────────────────────────────────────
  label=1              → salida al take profit (2.5x)
  label=0 + rug        → recuperación del 15% del capital
  label=0 sin rug      → salida fija a NOLABEL_EXIT_MULTIPLIER (0.80x)
                          Representa un stop loss realista: no puedes
                          vender en el máximo histórico.

  simulate_pnl() devuelve P&L por unidad (no escalado por TRADE_SIZE).
  TRADE_SIZE solo se aplica en la columna Capital del output.

Uso:
    python src/backtest.py
"""

import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── Constantes de simulación ───────────────────────────────────
SLIPPAGE                = 0.03   # slippage entrada + salida
TAKE_PROFIT_X           = 2.5    # múltiplo objetivo (coincide con label=1)
RUG_RECOVERY            = 0.15   # fracción recuperada en rug
NOLABEL_EXIT_MULTIPLIER = 0.80   # salida fija para label=0 sin rug
                                  # (stop loss ~20% por debajo de entrada)
TRADE_SIZE              = 50     # capital base por señal (en unidades monetarias)

# Pérdida media estimada cuando la señal fracasa (para fórmula EV):
#   42% rugs     → pnl/u ≈ -0.88
#   58% no-rugs  → pnl/u = NOLABEL_EXIT_MULTIPLIER × (1-slip) - (1+slip) ≈ -0.25
#   Promedio: 0.42 × (-0.88) + 0.58 × (-0.25) ≈ -0.52
AVG_LOSS_ON_FAILURE = -0.52

# Break-even:  P_be = |loss| / (gain + |loss|)
# gain = 2.5 × (1-0.03) - (1+0.03) = 1.395
_GAIN_SUCCESS       = TAKE_PROFIT_X * (1 - SLIPPAGE) - (1 + SLIPPAGE)
PRECISION_BREAKEVEN = abs(AVG_LOSS_ON_FAILURE) / (
    _GAIN_SUCCESS + abs(AVG_LOSS_ON_FAILURE)
)


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_metadata() -> dict:
    path = "models/metadata.json"
    if not os.path.exists(path):
        print("  ⚠ metadata.json no encontrado. Ejecuta model.py primero.")
        return {}
    with open(path) as f:
        return json.load(f)


# ── Carga de señales ───────────────────────────────────────────

def load_signals(engine) -> pd.DataFrame:
    """
    Carga las señales del test set (con outcome_label conocido).
    Aplica drop_duplicates sobre coin_address para evitar que filas
    duplicadas en signals (sin UNIQUE constraint) inflen los resultados.
    """
    df = pd.read_sql("""
        SELECT
            s.coin_address,
            s.model_score,
            s.expected_multiple,
            s.ev_score,
            s.signal_tier,
            s.outcome_label      AS label,
            cp.max_multiple,
            cp.rug_detected,
            c.created_at
        FROM signals s
        INNER JOIN coin_prices cp ON s.coin_address = cp.coin_address
        INNER JOIN coins c        ON s.coin_address = c.coin_address
        WHERE s.outcome_label IS NOT NULL
        ORDER BY c.created_at ASC
    """, engine)

    df["rug_detected"] = df["rug_detected"].fillna(False).astype(bool)
    df["ev_score"]     = df["ev_score"].fillna(0.0).astype(float)
    df["model_score"]  = df["model_score"].fillna(0.0).astype(float)

    # Deduplicar por coin_address: la tabla signals no tiene UNIQUE
    # constraint en coin_address, por lo que múltiples ejecuciones de
    # model.py pueden generar filas duplicadas que inflan los resultados.
    n_before = len(df)
    df = df.drop_duplicates(subset=["coin_address"]).reset_index(drop=True)
    n_after  = len(df)
    if n_before > n_after:
        print(f"  ⚠ Eliminadas {n_before - n_after:,} filas duplicadas "
              f"(coin_address repetido en tabla signals).")

    print(f"  {len(df):,} señales del test set")
    print(f"  Positivos: {int(df['label'].sum())} ({df['label'].mean():.1%})")
    print(f"  Con EV > 0: {(df['ev_score'] > 0).sum():,}")
    return df


# ── Simulación de P&L ─────────────────────────────────────────

def simulate_pnl(
    label: int,
    max_multiple,        # mantenido en firma por compatibilidad, no se usa
    rug_detected: bool,
    slippage: float = SLIPPAGE,
) -> float:
    """
    P&L neto POR UNIDAD APOSTADA dado el outcome real.
    NO se multiplica por TRADE_SIZE aquí; eso ocurre solo en el output.

    Escenarios:
    - label=1              → exit al take profit (TAKE_PROFIT_X)
    - label=0 + rug        → exit a RUG_RECOVERY del precio de entrada
    - label=0 sin rug      → exit a NOLABEL_EXIT_MULTIPLIER del precio de entrada
                              (stop loss fijo, más realista que salir en el máximo)

    Ejemplo con SLIPPAGE=3%:
      label=1   → exit 2.5×0.97 - 1.03 = +1.395 por unidad
      rug       → exit 0.15×0.97 - 1.03 = -0.884 por unidad
      no-rug    → exit 0.80×0.97 - 1.03 = -0.254 por unidad
    """
    entry = 1.0 + slippage

    if label == 1:
        exit_val = TAKE_PROFIT_X * (1.0 - slippage)
    elif bool(rug_detected):
        exit_val = RUG_RECOVERY * (1.0 - slippage)
    else:
        exit_val = NOLABEL_EXIT_MULTIPLIER * (1.0 - slippage)

    return exit_val - entry   # per-unit, sin escalar por TRADE_SIZE


# ── Métricas ───────────────────────────────────────────────────

def compute_metrics(subset: pd.DataFrame, name: str) -> dict:
    """
    Calcula métricas de rentabilidad para un subconjunto de señales.
    Todos los valores de P&L se trabajan en unidades base (per-unit).
    TRADE_SIZE solo se aplica al mostrar el Capital total en pantalla.
    """
    if len(subset) == 0:
        return {"strategy": name, "n_signals": 0}

    pnls   = subset["pnl"].values   # per-unit
    labels = subset["label"].values
    n      = len(pnls)

    avg_pnl      = float(np.mean(pnls))          # por unidad
    total_pnl_pu = float(np.sum(pnls))           # suma por unidad
    total_capital = total_pnl_pu * TRADE_SIZE     # en capital real
    std_pnl      = float(np.std(pnls) + 1e-9)
    sharpe       = avg_pnl / std_pnl * np.sqrt(n)  # dimensionless
    win_rate     = float(np.mean(labels))
    n_wins       = int(np.sum(labels))

    # Max drawdown sobre P&L acumulado (per-unit) → convertido a capital
    cum         = np.cumsum(pnls)
    rolling_max = np.maximum.accumulate(cum)
    max_dd      = float(np.max(rolling_max - cum)) * TRADE_SIZE if len(cum) > 0 else 0.0

    # Fractional Kelly (Kelly/4) — dimensionless, calculado per-unit
    wins_pnl   = pnls[labels == 1]
    losses_pnl = pnls[labels == 0]
    if len(wins_pnl) > 0 and len(losses_pnl) > 0 and wins_pnl.mean() > 0:
        avg_win    = float(wins_pnl.mean())
        avg_loss   = float(abs(losses_pnl.mean()))
        kelly_full = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_frac = max(kelly_full / 4.0, 0.0)
    else:
        kelly_frac = 0.0

    return {
        "strategy":      name,
        "n_signals":     n,
        "n_wins":        n_wins,
        "win_rate":      win_rate,
        "avg_pnl":       avg_pnl,        # por unidad
        "total_capital": total_capital,   # en capital (× TRADE_SIZE)
        "sharpe":        sharpe,
        "max_dd":        max_dd,          # en capital (× TRADE_SIZE)
        "kelly_frac":    kelly_frac,
    }


# ── Output: tabla de resultados ────────────────────────────────

def print_results_table(metrics_list: list):
    """
    Columnas:
      AvgPnL/u  → P&L medio por unidad apostada (fracción, e.g. +0.20)
      Capital   → P&L total en capital = sum(pnl/u) × TRADE_SIZE
      MaxDD     → max drawdown en capital
    """
    size_label = f"×{TRADE_SIZE}"
    header = (f"  {'Estrategia':<33} {'N':>7} {'WinRate':>8} "
              f"{'AvgPnL/u':>9} {'Capital':>10} {'Sharpe':>7} "
              f"{'MaxDD':>8} {'Kelly/4':>8}")
    subhdr = (f"  {'':33} {'':7} {'':8} "
              f"{'':9} {size_label:>10} {'':7} "
              f"{size_label:>8} {'':8}")
    print(header)
    print(subhdr)
    print("  " + "─" * 100)

    for m in metrics_list:
        if m.get("n_signals", 0) == 0:
            print(f"  {m['strategy']:<33}  (sin señales)")
            continue
        ok = "✓" if m["avg_pnl"] > 0 else "✗"
        print(
            f"  {ok}{m['strategy']:<32} {m['n_signals']:>7,} "
            f"{m['win_rate']:>8.1%} {m['avg_pnl']:>+9.4f} "
            f"{m['total_capital']:>+10.2f} {m['sharpe']:>7.2f} "
            f"{m['max_dd']:>8.2f} {m['kelly_frac']:>8.2%}"
        )
    print("  ✓ = P&L/u positivo  ✗ = P&L/u negativo")


# ── Output: evolución P&L ──────────────────────────────────────

def print_pnl_evolution(df: pd.DataFrame, strategies: dict):
    """
    Curva de P&L acumulado en capital (per-unit × TRADE_SIZE).
    Separado en print() propio para evitar artefactos de terminal con \\n.
    """
    print()
    print(f"  Evolución capital acumulado (inicio=0, ×{TRADE_SIZE}):")
    print(f"  {'Estrategia':<33} {'Final':>9} {'Pico':>9} {'Valle':>9}")
    print("  " + "─" * 65)
    for name, mask in strategies.items():
        subset = df[mask].sort_values("created_at")
        if len(subset) < 2:
            continue
        cum = np.cumsum(subset["pnl"].values) * TRADE_SIZE  # en capital
        print(f"  {name:<33} {cum[-1]:>+9.2f} {cum.max():>+9.2f} {cum.min():>+9.2f}")


# ── Output: desglose mensual ───────────────────────────────────

def print_monthly_breakdown(df: pd.DataFrame, mask, strategy_name: str):
    """
    Desglose mensual de la estrategia indicada.
    Usa strftime en lugar de to_period para evitar duplicados
    en el groupby con índices Period en algunas versiones de pandas.
    Capital = sum(pnl/u) × TRADE_SIZE.
    """
    subset = df[mask].copy()
    if len(subset) < 5:
        return

    dt = pd.to_datetime(subset["created_at"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(None)
    subset["month"] = dt.dt.strftime("%Y-%m")

    monthly = (
        subset.groupby("month", sort=True)
        .agg(
            n_signals    = ("pnl", "count"),
            n_wins       = ("label", "sum"),
            total_pnl_pu = ("pnl", "sum"),    # per-unit
            avg_pnl_pu   = ("pnl", "mean"),   # per-unit
        )
        .reset_index()
        .drop_duplicates(subset=["month"])
    )
    monthly["capital"] = monthly["total_pnl_pu"] * TRADE_SIZE

    print()
    print(f"  Desglose mensual — {strategy_name}:")
    print(f"  {'Mes':<10} {'Señales':>8} {'Wins':>6} "
          f"{'Capital':>10} {'AvgPnL/u':>10}")
    print("  " + "─" * 52)
    for _, row in monthly.iterrows():
        ok = "✓" if row.avg_pnl_pu > 0 else "✗"
        print(f"  {ok}{str(row.month):<9} {int(row.n_signals):>8} "
              f"{int(row.n_wins):>6} {row.capital:>+10.2f} "
              f"{row.avg_pnl_pu:>+10.4f}")


# ── Output: análisis de tiers ──────────────────────────────────

def print_tier_analysis(df: pd.DataFrame):
    """
    Precisión y P&L real por tier asignado por el modelo.
    Capital = sum(pnl/u) × TRADE_SIZE.
    """
    print()
    print("  Análisis por tier (EV-based):")
    print(f"  {'Tier':<12} {'N':>6} {'WinRate':>8} "
          f"{'AvgPnL/u':>10} {'Capital':>10}")
    print("  " + "─" * 52)

    for tier in ["high", "medium", "low", None]:
        if tier is None:
            subset = df[df["signal_tier"].isna()]
            label  = "sin_señal"
        else:
            subset = df[df["signal_tier"] == tier]
            label  = tier

        if len(subset) == 0:
            continue

        win_rate  = float(subset["label"].mean())
        avg_pnl   = float(subset["pnl"].mean())
        capital   = float(subset["pnl"].sum()) * TRADE_SIZE
        ok = "✓" if avg_pnl > 0 else "✗"

        print(f"  {ok}{label:<11} {len(subset):>6,} {win_rate:>8.1%} "
              f"{avg_pnl:>+10.4f} {capital:>+10.2f}")


# ── Pipeline principal ─────────────────────────────────────────

def run_backtest(df: pd.DataFrame, metadata: dict) -> list:

    # Calcular P&L per-unit por trade (sin TRADE_SIZE)
    df = df.copy()
    df["pnl"] = [
        simulate_pnl(row.label, row.max_multiple, row.rug_detected)
        for row in df.itertuples()
    ]

    opt_t = float(metadata.get("optimal_threshold", 0.50))

    strategies = {
        "Sin filtro (baseline)":             df["model_score"] >= 0.0,
        "EV > 0":                            df["ev_score"] > 0,
        "EV > 0.10 (medium+)":              df["ev_score"] > 0.10,
        "EV > 0.30 (high)":                 df["ev_score"] > 0.30,
        f"Score ≥ {opt_t:.2f} (óptimo)":    df["model_score"] >= opt_t,
        "Score ≥ 0.70 (alta conf.)":         df["model_score"] >= 0.70,
    }

    metrics_list = [
        compute_metrics(df[mask], name)
        for name, mask in strategies.items()
    ]

    print()
    print("=" * 105)
    print("RESULTADOS BACKTESTING")
    print(
        f"  Slippage: {SLIPPAGE:.0%}  |  Take profit: {TAKE_PROFIT_X}x  |  "
        f"Exit label=0 no-rug: {NOLABEL_EXIT_MULTIPLIER:.0%}  |  "
        f"Rug recovery: {RUG_RECOVERY:.0%}  |  Trade size: {TRADE_SIZE}"
    )
    print(f"  Break-even precision: {PRECISION_BREAKEVEN:.1%}  |  "
          f"AvgPnL/u = P&L por unidad apostada  |  "
          f"Capital = total × {TRADE_SIZE}")
    print("=" * 105)

    print_results_table(metrics_list)
    print_pnl_evolution(df, strategies)

    opt_name = f"Score ≥ {opt_t:.2f} (óptimo)"
    if opt_name in strategies:
        print_monthly_breakdown(df, strategies[opt_name], opt_name)

    print_tier_analysis(df)

    print()
    print("  ⚠  Supuestos del backtesting:")
    print(f"     · label=0 no rug → salida fija al {NOLABEL_EXIT_MULTIPLIER:.0%} del precio de entrada.")
    print("       Representa un stop loss realista: no es posible vender en el máximo.")
    print(f"     · label=0 rug    → recuperación del {RUG_RECOVERY:.0%} del capital.")
    print(f"     · Slippage {SLIPPAGE:.0%} aplicado en entrada y salida.")
    print("     · Sin costes de gas ni impacto de liquidez.")
    print(f"     · Trade size: {TRADE_SIZE} unidades de capital por señal.")

    return metrics_list


def main():
    print("=" * 60)
    print("BACKTESTING DETALLADO")
    print(f"  Modelo P&L: exit label=0 no-rug = {NOLABEL_EXIT_MULTIPLIER:.0%} entrada")
    print(f"  Break-even precision: {PRECISION_BREAKEVEN:.1%}")
    print(f"  Trade size: {TRADE_SIZE} unidades por señal")
    print("=" * 60)

    engine   = get_engine()
    metadata = load_metadata()

    if metadata:
        print(f"\n  Modelo entrenado:       {metadata.get('trained_at', '—')}")
        print(f"  Umbral óptimo guardado: {metadata.get('optimal_threshold', '—')}")

    print("\n[1/2] Cargando señales del test set...")
    df = load_signals(engine)

    if df.empty:
        print("Sin señales. Ejecuta model.py primero.")
        return

    print("\n[2/2] Ejecutando backtesting...")
    run_backtest(df, metadata)

    print()
    print("=" * 60)
    print("COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()