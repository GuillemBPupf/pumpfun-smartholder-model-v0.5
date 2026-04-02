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
  - P&L medio por señal y P&L total acumulado
  - Sharpe simplificado
  - Max drawdown sobre P&L acumulado
  - Fractional Kelly recomendado (Kelly/4)

Output adicional:
  - Evolución del P&L acumulado (inicio → fin → pico → valle)
  - Desglose mensual de la estrategia óptima
  - Notas sobre supuestos del backtesting

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

# ── Constantes (deben coincidir con model.py) ──────────────────
SLIPPAGE        = 0.03
TAKE_PROFIT_X   = 2.5
STOP_LOSS_FRAC  = 0.50
RUG_RECOVERY    = 0.15
AVG_LOSS_ON_FAILURE = -0.55

_GAIN_SUCCESS       = TAKE_PROFIT_X * (1 - SLIPPAGE) - (1 + SLIPPAGE)
PRECISION_BREAKEVEN = abs(AVG_LOSS_ON_FAILURE) / (_GAIN_SUCCESS + abs(AVG_LOSS_ON_FAILURE))


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
    Carga las señales del test set (las que tienen outcome_label conocido)
    junto con los datos de precio necesarios para la simulación.
    """
    df = pd.read_sql("""
        SELECT
            s.coin_address,
            s.model_score,
            s.expected_multiple,
            s.ev_score,
            s.signal_tier,
            s.outcome_label          AS label,
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

    print(f"  {len(df):,} señales del test set")
    print(f"  Positivos: {int(df['label'].sum())} ({df['label'].mean():.1%})")
    print(f"  Con EV > 0: {(df['ev_score'] > 0).sum():,}")
    return df


# ── Simulación de P&L ─────────────────────────────────────────

def simulate_pnl(
    label: int,
    max_multiple,
    rug_detected: bool,
    slippage: float = SLIPPAGE,
) -> float:
    """
    P&L neto por unidad apostada dado el outcome real.

    Supuestos:
    - label=1        → exit al take profit (2.5x)
    - label=0 + rug  → exit al 15% del entry (pérdida casi total)
    - label=0 no-rug → exit al max(min(max_multiple, 2.5x), 0.5x)
      ⚠ Optimista: asume salida cerca del máximo histórico.
    """
    entry = 1.0 + slippage

    if label == 1:
        exit_val = TAKE_PROFIT_X * (1.0 - slippage)

    elif bool(rug_detected):
        exit_val = RUG_RECOVERY * (1.0 - slippage)

    else:
        mm = (
            float(max_multiple)
            if max_multiple is not None and not np.isnan(float(max_multiple))
            else 1.0
        )
        effective = max(min(mm, TAKE_PROFIT_X), STOP_LOSS_FRAC)
        exit_val  = effective * (1.0 - slippage)

    return exit_val - entry


# ── Métricas ───────────────────────────────────────────────────

def compute_metrics(subset: pd.DataFrame, name: str) -> dict:
    """Calcula métricas completas de rentabilidad para un subconjunto."""
    if len(subset) == 0:
        return {"strategy": name, "n_signals": 0}

    pnls   = subset["pnl"].values
    labels = subset["label"].values
    n      = len(pnls)

    avg_pnl   = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))
    std_pnl   = float(np.std(pnls) + 1e-9)
    sharpe    = avg_pnl / std_pnl * np.sqrt(n)
    win_rate  = float(np.mean(labels))
    n_wins    = int(np.sum(labels))

    # Max drawdown sobre P&L acumulado cronológico
    cum         = np.cumsum(pnls)
    rolling_max = np.maximum.accumulate(cum)
    max_dd      = float(np.max(rolling_max - cum)) if len(cum) > 0 else 0.0

    # Fractional Kelly (Kelly/4 para seguridad)
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
        "strategy":   name,
        "n_signals":  n,
        "n_wins":     n_wins,
        "win_rate":   win_rate,
        "avg_pnl":    avg_pnl,
        "total_pnl":  total_pnl,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "kelly_frac": kelly_frac,
    }


# ── Output: tabla de resultados ────────────────────────────────

def print_results_table(metrics_list: list):
    header = (f"  {'Estrategia':<33} {'N':>7} {'WinRate':>8} "
              f"{'AvgPnL':>9} {'Total':>8} {'Sharpe':>7} "
              f"{'MaxDD':>7} {'Kelly/4':>8}")
    print(header)
    print("  " + "─" * 95)

    for m in metrics_list:
        if m.get("n_signals", 0) == 0:
            print(f"  {m['strategy']:<33}  (sin señales)")
            continue
        ok = "✓" if m["avg_pnl"] > 0 else " "
        print(
            f"  {ok}{m['strategy']:<32} {m['n_signals']:>7,} "
            f"{m['win_rate']:>8.1%} {m['avg_pnl']:>+9.4f} "
            f"{m['total_pnl']:>+8.2f} {m['sharpe']:>7.2f} "
            f"{m['max_dd']:>7.2f} {m['kelly_frac']:>8.2%}"
        )
    print("  ✓ = P&L positivo por señal")


# ── Output: evolución P&L ──────────────────────────────────────

def print_pnl_evolution(df: pd.DataFrame, strategies: dict):
    """Curva de P&L acumulado resumida por estrategia."""
    print("\n  Evolución P&L acumulado (inicio=0):")
    print(f"  {'Estrategia':<33} {'Final':>8} {'Pico':>8} {'Valle':>8}")
    print("  " + "─" * 65)
    for name, mask in strategies.items():
        subset = df[mask].sort_values("created_at")
        if len(subset) < 2:
            continue
        cum = np.cumsum(subset["pnl"].values)
        print(f"  {name:<33} {cum[-1]:>+8.2f} {cum.max():>+8.2f} {cum.min():>+8.2f}")


# ── Output: desglose mensual ───────────────────────────────────

def print_monthly_breakdown(df: pd.DataFrame, mask, strategy_name: str):
    """Desglose mensual de la estrategia indicada."""
    subset = df[mask].copy()
    if len(subset) < 5:
        return

    # Convertir a período mensual (maneja timestamps con timezone)
    dt = pd.to_datetime(subset["created_at"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(None)
    subset["month"] = dt.dt.to_period("M")

    monthly = (
        subset.groupby("month")
        .agg(
            n_signals = ("pnl", "count"),
            n_wins    = ("label", "sum"),
            total_pnl = ("pnl", "sum"),
            avg_pnl   = ("pnl", "mean"),
        )
        .reset_index()
    )

    print(f"\n  Desglose mensual — {strategy_name}:")
    print(f"  {'Mes':<10} {'Señales':>8} {'Wins':>6} "
          f"{'TotalPnL':>10} {'AvgPnL':>9}")
    print("  " + "─" * 50)
    for _, row in monthly.iterrows():
        ok = "✓" if row.avg_pnl > 0 else " "
        print(f"  {ok}{str(row.month):<9} {int(row.n_signals):>8} "
              f"{int(row.n_wins):>6} {row.total_pnl:>+10.2f} "
              f"{row.avg_pnl:>+9.4f}")


# ── Output: análisis de tiers ──────────────────────────────────

def print_tier_analysis(df: pd.DataFrame):
    """Precisión y P&L real por tier asignado por el modelo."""
    tiers = ["high", "medium", "low", None]
    print("\n  Análisis por tier (EV-based):")
    print(f"  {'Tier':<12} {'N':>6} {'WinRate':>8} {'AvgPnL':>9} {'TotalPnL':>10}")
    print("  " + "─" * 50)

    for tier in tiers:
        if tier is None:
            mask = df["signal_tier"].isna()
            label = "sin_señal"
        else:
            mask  = df["signal_tier"] == tier
            label = tier

        subset = df[mask]
        if len(subset) == 0:
            continue

        win_rate  = subset["label"].mean()
        avg_pnl   = subset["pnl"].mean()
        total_pnl = subset["pnl"].sum()
        ok = "✓" if avg_pnl > 0 else " "

        print(f"  {ok}{label:<11} {len(subset):>6,} {win_rate:>8.1%} "
              f"{avg_pnl:>+9.4f} {total_pnl:>+10.2f}")


# ── Pipeline principal ─────────────────────────────────────────

def run_backtest(df: pd.DataFrame, metadata: dict) -> list:

    # Calcular P&L real por trade
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

    # ── Tabla principal ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTADOS BACKTESTING")
    print(
        f"  Slippage: {SLIPPAGE:.0%}  |  Take profit: {TAKE_PROFIT_X}x  |  "
        f"Stop loss floor: {STOP_LOSS_FRAC}x  |  Rug recovery: {RUG_RECOVERY}x"
    )
    print(f"  Break-even precision: {PRECISION_BREAKEVEN:.1%}")
    print("=" * 100)

    print_results_table(metrics_list)

    # ── Evolución P&L ─────────────────────────────────────────
    print_pnl_evolution(df, strategies)

    # ── Desglose mensual para la estrategia óptima ────────────
    opt_name = f"Score ≥ {opt_t:.2f} (óptimo)"
    if opt_name in strategies:
        print_monthly_breakdown(df, strategies[opt_name], opt_name)

    # ── Análisis por tier ─────────────────────────────────────
    print_tier_analysis(df)

    # ── Notas sobre supuestos ─────────────────────────────────
    print("\n  ⚠  Supuestos del backtesting:")
    print("     · label=0 no rug → salida asumida al máximo histórico (OPTIMISTA).")
    print("       En producción real saldrás antes o después del pico.")
    print("     · label=0 rug   → recuperación del 15% del capital.")
    print("     · Slippage 3% aplicado en entrada y salida.")
    print("     · Sin costes de gas ni impacto de liquidez.")
    print("     · Cada señal = 1 unidad de capital (sin sizing).")

    return metrics_list


def main():
    print("=" * 60)
    print("BACKTESTING DETALLADO")
    print("=" * 60)

    engine   = get_engine()
    metadata = load_metadata()

    if metadata:
        print(f"\n  Modelo entrenado: {metadata.get('trained_at', '—')}")
        print(f"  Umbral óptimo guardado: {metadata.get('optimal_threshold', '—')}")
        print(f"  Break-even precision:   {metadata.get('precision_breakeven', '—')}")

    print("\n[1/2] Cargando señales del test set...")
    df = load_signals(engine)

    if df.empty:
        print("Sin señales. Ejecuta model.py primero.")
        return

    print("\n[2/2] Ejecutando backtesting...")
    run_backtest(df, metadata)

    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()