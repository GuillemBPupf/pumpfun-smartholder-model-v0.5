"""
wallet_scoring.py
-----------------
Calcula el performance_score de cada wallet usando SOLO su historial
en el conjunto de entrenamiento (train_coins), evitando data leakage.

Fórmula del score:

  Paso 1 — Bayesian win rate:
    bayesian_wr = (wins + PRIOR * global_wr) / (N + PRIOR)
    Estabiliza wallets con pocas apariciones acercándolas al global.

  Paso 2 — Lift sobre baseline:
    lift = bayesian_win_rate / global_win_rate
    Una wallet promedio tiene lift = 1.0.
    lift >= MAX_LIFT → win_component = 1.0

  Paso 3 — Conviction ratio:
    ratio entre el SOL medio apostado en coins ganadoras vs perdedoras.
    > 1.0 → pone más dinero cuando acierta → señal de conocimiento real.

  Paso 4 — Score final:
    win_component        = clip(lift / MAX_LIFT, 0, 1)
    roi_component        = clip(avg_roi_capped / 5.0, 0, 1)
    entry_component      = clip(1 - avg_entry_seconds / 180, 0, 1)
    conviction_component = clip((conviction_ratio - 1) / 3, 0, 1)

    base_score = (win_component        * 0.50
               + roi_component         * 0.20
               + entry_component       * 0.15
               + conviction_component  * 0.15)
    penalización = negative_rate * PENALIZATION
    score = clip(base_score - penalización, 0, 1)

Parámetros:
    ROI_CAP         = 10.0   → cap por coin antes de media
    PRIOR_STRENGTH  = 5      → pseudo-observaciones Bayesian
    MAX_LIFT        = 8.0    → lift en que win_component = 1.0
                               Con baseline 5%, win_rate ~40% ya satura.
    PENALIZATION    = 0.20   → peso de la penalización por rugs
    MIN_APPEARANCES = 5      → mínimo para score_reliable = TRUE

Uso:
    python src/wallet_scoring.py

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from splitter import get_train_test_coins

load_dotenv()

ROI_CAP         = 10.0
MIN_APPEARANCES = 5
PRIOR_STRENGTH  = 5
MAX_LIFT        = 8.0
PENALIZATION    = 0.20


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_data(engine, train_coins: set) -> pd.DataFrame:
    """
    Carga early_buyers + coin_prices SOLO para coins de train.
    Incluye first_entry_seconds y total_sol_spent para los nuevos scores.
    """
    print("  Cargando datos de early_buyers + coin_prices (solo train)...")
    df = pd.read_sql("""
        SELECT
            eb.wallet_address,
            eb.coin_address,
            eb.first_entry_seconds,
            eb.total_sol_spent,
            eb.tier,
            cp.label,
            cp.max_multiple,
            cp.rug_detected,
            c.created_at
        FROM early_buyers eb
        INNER JOIN coin_prices cp ON eb.coin_address = cp.coin_address
        INNER JOIN coins c        ON eb.coin_address = c.coin_address
        WHERE cp.label IS NOT NULL
    """, engine)

    before = len(df)
    df     = df[df["coin_address"].isin(train_coins)]
    print(f"  {len(df):,} filas (de {before:,} totales) | "
          f"solo {len(train_coins):,} coins de train")
    print(f"  {df['wallet_address'].nunique():,} wallets únicas en train")
    return df


def load_first_seen(engine) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT
            eb.wallet_address,
            MIN(c.created_at) AS first_seen_at
        FROM early_buyers eb
        INNER JOIN coins c ON eb.coin_address = c.coin_address
        GROUP BY eb.wallet_address
    """, engine)


# ── Cálculo de métricas ────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:

    # ── Baseline del train set ────────────────────────────────
    global_win_rate = df["label"].mean()
    global_neg_rate = df["rug_detected"].mean()
    print(f"  Win rate global (train):   {global_win_rate:.4f}  "
          f"({global_win_rate:.1%})")
    print(f"  Rug rate global (train):   {global_neg_rate:.4f}  "
          f"({global_neg_rate:.1%})")

    # ── Cap de ROI ────────────────────────────────────────────
    print("  Aplicando cap de ROI por coin...")
    df = df.copy()
    df["roi_capped"] = df["max_multiple"].clip(upper=ROI_CAP).fillna(1.0)

    # ── Métricas base por wallet ──────────────────────────────
    print("  Calculando métricas por wallet (sobre datos de train)...")

    base = df.groupby("wallet_address").agg(
        appearances_total = ("coin_address", "count"),
        wins              = ("label",        "sum"),
        negative_rate     = ("rug_detected", "mean"),
    ).reset_index()
    base["win_rate"] = base["wins"] / base["appearances_total"]

    avg_roi = (
        df.groupby("wallet_address")["roi_capped"]
        .mean()
        .reset_index()
        .rename(columns={"roi_capped": "avg_roi_capped"})
    )
    base = base.merge(avg_roi, on="wallet_address", how="left")
    base["avg_roi_capped"] = base["avg_roi_capped"].fillna(1.0)

    # ── Velocidad de entrada ──────────────────────────────────
    # Wallets que entran antes tienen ventaja de información.
    # entry_component = 1.0 si avg_entry_seconds = 0
    #                 = 0.0 si avg_entry_seconds = 180
    print("  Calculando velocidad de entrada...")
    entry_stats = df.groupby("wallet_address").agg(
        avg_entry_seconds = ("first_entry_seconds", "mean"),
        pct_tier1_entries = ("tier", lambda x: (x == 1).mean()),
    ).reset_index()
    base = base.merge(entry_stats, on="wallet_address", how="left")
    base["avg_entry_seconds"] = base["avg_entry_seconds"].fillna(90.0)
    base["pct_tier1_entries"] = base["pct_tier1_entries"].fillna(0.0)

    # ── Conviction ratio ──────────────────────────────────────
    # Ratio SOL medio en winners vs SOL medio en losers.
    # Si la wallet pone sistemáticamente más dinero cuando acierta,
    # indica criterio real, no suerte.
    #   conviction_ratio > 1.0 → apuesta más en ganadores (buena señal)
    #   conviction_ratio = 1.0 → neutral
    #   conviction_ratio < 1.0 → apuesta más en perdedores (mala señal)
    print("  Calculando conviction ratio...")

    sol_by_outcome = (
        df.groupby(["wallet_address", "label"])["total_sol_spent"]
        .mean()
        .unstack()
    )
    sol_by_outcome.columns = ["avg_sol_loss", "avg_sol_win"]
    sol_by_outcome = sol_by_outcome.reset_index()
    sol_by_outcome["avg_sol_win"]  = sol_by_outcome["avg_sol_win"].fillna(0.0)
    sol_by_outcome["avg_sol_loss"] = sol_by_outcome["avg_sol_loss"].fillna(0.0)

    sol_by_outcome["conviction_ratio"] = np.where(
        sol_by_outcome["avg_sol_loss"] > 0,
        (sol_by_outcome["avg_sol_win"] /
         sol_by_outcome["avg_sol_loss"]).clip(0.0, 10.0),
        1.0   # sin losers → neutral (no penalizar)
    )
    # Sin winners → convicción no demostrada
    sol_by_outcome.loc[
        sol_by_outcome["avg_sol_win"] == 0, "conviction_ratio"
    ] = 0.5

    base = base.merge(
        sol_by_outcome[["wallet_address", "conviction_ratio"]],
        on="wallet_address", how="left"
    )
    base["conviction_ratio"] = base["conviction_ratio"].fillna(1.0)

    # ── Bayesian win rate y lift ──────────────────────────────
    print("  Calculando Bayesian win rate y lift...")
    alpha = PRIOR_STRENGTH * global_win_rate
    base["bayesian_win_rate"] = (
        (base["wins"] + alpha) /
        (base["appearances_total"] + PRIOR_STRENGTH)
    )
    base["win_lift"] = base["bayesian_win_rate"] / global_win_rate

    # ── Componentes del score ─────────────────────────────────
    print("  Calculando scores...")

    # win: con MAX_LIFT=8 y baseline ~5%, una wallet con ~40%
    #      de win rate ya satura este componente.
    win_component = (base["win_lift"] / MAX_LIFT).clip(0.0, 1.0)

    # roi: 5x capeado = máximo
    roi_component = (base["avg_roi_capped"] / 5.0).clip(0.0, 1.0)

    # entry: cuanto antes entra, mayor el componente
    entry_component = (
        1.0 - base["avg_entry_seconds"] / 180.0
    ).clip(0.0, 1.0)

    # conviction: ratio=1 → 0.0, ratio=4 → 1.0
    conviction_component = (
        (base["conviction_ratio"] - 1.0) / 3.0
    ).clip(0.0, 1.0)

    penalization = base["negative_rate"] * PENALIZATION

    raw_score = (
        win_component        * 0.65 +
        roi_component        * 0.20 +
        entry_component      * 0.05 +
        conviction_component * 0.10 -
        penalization
    ).clip(0.0, 1.0)

    base["score_reliable"]    = base["appearances_total"] >= MIN_APPEARANCES
    base["performance_score"] = raw_score

    # ── Diagnóstico ───────────────────────────────────────────
    reliable = base[base["score_reliable"]]

    print(f"\n  Diagnóstico wallets fiables ({len(reliable):,}):")
    print(f"    Win rate medio:             {reliable['win_rate'].mean():.3f}")
    print(f"    Lift medio:                 {reliable['win_lift'].mean():.2f}x")
    print(f"    Avg ROI capeado medio:      {reliable['avg_roi_capped'].mean():.2f}x")
    print(f"    Avg entry seconds medio:    {reliable['avg_entry_seconds'].mean():.1f}s")
    print(f"    Pct tier1 entries medio:    {reliable['pct_tier1_entries'].mean():.1%}")
    print(f"    Conviction ratio medio:     {reliable['conviction_ratio'].mean():.2f}x")
    print(f"    Negative rate medio:        {reliable['negative_rate'].mean():.3f}")
    print(f"    Score medio:                {reliable['performance_score'].mean():.3f}")

    return base


# ── Escritura en base de datos ─────────────────────────────────

def save_metrics(engine, metrics: pd.DataFrame, first_seen: pd.DataFrame):
    metrics = metrics.merge(first_seen, on="wallet_address", how="left")
    metrics["last_calculated_at"] = datetime.now(timezone.utc)

    for col in ["win_rate", "avg_roi_capped", "negative_rate",
                "performance_score"]:
        if col in metrics.columns:
            metrics[col] = metrics[col].round(6)

    metrics = metrics.rename(columns={"avg_roi_capped": "avg_roi"})

    print(f"\n  Guardando {len(metrics):,} wallets en wallet_metrics...")
    rows       = metrics.to_dict(orient="records")
    batch_size = 10_000
    inserted   = 0

    with engine.connect() as conn:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            conn.execute(text("""
                INSERT INTO wallet_metrics (
                    wallet_address, appearances_total,
                    win_rate, avg_roi, negative_rate,
                    performance_score, score_reliable,
                    last_calculated_at, first_seen_at
                )
                VALUES (
                    :wallet_address, :appearances_total,
                    :win_rate, :avg_roi, :negative_rate,
                    :performance_score, :score_reliable,
                    :last_calculated_at, :first_seen_at
                )
                ON CONFLICT (wallet_address) DO UPDATE SET
                    appearances_total  = EXCLUDED.appearances_total,
                    win_rate           = EXCLUDED.win_rate,
                    avg_roi            = EXCLUDED.avg_roi,
                    negative_rate      = EXCLUDED.negative_rate,
                    performance_score  = EXCLUDED.performance_score,
                    score_reliable     = EXCLUDED.score_reliable,
                    last_calculated_at = EXCLUDED.last_calculated_at
            """), batch)
            conn.commit()
            inserted += len(batch)
            print(f"    {inserted:,} / {len(rows):,} wallets...",
                  end="\r", flush=True)

    print(f"    {len(rows):,} wallets guardadas.              ")


# ── Resumen ────────────────────────────────────────────────────

def print_summary(engine):
    with engine.connect() as conn:
        total    = conn.execute(text(
            "SELECT COUNT(*) FROM wallet_metrics")).scalar()
        reliable = conn.execute(text(
            "SELECT COUNT(*) FROM wallet_metrics WHERE score_reliable = TRUE"
        )).scalar()
        high     = conn.execute(text(
            "SELECT COUNT(*) FROM wallet_metrics "
            "WHERE performance_score >= 0.5 AND score_reliable = TRUE"
        )).scalar()
        neg      = conn.execute(text(
            "SELECT COUNT(*) FROM wallet_metrics WHERE negative_rate > 0.4"
        )).scalar()

    with engine.connect() as conn:
        top10 = conn.execute(text("""
            SELECT wallet_address, appearances_total, win_rate,
                   avg_roi, negative_rate, performance_score
            FROM wallet_metrics
            WHERE score_reliable = TRUE
            ORDER BY performance_score DESC
            LIMIT 10
        """)).fetchall()

        score_dist = conn.execute(text("""
            SELECT ROUND(performance_score::numeric, 1) AS bucket,
                   COUNT(*) AS n
            FROM wallet_metrics
            GROUP BY bucket ORDER BY bucket
        """)).fetchall()

    print(f"\n  Total wallets:              {total:>10,}")
    print(f"  Wallets fiables (>={MIN_APPEARANCES}):       {reliable:>10,}")
    print(f"  Fiables con score >= 0.5:   {high:>10,}")
    print(f"  Negative rate > 40%:        {neg:>10,}")

    print(f"\n  Distribución de scores:")
    for row in score_dist:
        bar = "█" * min(int(row[1] / max(total / 50, 1)), 40)
        print(f"    {float(row[0]):.1f}  {bar}  ({row[1]:,})")

    print(f"\n  Top 10 wallets fiables:")
    print(f"  {'wallet':<20} {'apps':>4} {'win%':>6} "
          f"{'roi':>7} {'neg%':>6} {'score':>6}")
    print("  " + "-" * 55)
    for r in top10:
        addr = r[0][:18] + ".."
        print(f"  {addr:<20} {r[1]:>4} {r[2]:>6.2f} "
              f"{r[3]:>7.2f}x {r[4]:>6.2f} {r[5]:>6.3f}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WALLET SCORING — lift + Bayesian + entry speed + conviction")
    print(f"  ROI cap:          {ROI_CAP}x")
    print(f"  Prior strength:   {PRIOR_STRENGTH} pseudo-observaciones")
    print(f"  MAX_LIFT:         {MAX_LIFT}x  (~40% win rate satura con baseline 5%)")
    print(f"  Pesos:            win=50%, roi=20%, entry=15%, conviction=15%")
    print(f"  Penalización:     negative_rate × {PENALIZATION}")
    print(f"  Min apariciones:  {MIN_APPEARANCES}  (flag score_reliable)")
    print("=" * 60)

    engine = get_engine()

    print("\n[0/3] Determinando split temporal...")
    train_coins, test_coins = get_train_test_coins(engine)
    print(f"  Train: {len(train_coins):,} coins | Test: {len(test_coins):,} coins")

    print("\n[1/3] Cargando datos (solo train)...")
    df         = load_data(engine, train_coins)
    first_seen = load_first_seen(engine)

    if df.empty:
        print("Sin datos en train. Verifica early_buyers y coin_prices.")
        return

    print("\n[2/3] Calculando métricas...")
    metrics = calculate_metrics(df)

    print("\n[3/3] Guardando en wallet_metrics...")
    save_metrics(engine, metrics, first_seen)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print_summary(engine)
    print("=" * 60)


if __name__ == "__main__":
    main()