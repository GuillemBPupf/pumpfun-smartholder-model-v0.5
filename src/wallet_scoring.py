"""
wallet_scoring.py
-----------------
Calcula el performance_score de cada wallet usando su historial
completo en el dataset.

Fórmula del score:
  roi_capeado     = min(max_multiple, 10)  por cada coin
  avg_roi_capped  = media de roi_capeado (resistente a outliers extremos)

  base_score   = win_rate * 0.80 + min(avg_roi_capped / 5, 1) * 0.20
  penalización = negative_rate * 0.30
  score final  = max(0, base_score - penalización)

  Si appearances_total < 5 → score_reliable = FALSE
    y el score se fuerza a 0.1 (neutro)

Uso:
    python src/wallet_scoring.py

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Cap máximo de ROI por coin antes de calcular la media
ROI_CAP = 10.0

# Umbral de apariciones para considerar el score fiable
MIN_APPEARANCES = 5

# Score neutro para wallets sin historial fiable
NEUTRAL_SCORE = 0.1


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_data(engine) -> pd.DataFrame:
    """
    Carga el cruce entre early_buyers y coin_prices.
    Solo coins con label calculada (tienen precio a 5min).
    """
    query = """
        SELECT
            eb.wallet_address,
            eb.coin_address,
            cp.label,
            cp.max_multiple,
            cp.rug_detected,
            c.created_at
        FROM early_buyers eb
        INNER JOIN coin_prices cp ON eb.coin_address = cp.coin_address
        INNER JOIN coins c        ON eb.coin_address = c.coin_address
        WHERE cp.label IS NOT NULL
    """
    print("  Cargando datos de early_buyers + coin_prices...")
    df = pd.read_sql(query, engine)
    print(f"  {len(df):,} filas cargadas "
          f"({df['wallet_address'].nunique():,} wallets únicas, "
          f"{df['coin_address'].nunique():,} coins únicas)")
    return df


def load_first_seen(engine) -> pd.DataFrame:
    """
    Timestamp de la primera aparición de cada wallet
    (sobre todas sus apariciones, no solo las con precio calculado).
    """
    query = """
        SELECT
            eb.wallet_address,
            MIN(c.created_at) AS first_seen_at
        FROM early_buyers eb
        INNER JOIN coins c ON eb.coin_address = c.coin_address
        GROUP BY eb.wallet_address
    """
    return pd.read_sql(query, engine)


# ── Cálculo de métricas ────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada wallet calcula:
      appearances_total : coins con precio calculado en que participó
      win_rate          : % de coins con label=1
      avg_roi_capped    : media del ROI capeado a ROI_CAP por coin
      negative_rate     : % de coins con rug_detected=TRUE
      performance_score : score final 0-1
      score_reliable    : TRUE si appearances_total >= MIN_APPEARANCES
    """
    print("  Aplicando cap de ROI por coin...")
    df["roi_capped"] = df["max_multiple"].clip(upper=ROI_CAP).fillna(1.0)

    print("  Calculando métricas por wallet...")

    # Métricas base por wallet
    base = df.groupby("wallet_address").agg(
        appearances_total = ("coin_address", "count"),
        win_rate          = ("label",        "mean"),
        negative_rate     = ("rug_detected", "mean"),
    ).reset_index()

    # Media del ROI capeado
    avg_roi = (
        df.groupby("wallet_address")["roi_capped"]
        .mean()
        .reset_index()
        .rename(columns={"roi_capped": "avg_roi_capped"})
    )

    metrics = base.merge(avg_roi, on="wallet_address", how="left")
    metrics["avg_roi_capped"] = metrics["avg_roi_capped"].fillna(1.0)

    print("  Calculando scores...")

    # Componente ROI normalizado a [0,1]
    # avg_roi_capped=5 → componente_roi=1.0 (techo en 5x de media)
    componente_roi = np.minimum(metrics["avg_roi_capped"] / 5.0, 1.0)

    base_score   = metrics["win_rate"] * 0.80 + componente_roi * 0.20
    penalizacion = metrics["negative_rate"] * 0.30
    raw_score    = (base_score - penalizacion).clip(lower=0.0, upper=1.0)

    metrics["score_reliable"]    = metrics["appearances_total"] >= MIN_APPEARANCES
    metrics["performance_score"] = np.where(
        metrics["score_reliable"],
        raw_score,
        NEUTRAL_SCORE
    )

    # Diagnóstico
    reliable = metrics[metrics["score_reliable"]]
    print(f"\n  Diagnóstico de wallets fiables ({len(reliable):,}):")
    print(f"    Win rate medio:        {reliable['win_rate'].mean():.3f}")
    print(f"    Avg ROI capeado medio: {reliable['avg_roi_capped'].mean():.2f}x")
    print(f"    Negative rate medio:   {reliable['negative_rate'].mean():.3f}")

    return metrics


# ── Escritura en base de datos ─────────────────────────────────

def save_metrics(engine, metrics: pd.DataFrame, first_seen: pd.DataFrame):
    """
    Inserta o actualiza los scores en wallet_metrics.
    avg_roi en la BD almacena avg_roi_capped (más representativo).
    """
    metrics = metrics.merge(first_seen, on="wallet_address", how="left")
    metrics["last_calculated_at"] = datetime.now(timezone.utc)

    for col in ["win_rate", "avg_roi_capped", "negative_rate", "performance_score"]:
        metrics[col] = metrics[col].round(6)

    metrics = metrics.rename(columns={"avg_roi_capped": "avg_roi"})

    print(f"\n  Guardando {len(metrics):,} wallets en wallet_metrics...")

    rows       = metrics.to_dict(orient="records")
    batch_size = 10_000
    inserted   = 0

    with engine.connect() as conn:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            conn.execute(
                text("""
                    INSERT INTO wallet_metrics (
                        wallet_address,
                        appearances_total,
                        win_rate,
                        avg_roi,
                        negative_rate,
                        performance_score,
                        score_reliable,
                        last_calculated_at,
                        first_seen_at
                    )
                    VALUES (
                        :wallet_address,
                        :appearances_total,
                        :win_rate,
                        :avg_roi,
                        :negative_rate,
                        :performance_score,
                        :score_reliable,
                        :last_calculated_at,
                        :first_seen_at
                    )
                    ON CONFLICT (wallet_address) DO UPDATE SET
                        appearances_total  = EXCLUDED.appearances_total,
                        win_rate           = EXCLUDED.win_rate,
                        avg_roi            = EXCLUDED.avg_roi,
                        negative_rate      = EXCLUDED.negative_rate,
                        performance_score  = EXCLUDED.performance_score,
                        score_reliable     = EXCLUDED.score_reliable,
                        last_calculated_at = EXCLUDED.last_calculated_at
                """),
                batch
            )
            conn.commit()
            inserted += len(batch)
            print(f"    {inserted:,} / {len(rows):,} wallets...",
                  end="\r", flush=True)

    print(f"    {len(rows):,} wallets guardadas.              ")


# ── Resumen ────────────────────────────────────────────────────

def print_summary(engine):
    with engine.connect() as conn:
        total    = conn.execute(text("SELECT COUNT(*) FROM wallet_metrics")).scalar()
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
            SELECT
                ROUND(performance_score::numeric, 1) AS bucket,
                COUNT(*) AS n
            FROM wallet_metrics
            GROUP BY bucket
            ORDER BY bucket
        """)).fetchall()

    print(f"\n  Total wallets:              {total:>10,}")
    print(f"  Wallets fiables (>={MIN_APPEARANCES}):       {reliable:>10,}")
    print(f"  Fiables con score >= 0.5:   {high:>10,}")
    print(f"  Negative rate > 40%:        {neg:>10,}")

    print(f"\n  Distribución de scores (todas las wallets):")
    for row in score_dist:
        bar = "█" * min(int(row[1] / max(total / 50, 1)), 40)
        print(f"    {float(row[0]):.1f}  {bar}  ({row[1]:,})")

    print(f"\n  Top 10 wallets fiables:")
    print(f"  {'wallet':<20} {'apps':>4} {'win%':>6} {'roi_avg':>8} "
          f"{'neg%':>6} {'score':>6}")
    print("  " + "-" * 60)
    for r in top10:
        addr = r[0][:18] + ".."
        print(f"  {addr:<20} {r[1]:>4} {r[2]:>6.2f} {r[3]:>8.2f}x "
              f"{r[4]:>6.2f} {r[5]:>6.3f}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WALLET SCORING")
    print(f"  ROI cap por coin:          {ROI_CAP}x")
    print(f"  Pesos: win_rate=80%, avg_roi_capped=20%")
    print(f"  Penalización: negative_rate * 0.30")
    print(f"  Mínimo apariciones fiable: {MIN_APPEARANCES}")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/3] Cargando datos...")
    df         = load_data(engine)
    first_seen = load_first_seen(engine)

    if df.empty:
        print("Sin datos. Verifica que early_buyers y coin_prices tengan filas.")
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