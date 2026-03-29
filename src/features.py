"""
features.py
-----------
Calcula las features de cada coin para el modelo de ML.

Combina:
  - early_buyers       → comportamiento de compra en primeros 3min
  - wallet_metrics     → calidad histórica de cada wallet
  - coin_prices        → label y precio
  - coins              → creator_wallet y timestamp de creación

Output: tabla coin_features (una fila por coin con label calculada)

Uso:
    python src/features.py

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


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_all_data(engine):
    """Carga todas las tablas necesarias en memoria."""

    print("  Cargando early_buyers...")
    eb = pd.read_sql("""
        SELECT
            eb.coin_address,
            eb.wallet_address,
            eb.first_entry_seconds,
            eb.total_sol_spent,
            eb.n_trades,
            eb.tier
        FROM early_buyers eb
    """, engine)
    print(f"    {len(eb):,} filas, {eb['coin_address'].nunique():,} coins")

    print("  Cargando wallet_metrics...")
    wm = pd.read_sql("""
        SELECT
            wallet_address,
            appearances_total,
            performance_score,
            score_reliable,
            negative_rate
        FROM wallet_metrics
    """, engine)
    print(f"    {len(wm):,} wallets")

    print("  Cargando coin_prices...")
    cp = pd.read_sql("""
        SELECT
            coin_address,
            label,
            max_multiple,
            rug_detected
        FROM coin_prices
        WHERE label IS NOT NULL
    """, engine)
    print(f"    {len(cp):,} coins con label")

    print("  Cargando coins...")
    coins = pd.read_sql("""
        SELECT coin_address, creator_wallet, created_at
        FROM coins
    """, engine)
    print(f"    {len(coins):,} coins")

    return eb, wm, cp, coins


# ── Co-ocurrencia ──────────────────────────────────────────────

def compute_cooccurrence(eb: pd.DataFrame, cp: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par de wallets, cuenta cuántas coins exitosas (label=1)
    han comprado juntas.

    Devuelve un DataFrame con columnas:
        wallet_a, wallet_b, successful_cooccurrences
    """
    print("  Calculando co-ocurrencias en coins exitosas...")

    # Solo coins exitosas
    successful_coins = set(cp[cp["label"] == 1]["coin_address"])
    eb_success = eb[eb["coin_address"].isin(successful_coins)][
        ["coin_address", "wallet_address"]
    ]

    if eb_success.empty:
        print("    Sin coins exitosas para calcular co-ocurrencia.")
        return pd.DataFrame(columns=["wallet_a", "wallet_b", "cooc_count"])

    # Self-join para obtener todos los pares por coin
    pairs = eb_success.merge(eb_success, on="coin_address", suffixes=("_a", "_b"))
    # Eliminar pares de una wallet consigo misma
    pairs = pairs[pairs["wallet_address_a"] < pairs["wallet_address_b"]]

    if pairs.empty:
        return pd.DataFrame(columns=["wallet_a", "wallet_b", "cooc_count"])

    # Contar cuántas coins comparten cada par
    cooc = (
        pairs.groupby(["wallet_address_a", "wallet_address_b"])
        .size()
        .reset_index(name="cooc_count")
        .rename(columns={
            "wallet_address_a": "wallet_a",
            "wallet_address_b": "wallet_b"
        })
    )

    print(f"    {len(cooc):,} pares de wallets con co-ocurrencia > 0")
    return cooc


def get_avg_cooccurrence(coin_wallets: pd.Series, cooc: pd.DataFrame) -> float:
    """
    Dado el conjunto de wallets de una coin, calcula la media de
    co-ocurrencias entre todos sus pares.
    """
    if len(coin_wallets) < 2 or cooc.empty:
        return 0.0

    wallets = set(coin_wallets)

    # Filtrar pares donde ambas wallets están en esta coin
    mask = (
        cooc["wallet_a"].isin(wallets) &
        cooc["wallet_b"].isin(wallets)
    )
    relevant = cooc[mask]

    if relevant.empty:
        return 0.0

    return float(relevant["cooc_count"].mean())


# ── Cálculo de features por coin ───────────────────────────────

def compute_features(
    eb: pd.DataFrame,
    wm: pd.DataFrame,
    cp: pd.DataFrame,
    coins: pd.DataFrame,
    cooc: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula todas las features para cada coin que tiene label.
    """
    # Unir early_buyers con wallet_metrics
    eb_scored = eb.merge(wm, on="wallet_address", how="left")

    # Score neutro para wallets sin historial en wallet_metrics
    eb_scored["performance_score"] = eb_scored["performance_score"].fillna(0.1)
    eb_scored["score_reliable"]    = eb_scored["score_reliable"].fillna(False)
    eb_scored["negative_rate"]     = eb_scored["negative_rate"].fillna(0.0)
    eb_scored["appearances_total"] = eb_scored["appearances_total"].fillna(0)

    # Unir coins con su creator_wallet y created_at
    coins_meta = coins[["coin_address", "creator_wallet", "created_at"]].copy()

    # Solo procesar coins que tienen label
    target_coins = set(cp["coin_address"])
    eb_target    = eb_scored[eb_scored["coin_address"].isin(target_coins)]

    print(f"  Calculando features para {len(target_coins):,} coins...")

    records = []
    total   = len(target_coins)

    for idx, coin_addr in enumerate(target_coins):
        if idx % 500 == 0:
            print(f"    {idx:,} / {total:,} coins...", end="\r", flush=True)

        coin_eb  = eb_target[eb_target["coin_address"] == coin_addr]
        meta_row = coins_meta[coins_meta["coin_address"] == coin_addr]

        if coin_eb.empty:
            continue

        wallets         = coin_eb["wallet_address"]
        creator_wallet  = meta_row["creator_wallet"].values[0] if not meta_row.empty else None
        created_at      = meta_row["created_at"].values[0]     if not meta_row.empty else None

        # ── Bloque 1: Composición de wallets ──────────────────

        n_early_buyers        = len(coin_eb)
        n_reliable_wallets    = int(coin_eb["score_reliable"].sum())
        avg_wallet_score      = float(coin_eb["performance_score"].mean())
        max_wallet_score      = float(coin_eb["performance_score"].max())
        pct_high_score        = float((coin_eb["performance_score"] > 0.6).mean())
        pct_negative_wallets  = float((coin_eb["negative_rate"] > 0.4).mean())
        pct_new_wallets       = float((coin_eb["appearances_total"] < 3).mean())

        # ── Bloque 2: Co-ocurrencia ────────────────────────────

        avg_cooccurrence = get_avg_cooccurrence(wallets, cooc)

        # ── Bloque 3: Comportamiento de compra ────────────────

        total_volume_sol = float(coin_eb["total_sol_spent"].sum())
        avg_buy_size_sol = float(coin_eb["total_sol_spent"].mean())
        std_buy_size_sol = float(coin_eb["total_sol_spent"].std()) if n_early_buyers > 1 else 0.0

        # Concentración top 5
        top5_vol         = coin_eb["total_sol_spent"].nlargest(5).sum()
        concentration_t5 = float(top5_vol / total_volume_sol) if total_volume_sol > 0 else 0.0

        n_tier1_buyers   = int((coin_eb["tier"] == 1).sum())

        # Creator como early buyer
        creator_is_buyer = bool(
            creator_wallet is not None and
            creator_wallet in wallets.values
        )

        # ── Bloque 4: Velocidad de acumulación ────────────────

        buys_first_20s   = int((coin_eb["first_entry_seconds"] <= 20).sum())
        buys_20_to_60    = int(
            ((coin_eb["first_entry_seconds"] > 20) &
             (coin_eb["first_entry_seconds"] <= 60)).sum()
        )
        buys_60_to_180   = int(
            ((coin_eb["first_entry_seconds"] > 60) &
             (coin_eb["first_entry_seconds"] <= 180)).sum()
        )

        first_90s_buys   = int((coin_eb["first_entry_seconds"] <= 90).sum())
        last_90s_buys    = int((coin_eb["first_entry_seconds"] > 90).sum())
        acceleration_ratio = float(first_90s_buys / max(last_90s_buys, 1))

        # Tiempo hasta la quinta compra
        sorted_times     = coin_eb["first_entry_seconds"].sort_values()
        time_to_5th      = float(sorted_times.iloc[4]) if len(sorted_times) >= 5 else None

        # ── Bloque 5: Contexto temporal ───────────────────────

        if created_at is not None:
            ts = pd.Timestamp(created_at)
            hour_utc    = int(ts.hour)
            day_of_week = int(ts.dayofweek)  # 0=lunes
        else:
            hour_utc    = None
            day_of_week = None

        records.append({
            "coin_address":           coin_addr,
            "calculated_at":          datetime.now(timezone.utc),
            "n_early_buyers":         n_early_buyers,
            "n_reliable_wallets":     n_reliable_wallets,
            "avg_wallet_score":       round(avg_wallet_score, 6),
            "max_wallet_score":       round(max_wallet_score, 6),
            "pct_high_score_wallets": round(pct_high_score, 6),
            "pct_negative_wallets":   round(pct_negative_wallets, 6),
            "pct_new_wallets":        round(pct_new_wallets, 6),
            "avg_cooccurrence_score": round(avg_cooccurrence, 4),
            "total_volume_sol":       round(total_volume_sol, 6),
            "avg_buy_size_sol":       round(avg_buy_size_sol, 6),
            "std_buy_size_sol":       round(std_buy_size_sol, 6),
            "concentration_top5":     round(concentration_t5, 6),
            "n_tier1_buyers":         n_tier1_buyers,
            "creator_is_buyer":       creator_is_buyer,
            "buys_in_first_20s":      buys_first_20s,
            "buys_20s_to_60s":        buys_20_to_60,
            "buys_60s_to_180s":       buys_60_to_180,
            "acceleration_ratio":     round(acceleration_ratio, 4),
            "time_to_5th_buy":        round(time_to_5th, 2) if time_to_5th else None,
            "hour_utc":               hour_utc,
            "day_of_week":            day_of_week,
        })

    print(f"    {total:,} / {total:,} coins procesadas.   ")
    return pd.DataFrame(records)


# ── Escritura en base de datos ─────────────────────────────────

def save_features(engine, df: pd.DataFrame):
    """Inserta o actualiza coin_features."""

    print(f"  Guardando {len(df):,} filas en coin_features...")

    rows       = df.to_dict(orient="records")
    batch_size = 2_000
    inserted   = 0

    with engine.connect() as conn:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            conn.execute(text("""
                INSERT INTO coin_features (
                    coin_address, calculated_at,
                    n_early_buyers, n_reliable_wallets,
                    avg_wallet_score, max_wallet_score,
                    pct_high_score_wallets, pct_negative_wallets, pct_new_wallets,
                    avg_cooccurrence_score,
                    total_volume_sol, avg_buy_size_sol, std_buy_size_sol,
                    concentration_top5, n_tier1_buyers, creator_is_buyer,
                    buys_in_first_20s, buys_20s_to_60s, buys_60s_to_180s,
                    acceleration_ratio, time_to_5th_buy,
                    hour_utc, day_of_week
                )
                VALUES (
                    :coin_address, :calculated_at,
                    :n_early_buyers, :n_reliable_wallets,
                    :avg_wallet_score, :max_wallet_score,
                    :pct_high_score_wallets, :pct_negative_wallets, :pct_new_wallets,
                    :avg_cooccurrence_score,
                    :total_volume_sol, :avg_buy_size_sol, :std_buy_size_sol,
                    :concentration_top5, :n_tier1_buyers, :creator_is_buyer,
                    :buys_in_first_20s, :buys_20s_to_60s, :buys_60s_to_180s,
                    :acceleration_ratio, :time_to_5th_buy,
                    :hour_utc, :day_of_week
                )
                ON CONFLICT (coin_address) DO UPDATE SET
                    calculated_at           = EXCLUDED.calculated_at,
                    n_early_buyers          = EXCLUDED.n_early_buyers,
                    n_reliable_wallets      = EXCLUDED.n_reliable_wallets,
                    avg_wallet_score        = EXCLUDED.avg_wallet_score,
                    max_wallet_score        = EXCLUDED.max_wallet_score,
                    pct_high_score_wallets  = EXCLUDED.pct_high_score_wallets,
                    pct_negative_wallets    = EXCLUDED.pct_negative_wallets,
                    pct_new_wallets         = EXCLUDED.pct_new_wallets,
                    avg_cooccurrence_score  = EXCLUDED.avg_cooccurrence_score,
                    total_volume_sol        = EXCLUDED.total_volume_sol,
                    avg_buy_size_sol        = EXCLUDED.avg_buy_size_sol,
                    std_buy_size_sol        = EXCLUDED.std_buy_size_sol,
                    concentration_top5      = EXCLUDED.concentration_top5,
                    n_tier1_buyers          = EXCLUDED.n_tier1_buyers,
                    creator_is_buyer        = EXCLUDED.creator_is_buyer,
                    buys_in_first_20s       = EXCLUDED.buys_in_first_20s,
                    buys_20s_to_60s         = EXCLUDED.buys_20s_to_60s,
                    buys_60s_to_180s        = EXCLUDED.buys_60s_to_180s,
                    acceleration_ratio      = EXCLUDED.acceleration_ratio,
                    time_to_5th_buy         = EXCLUDED.time_to_5th_buy,
                    hour_utc                = EXCLUDED.hour_utc,
                    day_of_week             = EXCLUDED.day_of_week
            """), batch)
            conn.commit()
            inserted += len(batch)
            print(f"    {inserted:,} / {len(rows):,} filas...", end="\r", flush=True)

    print(f"    {len(rows):,} filas guardadas.              ")


# ── Resumen ────────────────────────────────────────────────────

def print_summary(engine):
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM coin_features")
        ).scalar()

        stats = conn.execute(text("""
            SELECT
                AVG(n_early_buyers)         AS avg_buyers,
                AVG(avg_wallet_score)       AS avg_wscore,
                AVG(pct_new_wallets)        AS avg_new,
                AVG(acceleration_ratio)     AS avg_accel,
                SUM(CASE WHEN creator_is_buyer THEN 1 ELSE 0 END) AS creator_buyers,
                AVG(avg_cooccurrence_score) AS avg_cooc
            FROM coin_features
        """)).fetchone()

    print(f"\n  coin_features generadas: {total:,}")
    if stats and stats[0]:
        print(f"  Avg early buyers:        {stats[0]:.1f}")
        print(f"  Avg wallet score:        {stats[1]:.4f}")
        print(f"  Avg % wallets nuevas:    {stats[2]:.2%}")
        print(f"  Avg acceleration ratio:  {stats[3]:.2f}")
        print(f"  Coins con creator buyer: {stats[4]:,}")
        print(f"  Avg co-ocurrencia:       {stats[5]:.4f}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FEATURES — coin_features")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/4] Cargando datos...")
    eb, wm, cp, coins = load_all_data(engine)

    print("\n[2/4] Calculando co-ocurrencias...")
    cooc = compute_cooccurrence(eb, cp)

    print("\n[3/4] Calculando features por coin...")
    features_df = compute_features(eb, wm, cp, coins, cooc)

    if features_df.empty:
        print("Sin features generadas. Verifica que los datos estén cargados.")
        return

    print("\n[4/4] Guardando en coin_features...")
    save_features(engine, features_df)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print_summary(engine)
    print("=" * 60)


if __name__ == "__main__":
    main()