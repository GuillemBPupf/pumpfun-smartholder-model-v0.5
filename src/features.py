"""
features.py
-----------
Calcula las features de cada coin para el modelo de ML.
Versión vectorizada + co-ocurrencia sin leakage (Opción A):
  - La matriz de co-ocurrencia se construye SOLO con coins del
    conjunto de entrenamiento (80% más antiguo).
  - Esa misma matriz se aplica al test, sin que el test aporte
    información a la co-ocurrencia.

Uso:
    python src/features.py

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

load_dotenv()

# Mismo split que usa model.py — crítico que sean consistentes
TRAIN_RATIO = 0.80


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_all_data(engine):
    print("  Cargando early_buyers + wallet_metrics (join en SQL)...")
    eb_scored = pd.read_sql("""
        SELECT
            eb.coin_address,
            eb.wallet_address,
            eb.first_entry_seconds,
            eb.total_sol_spent,
            eb.n_trades,
            eb.tier,
            COALESCE(wm.performance_score, 0.1)  AS performance_score,
            COALESCE(wm.score_reliable,    FALSE) AS score_reliable,
            COALESCE(wm.negative_rate,     0.0)   AS negative_rate,
            COALESCE(wm.appearances_total, 0)     AS appearances_total
        FROM early_buyers eb
        LEFT JOIN wallet_metrics wm ON eb.wallet_address = wm.wallet_address
    """, engine)
    print(f"    {len(eb_scored):,} filas early_buyers")

    print("  Cargando coin_prices...")
    cp = pd.read_sql("""
        SELECT coin_address, label, max_multiple, rug_detected
        FROM coin_prices
        WHERE label IS NOT NULL
    """, engine)
    print(f"    {len(cp):,} coins con label")

    print("  Cargando coins (meta)...")
    coins = pd.read_sql("""
        SELECT coin_address, creator_wallet, created_at
        FROM coins
    """, engine)
    print(f"    {len(coins):,} coins")

    return eb_scored, cp, coins


# ── Split temporal para co-ocurrencia sin leakage ─────────────

def get_train_coins(cp: pd.DataFrame, coins: pd.DataFrame) -> set:
    """
    Devuelve el conjunto de coin_addresses que pertenecen al
    80% más antiguo del dataset (igual que model.py).
    La co-ocurrencia se calculará SOLO con estas coins.
    """
    # Unir coin_prices con timestamps para ordenar temporalmente
    cp_with_time = cp.merge(
        coins[["coin_address", "created_at"]],
        on="coin_address", how="left"
    ).sort_values("created_at")

    split_idx   = int(len(cp_with_time) * TRAIN_RATIO)
    train_coins = set(cp_with_time.iloc[:split_idx]["coin_address"])
    test_coins  = set(cp_with_time.iloc[split_idx:]["coin_address"])

    print(f"    Train coins: {len(train_coins):,} | Test coins: {len(test_coins):,}")
    return train_coins


# ── Co-ocurrencia (solo train, aplicada a todos) ───────────────

def compute_cooccurrence(
    eb: pd.DataFrame,
    cp: pd.DataFrame,
    train_coins: set
) -> pd.Series:
    """
    Construye la matriz de co-ocurrencia usando SOLO coins del
    conjunto de entrenamiento con label=1.

    La aplica luego a TODAS las coins (train + test) para calcular
    avg_cooccurrence_score sin leakage.
    """
    print("  Calculando co-ocurrencias (solo sobre coins de train)...")

    # Solo coins exitosas del conjunto de entrenamiento
    successful_train_coins = set(
        cp[(cp["label"] == 1) & (cp["coin_address"].isin(train_coins))]["coin_address"]
    )
    print(f"    Coins exitosas en train: {len(successful_train_coins):,}")

    eb_success = eb[eb["coin_address"].isin(successful_train_coins)][
        ["coin_address", "wallet_address"]
    ].drop_duplicates()

    if eb_success.empty:
        print("    Sin coins exitosas en train para co-ocurrencia.")
        return pd.Series(dtype=float)

    # Pares de wallets en coins exitosas de train
    pairs = eb_success.merge(
        eb_success, on="coin_address", suffixes=("_a", "_b")
    )
    pairs = pairs[pairs["wallet_address_a"] < pairs["wallet_address_b"]]

    if pairs.empty:
        return pd.Series(dtype=float)

    # Conteo de co-ocurrencias entre pares
    pair_counts = (
        pairs.groupby(["wallet_address_a", "wallet_address_b"])
        .size()
        .reset_index(name="cooc_count")
    )
    print(f"    {len(pair_counts):,} pares únicos con co-ocurrencia > 0")

    # Aplicar a TODAS las coins (train + test)
    all_coins  = eb[["coin_address", "wallet_address"]].drop_duplicates()
    coin_pairs = all_coins.merge(
        all_coins, on="coin_address", suffixes=("_a", "_b")
    )
    coin_pairs = coin_pairs[
        coin_pairs["wallet_address_a"] < coin_pairs["wallet_address_b"]
    ]

    if coin_pairs.empty:
        return pd.Series(dtype=float)

    coin_pairs = coin_pairs.merge(
        pair_counts,
        on=["wallet_address_a", "wallet_address_b"],
        how="left"
    )
    coin_pairs["cooc_count"] = coin_pairs["cooc_count"].fillna(0)

    cooc_score = (
        coin_pairs.groupby("coin_address")["cooc_count"]
        .mean()
        .rename("avg_cooccurrence_score")
    )
    print(f"    Co-ocurrencia calculada para {len(cooc_score):,} coins")
    return cooc_score


# ── Features vectorizadas ──────────────────────────────────────

def compute_features(
    eb: pd.DataFrame,
    cp: pd.DataFrame,
    coins: pd.DataFrame,
    cooc: pd.Series
) -> pd.DataFrame:
    """
    Calcula todas las features usando groupby de pandas.
    Sin bucles por coin.
    """
    target_coins = set(cp["coin_address"])
    eb           = eb[eb["coin_address"].isin(target_coins)].copy()
    coins_meta   = coins[["coin_address", "creator_wallet", "created_at"]].copy()

    print(f"  {len(target_coins):,} coins a procesar, "
          f"{len(eb):,} filas de early_buyers")

    # ── Bloque 1: Composición de wallets ──────────────────────
    print("  Calculando bloque 1: composición de wallets...")
    wallet_block = eb.groupby("coin_address").agg(
        n_early_buyers         = ("wallet_address",    "nunique"),
        n_reliable_wallets     = ("score_reliable",    "sum"),
        avg_wallet_score       = ("performance_score", "mean"),
        max_wallet_score       = ("performance_score", "max"),
        pct_high_score_wallets = ("performance_score",
                                   lambda x: (x > 0.6).mean()),
        pct_negative_wallets   = ("negative_rate",
                                   lambda x: (x > 0.4).mean()),
        pct_new_wallets        = ("appearances_total",
                                   lambda x: (x < 3).mean()),
    ).reset_index()

    # ── Bloque 2: Co-ocurrencia ────────────────────────────────
    print("  Calculando bloque 2: co-ocurrencia...")
    cooc_df = (
        cooc.reset_index()
        if not cooc.empty
        else pd.DataFrame(columns=["coin_address", "avg_cooccurrence_score"])
    )

    # ── Bloque 3: Comportamiento de compra ────────────────────
    print("  Calculando bloque 3: comportamiento de compra...")
    buy_block = eb.groupby("coin_address").agg(
        total_volume_sol = ("total_sol_spent", "sum"),
        avg_buy_size_sol = ("total_sol_spent", "mean"),
        std_buy_size_sol = ("total_sol_spent", "std"),
        n_tier1_buyers   = ("tier", lambda x: (x == 1).sum()),
    ).reset_index()
    buy_block["std_buy_size_sol"] = buy_block["std_buy_size_sol"].fillna(0.0)

    # Concentración top 5
    top5 = (
        eb.groupby(["coin_address", "wallet_address"])["total_sol_spent"]
        .sum()
        .reset_index()
        .sort_values(["coin_address", "total_sol_spent"],
                     ascending=[True, False])
    )
    top5_sum = (
        top5.groupby("coin_address")
        .apply(lambda g: g["total_sol_spent"].nlargest(5).sum(),
               include_groups=False)
        .reset_index(name="top5_vol")
    )
    buy_block = buy_block.merge(top5_sum, on="coin_address", how="left")
    buy_block["concentration_top5"] = (
        buy_block["top5_vol"] /
        buy_block["total_volume_sol"].replace(0, np.nan)
    ).fillna(0.0)
    buy_block.drop(columns=["top5_vol"], inplace=True)

    # Creator is buyer (Opción B):
    # TRUE si el primer comprador de la coin realizó más de una
    # compra en los primeros 3 minutos → señal de acumulación del deployer
    print("  Calculando creator_is_buyer (deployer acumulando)...")
    eb_with_creator = eb.merge(
        coins_meta[["coin_address", "creator_wallet"]],
        on="coin_address", how="left"
    )
    creator_flag = (
        eb_with_creator[
            eb_with_creator["wallet_address"] == eb_with_creator["creator_wallet"]
        ]
        .groupby("coin_address")["n_trades"]
        .sum()
        .reset_index()
        .assign(creator_is_buyer=lambda d: d["n_trades"] > 1)
        [["coin_address", "creator_is_buyer"]]
    )
    all_coins_df = pd.DataFrame({"coin_address": list(target_coins)})
    creator_flag = all_coins_df.merge(
        creator_flag, on="coin_address", how="left"
    )
    creator_flag["creator_is_buyer"] = (
        creator_flag["creator_is_buyer"].fillna(False)
    )

    # ── Bloque 4: Velocidad de acumulación ────────────────────
    print("  Calculando bloque 4: velocidad de acumulación...")
    speed_block = eb.groupby("coin_address").agg(
        buys_in_first_20s = ("first_entry_seconds",
                              lambda x: (x <= 20).sum()),
        buys_20s_to_60s   = ("first_entry_seconds",
                              lambda x: ((x > 20) & (x <= 60)).sum()),
        buys_60s_to_180s  = ("first_entry_seconds",
                              lambda x: ((x > 60) & (x <= 180)).sum()),
    ).reset_index()

    first_90 = (
        eb.groupby("coin_address")["first_entry_seconds"]
        .apply(lambda x: (x <= 90).sum())
        .reset_index(name="buys_first_90s")
    )
    last_90 = (
        eb.groupby("coin_address")["first_entry_seconds"]
        .apply(lambda x: (x > 90).sum())
        .reset_index(name="buys_last_90s")
    )
    speed_block = speed_block.merge(first_90, on="coin_address")
    speed_block = speed_block.merge(last_90,  on="coin_address")
    speed_block["acceleration_ratio"] = (
        speed_block["buys_first_90s"] /
        speed_block["buys_last_90s"].clip(lower=1)
    )
    speed_block.drop(
        columns=["buys_first_90s", "buys_last_90s"], inplace=True
    )

    # Tiempo hasta la quinta compra
    print("  Calculando time_to_5th_buy...")
    def fifth_buy_time(x):
        s = x.sort_values()
        return float(s.iloc[4]) if len(s) >= 5 else np.nan

    time5 = (
        eb.groupby("coin_address")["first_entry_seconds"]
        .apply(fifth_buy_time)
        .reset_index(name="time_to_5th_buy")
    )

    # ── Bloque 5: Contexto temporal ───────────────────────────
    print("  Calculando bloque 5: contexto temporal...")
    coins_time = coins_meta[["coin_address", "created_at"]].copy()
    coins_time["created_at"] = pd.to_datetime(
        coins_time["created_at"], utc=True
    )
    coins_time["hour_utc"]    = coins_time["created_at"].dt.hour.astype("Int64")
    coins_time["day_of_week"] = coins_time["created_at"].dt.dayofweek.astype("Int64")
    coins_time = coins_time[["coin_address", "hour_utc", "day_of_week"]]

    # ── Unir todos los bloques ─────────────────────────────────
    print("  Uniendo bloques...")
    result = wallet_block.copy()
    for df_block in [cooc_df, buy_block, creator_flag,
                     speed_block, time5, coins_time]:
        result = result.merge(df_block, on="coin_address", how="left")

    if "avg_cooccurrence_score" in result.columns:
        result["avg_cooccurrence_score"] = (
            result["avg_cooccurrence_score"].fillna(0.0)
        )
    else:
        result["avg_cooccurrence_score"] = 0.0

    result["creator_is_buyer"] = result["creator_is_buyer"].fillna(False)
    result["calculated_at"]    = datetime.now(timezone.utc)

    num_cols = [
        "avg_wallet_score", "max_wallet_score",
        "pct_high_score_wallets", "pct_negative_wallets", "pct_new_wallets",
        "avg_cooccurrence_score", "total_volume_sol",
        "avg_buy_size_sol", "std_buy_size_sol",
        "concentration_top5", "acceleration_ratio",
    ]
    for col in num_cols:
        if col in result.columns:
            result[col] = result[col].round(6)

    print(f"  Features calculadas para {len(result):,} coins")
    return result


# ── Escritura en base de datos ─────────────────────────────────

def save_features(engine, df: pd.DataFrame):
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
                    pct_high_score_wallets, pct_negative_wallets,
                    pct_new_wallets, avg_cooccurrence_score,
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
                    :pct_high_score_wallets, :pct_negative_wallets,
                    :pct_new_wallets, :avg_cooccurrence_score,
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
            print(f"    {inserted:,} / {len(rows):,}...", end="\r", flush=True)

    print(f"    {len(rows):,} filas guardadas.              ")


# ── Resumen ────────────────────────────────────────────────────

def print_summary(engine):
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM coin_features")
        ).scalar()
        stats = conn.execute(text("""
            SELECT
                ROUND(AVG(n_early_buyers)::numeric, 1)           AS avg_buyers,
                ROUND(AVG(avg_wallet_score)::numeric, 4)          AS avg_wscore,
                ROUND(AVG(pct_new_wallets)::numeric, 4)           AS avg_new,
                ROUND(AVG(acceleration_ratio)::numeric, 2)        AS avg_accel,
                SUM(CASE WHEN creator_is_buyer THEN 1 ELSE 0 END) AS creator_buyers,
                ROUND(AVG(avg_cooccurrence_score)::numeric, 4)    AS avg_cooc
            FROM coin_features
        """)).fetchone()

    print(f"\n  coin_features generadas:   {total:,}")
    if stats and stats[0]:
        print(f"  Avg early buyers:          {stats[0]}")
        print(f"  Avg wallet score:          {stats[1]}")
        print(f"  Avg % wallets nuevas:      {float(stats[2]):.2%}")
        print(f"  Avg acceleration ratio:    {stats[3]}")
        print(f"  Coins con creator buyer:   {stats[4]:,}")
        print(f"  Avg co-ocurrencia:         {stats[5]}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FEATURES — coin_features (vectorizado, sin leakage)")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/4] Cargando datos...")
    eb, cp, coins = load_all_data(engine)

    print("\n[2/4] Determinando split temporal y calculando co-ocurrencias...")
    train_coins = get_train_coins(cp, coins)
    cooc        = compute_cooccurrence(eb, cp, train_coins)

    print("\n[3/4] Calculando features (vectorizado)...")
    features_df = compute_features(eb, cp, coins, cooc)

    if features_df.empty:
        print("Sin features generadas.")
        return

    print("\n[4/4] Guardando en coin_features...")
    save_features(engine, features_df)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print_summary(engine)
    print("=" * 60)


if __name__ == "__main__":
    main()