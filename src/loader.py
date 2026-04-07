"""
loader.py
---------
Carga los CSVs de data/raw/ en PostgreSQL.

Fuentes:
    data/raw/coins/coins_complete.csv       → tabla coins
    data/raw/early_trades/trades_*.csv      → tabla early_buyers
    data/raw/prices/prices_*.csv            → tabla coin_prices

Uso:
    python src/loader.py

Maneja duplicados con ON CONFLICT DO NOTHING.
Si se ejecuta varias veces, ignora lo que ya está cargado.

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import glob
import pandas as pd
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


# ── Helpers ────────────────────────────────────────────────────

def insert_batches(conn, sql, df, batch_size=10_000, label="filas"):
    """Inserta un DataFrame en lotes mostrando progreso."""
    total    = len(df)
    inserted = 0
    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size]
        conn.execute(text(sql), batch.to_dict(orient="records"))
        conn.commit()
        inserted += len(batch)
        print(f"    {inserted:,} / {total:,} {label}...", end="\r", flush=True)
    print(f"    {total:,} {label} procesadas.              ")
    return total


def get_valid_coins(engine):
    """Devuelve el conjunto de coin_addresses que existen en la BD."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT coin_address FROM coins"))
        return set(row[0] for row in result)


def clean_nil(df):
    """
    Dune devuelve '<nil>' como string para valores nulos.
    Esta función lo convierte a None para que PostgreSQL lo acepte.
    """
    return df.replace("<nil>", None)


# ── Carga de coins ─────────────────────────────────────────────

def load_coins(engine):
    files = glob.glob("data/raw/coins/coins_complete.csv")
    if not files:
        files = sorted(glob.glob("data/raw/coins/coins_2*.csv"))
    if not files:
        print("  No se encontraron archivos de coins.")
        return 0

    total = 0
    for filepath in files:
        print(f"  Leyendo {filepath}...")
        df = pd.read_csv(filepath)
        df = clean_nil(df)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df[["coin_address", "created_at", "creator_wallet"]].copy()
        df = df.dropna(subset=["coin_address"]).drop_duplicates(subset="coin_address")

        print(f"  {len(df):,} coins a insertar...")
        with engine.connect() as conn:
            total += insert_batches(
                conn,
                """
                INSERT INTO coins (coin_address, created_at, creator_wallet)
                VALUES (:coin_address, :created_at, :creator_wallet)
                ON CONFLICT (coin_address) DO NOTHING
                """,
                df,
                batch_size=10_000,
                label="coins"
            )
    return total


# ── Carga de early buyers ──────────────────────────────────────

def load_early_buyers(engine):
    """
    Carga trades_YYYYMMDD.csv en early_buyers.
    Columnas esperadas (query agregada de Dune):
        coin_address, wallet_address, first_entry_seconds,
        total_sol_spent, total_usd_spent, n_trades, tier
    """
    files = sorted(glob.glob("data/raw/early_trades/trades_2*.csv"))
    if not files:
        print("  No se encontraron archivos de trades.")
        return 0

    valid_coins = get_valid_coins(engine)
    print(f"  Coins válidas en BD: {len(valid_coins):,}")

    total = 0
    for filepath in files:
        print(f"  Leyendo {filepath}...")
        df = pd.read_csv(filepath)
        df = clean_nil(df)

        rename_map = {}
        if "first_entry_seconds" not in df.columns and "seconds_since_launch" in df.columns:
            rename_map["seconds_since_launch"] = "first_entry_seconds"
        if "total_sol_spent" not in df.columns and "amount_sol" in df.columns:
            rename_map["amount_sol"] = "total_sol_spent"
        if "total_usd_spent" not in df.columns and "amount_usd" in df.columns:
            rename_map["amount_usd"] = "total_usd_spent"
        if rename_map:
            df = df.rename(columns=rename_map)

        for col, default in [("total_usd_spent", 0.0), ("n_trades", 1), ("tier", 2)]:
            if col not in df.columns:
                df[col] = default

        df = df.dropna(subset=["coin_address", "wallet_address"])
        df = df.drop_duplicates(subset=["coin_address", "wallet_address"])

        before = len(df)
        df     = df[df["coin_address"].isin(valid_coins)]
        if before - len(df) > 0:
            print(f"    {before - len(df):,} filas descartadas (coin no en BD)")

        if df.empty:
            print("    Sin datos válidos, saltando.")
            continue

        cols = [
            "coin_address", "wallet_address", "first_entry_seconds",
            "total_sol_spent", "total_usd_spent", "n_trades", "tier"
        ]
        df = df[[c for c in cols if c in df.columns]]

        print(f"  {len(df):,} early buyers a insertar...")
        with engine.connect() as conn:
            total += insert_batches(
                conn,
                """
                INSERT INTO early_buyers (
                    coin_address, wallet_address, first_entry_seconds,
                    total_sol_spent, total_usd_spent, n_trades, tier
                )
                VALUES (
                    :coin_address, :wallet_address, :first_entry_seconds,
                    :total_sol_spent, :total_usd_spent, :n_trades, :tier
                )
                ON CONFLICT (coin_address, wallet_address) DO NOTHING
                """,
                df,
                batch_size=5_000,
                label="early buyers"
            )
    return total


# ── Carga de precios ───────────────────────────────────────────

# Columnas esperadas en los CSVs de la query extendida de Dune.
# Orden fijo para mantener sincronizados loader.py y create_tables.sql.
PRICE_COLS = [
    "coin_address",
    "price_at_5min", "price_max_4h", "price_min_5min", "price_max_5min",
    "max_multiple", "label", "label_raw", "rug_detected", "sustained_10min",
    "sustained_score",
    "price_t5", "price_t10", "price_t15", "price_t20", "price_t25",
    "price_t30", "price_t45", "price_t60", "price_t90",
    "price_t120", "price_t180", "price_t240",
    "seconds_to_2x", "seconds_to_2_5x",
    "max_drawdown_1h", "max_drawdown_to_tp",
]


def load_prices(engine):
    """
    Carga prices_YYYYMMDD.csv en coin_prices.
    Todos los CSVs deben haber sido descargados con la query extendida.
    """
    files = sorted(glob.glob("data/raw/prices/prices_2*.csv"))
    if not files:
        print("  No se encontraron archivos de prices.")
        return 0

    valid_coins = get_valid_coins(engine)

    total = 0
    for filepath in files:
        print(f"  Leyendo {filepath}...")
        df = pd.read_csv(filepath)
        df = clean_nil(df)
        df = df.dropna(subset=["coin_address"])
        df = df.drop_duplicates(subset="coin_address")

        before = len(df)
        df     = df[df["coin_address"].isin(valid_coins)]
        if before - len(df) > 0:
            print(f"    {before - len(df):,} filas descartadas (coin no en BD)")

        if df.empty:
            continue

        # Verificar que el CSV tiene todas las columnas esperadas
        missing = [c for c in PRICE_COLS if c not in df.columns]
        if missing:
            print(f"    ⚠ Columnas ausentes: {missing}")
            print(f"    Asegúrate de usar la query extendida de Dune. Saltando.")
            continue

        # Normalizar rug_detected a bool
        df["rug_detected"] = df["rug_detected"].map(
            {True: True, False: False, "true": True, "false": False,
             "True": True, "False": False, 1: True, 0: False}
        ).fillna(False)

        df = df[PRICE_COLS]

        print(f"  {len(df):,} precios a insertar...")
        with engine.connect() as conn:
            total += insert_batches(
                conn,
                """
                INSERT INTO coin_prices (
                    coin_address,
                    price_at_5min, price_max_4h, price_min_5min, price_max_5min,
                    max_multiple, label, label_raw, rug_detected, sustained_10min,
                    sustained_score,
                    price_t5, price_t10, price_t15, price_t20, price_t25,
                    price_t30, price_t45, price_t60, price_t90,
                    price_t120, price_t180, price_t240,
                    seconds_to_2x, seconds_to_2_5x,
                    max_drawdown_1h, max_drawdown_to_tp
                )
                VALUES (
                    :coin_address,
                    :price_at_5min, :price_max_4h, :price_min_5min, :price_max_5min,
                    :max_multiple, :label, :label_raw, :rug_detected, :sustained_10min,
                    :sustained_score,
                    :price_t5, :price_t10, :price_t15, :price_t20, :price_t25,
                    :price_t30, :price_t45, :price_t60, :price_t90,
                    :price_t120, :price_t180, :price_t240,
                    :seconds_to_2x, :seconds_to_2_5x,
                    :max_drawdown_1h, :max_drawdown_to_tp
                )
                ON CONFLICT (coin_address) DO NOTHING
                """,
                df,
                batch_size=10_000,
                label="precios"
            )
    return total


# ── Verificación ───────────────────────────────────────────────

def verify(engine):
    tables = {
        "coins":          "SELECT COUNT(*) FROM coins",
        "early_buyers":   "SELECT COUNT(*) FROM early_buyers",
        "coin_prices":    "SELECT COUNT(*) FROM coin_prices",
        "wallet_metrics": "SELECT COUNT(*) FROM wallet_metrics",
        "coin_features":  "SELECT COUNT(*) FROM coin_features",
        "signals":        "SELECT COUNT(*) FROM signals",
    }
    print()
    with engine.connect() as conn:
        for table, query in tables.items():
            n = conn.execute(text(query)).scalar()
            print(f"  {table:<20} {n:>12,} filas")

    with engine.connect() as conn:
        n_label1  = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE label = 1")).scalar()
        n_label0  = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE label = 0")).scalar()
        n_null    = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE label IS NULL")).scalar()
        n_rug     = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE rug_detected = TRUE")).scalar()
        n_has_tp  = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE seconds_to_2_5x IS NOT NULL")).scalar()
        n_dd_tp   = conn.execute(text("SELECT COUNT(*) FROM coin_prices WHERE max_drawdown_to_tp > 0.20")).scalar()

    print(f"\n  Labels en coin_prices:")
    print(f"    label=1 (éxito):         {n_label1:>10,}")
    print(f"    label=0 (fracaso):        {n_label0:>10,}")
    print(f"    label=NULL:               {n_null:>10,}")
    print(f"    rugs detectados:          {n_rug:>10,}")
    print(f"\n  Columnas extendidas:")
    print(f"    Con seconds_to_2_5x:      {n_has_tp:>10,}")
    print(f"    DD > 20% antes del TP:    {n_dd_tp:>10,}  "
          f"(stop-loss habría saltado)")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LOADER — PostgreSQL")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/3] Cargando coins...")
    load_coins(engine)

    print("\n[2/3] Cargando early buyers...")
    load_early_buyers(engine)

    print("\n[3/3] Cargando precios...")
    load_prices(engine)

    print("\n" + "=" * 60)
    print("VERIFICACIÓN FINAL")
    verify(engine)
    print("=" * 60)


if __name__ == "__main__":
    main()