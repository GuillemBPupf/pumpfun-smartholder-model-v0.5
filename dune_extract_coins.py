"""
dune_extract_coins.py
---------------------
Extrae la lista completa de coins de pump.fun de los últimos 3 meses
usando la API de Dune Analytics, semana a semana.

Uso:
    python dune_extract_coins.py

Resultado:
    - Un CSV por semana en data/raw/coins/coins_YYYYMMDD.csv
    - Un archivo final consolidado: data/raw/coins/coins_complete.csv

Si el script se interrumpe, al volver a ejecutarlo salta
automáticamente las semanas que ya están descargadas.

Requisitos en .env:
    DUNE_API_KEY=...
    DUNE_QUERY_ID=...   (número de la query en dune.com/queries/XXXXXX)
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from dune_client.types import QueryParameter

load_dotenv()

# ── Configuración ──────────────────────────────────────────────

DUNE_API_KEY  = os.getenv("DUNE_API_KEY")
DUNE_QUERY_ID = int(os.getenv("DUNE_QUERY_ID", "0"))
OUTPUT_DIR    = "data/raw/coins"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Semanas a procesar ─────────────────────────────────────────
# Formato: (block_month, start_date, end_date)
# block_month = primer día del mes que cubre el inicio de la semana

WEEKS = [
    ("2025-12-01", "2025-12-23", "2025-12-30"),
    ("2025-12-01", "2025-12-30", "2026-01-06"),
    ("2026-01-01", "2026-01-06", "2026-01-13"),
    ("2026-01-01", "2026-01-13", "2026-01-20"),
    ("2026-01-01", "2026-01-20", "2026-01-27"),
    ("2026-01-01", "2026-01-27", "2026-02-03"),
    ("2026-02-01", "2026-02-03", "2026-02-10"),
    ("2026-02-01", "2026-02-10", "2026-02-17"),
    ("2026-02-01", "2026-02-17", "2026-02-24"),
    ("2026-02-01", "2026-02-24", "2026-03-03"),
    ("2026-03-01", "2026-03-03", "2026-03-10"),
    ("2026-03-01", "2026-03-10", "2026-03-17"),
    ("2026-03-01", "2026-03-17", "2026-03-25"),
]

# ── Funciones ──────────────────────────────────────────────────

def filename_for_week(start_date: str) -> str:
    return os.path.join(OUTPUT_DIR, f"coins_{start_date.replace('-', '')}.csv")


def extract_week(client, block_month, start_date, end_date):
    query = QueryBase(
        query_id=DUNE_QUERY_ID,
        params=[
            QueryParameter.text_type("block_month", block_month),
            QueryParameter.text_type("start_date",  start_date),
            QueryParameter.text_type("end_date",    end_date),
        ]
    )
    try:
        df = client.run_query_dataframe(query)
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def concatenate_all():
    files = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("coins_2") and f.endswith(".csv")
    ])

    if not files:
        print("No se encontraron archivos semanales.")
        return ""

    print(f"\nConsolidando {len(files)} archivos...")
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    combined["created_at"] = pd.to_datetime(combined["created_at"], utc=True)
    combined = (
        combined
        .sort_values("created_at")
        .drop_duplicates(subset="coin_address", keep="first")
        .reset_index(drop=True)
    )

    out = os.path.join(OUTPUT_DIR, "coins_complete.csv")
    combined.to_csv(out, index=False)
    return out


# ── Pipeline principal ─────────────────────────────────────────

def main():
    if not DUNE_API_KEY:
        print("ERROR: DUNE_API_KEY no encontrada en .env")
        return
    if DUNE_QUERY_ID == 0:
        print("ERROR: DUNE_QUERY_ID no configurado en .env")
        print("Entra en Dune, crea la query y copia el número de la URL.")
        return

    print("=" * 60)
    print("DUNE COINS EXTRACTOR — pump.fun")
    print(f"Query ID : {DUNE_QUERY_ID}")
    print(f"Semanas  : {len(WEEKS)}")
    print("=" * 60)

    client      = DuneClient(api_key=DUNE_API_KEY)
    total_coins = 0
    errores     = []

    for i, (block_month, start_date, end_date) in enumerate(WEEKS):
        filepath = filename_for_week(start_date)

        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            n = len(df_existing)
            total_coins += n
            print(f"[{i+1:>2}/{len(WEEKS)}] {start_date} → {end_date} "
                  f"| YA EXISTE ({n:>7,} coins)")
            continue

        print(f"[{i+1:>2}/{len(WEEKS)}] {start_date} → {end_date} | ejecutando...",
              end=" ", flush=True)

        df = extract_week(client, block_month, start_date, end_date)

        if df is None or df.empty:
            print("SIN RESULTADOS")
            errores.append(start_date)
            time.sleep(30)
            continue

        df.to_csv(filepath, index=False)
        total_coins += len(df)
        print(f"{len(df):>7,} coins → {filepath}")

        if i < len(WEEKS) - 1:
            time.sleep(15)

    out = concatenate_all()

    print("\n" + "=" * 60)
    print("RESUMEN")
    print(f"  Semanas OK    : {len(WEEKS) - len(errores)}/{len(WEEKS)}")
    print(f"  Total coins   : {total_coins:,}")
    if out:
        df_final = pd.read_csv(out)
        print(f"  Coins únicas  : {len(df_final):,}")
        print(f"  Archivo final : {out}")
    if errores:
        print(f"  Semanas error : {errores}")
        print("  Vuelve a ejecutar para reintentar.")
    print("=" * 60)


if __name__ == "__main__":
    main()