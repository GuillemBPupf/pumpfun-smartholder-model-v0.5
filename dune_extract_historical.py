"""
dune_extract_historical.py
--------------------------
Extrae early trades y precios históricos de pump.fun desde Dune Analytics.
Procesa día a día para evitar timeouts y rota entre múltiples API keys
solo cuando una se queda SIN CRÉDITOS (402).

Los errores de rate limit (429) se manejan con espera y reintento
sobre la MISMA key, sin rotarla.

Si se interrumpe, al volver a ejecutar continúa desde donde se paró.

Uso:
    python dune_extract_historical.py

Resultado:
    data/raw/early_trades/trades_YYYYMMDD.csv
    data/raw/prices/prices_YYYYMMDD.csv

Requisitos en .env:
    DUNE_API_KEY_1=...
    DUNE_API_KEY_2=...
    DUNE_QUERY_ID_TRADES=...
    DUNE_QUERY_ID_PRICES=...
"""

import os
import time
import logging
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from dune_client.types import QueryParameter

load_dotenv()

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dune_extraction.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────

API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("DUNE_API_KEY_") and v.strip()
]

QUERY_ID_TRADES = int(os.getenv("DUNE_QUERY_ID_TRADES", "0"))
QUERY_ID_PRICES = int(os.getenv("DUNE_QUERY_ID_PRICES", "0"))

DIR_TRADES = "data/raw/early_trades"
DIR_PRICES = "data/raw/prices"

os.makedirs(DIR_TRADES, exist_ok=True)
os.makedirs(DIR_PRICES, exist_ok=True)

START_DATE = date(2025, 12, 23)
END_DATE   = date(2026, 3, 25)

# Pausa entre días (segundos)
PAUSE_BETWEEN_DAYS = 12

# Pausa cuando hay rate limit 429 (segundos) — espera y reintenta misma key
PAUSE_ON_RATE_LIMIT = 60

# Pausa cuando hay error genérico (segundos)
PAUSE_ON_GENERIC_ERROR = 30

# Máximo de reintentos por rate limit antes de rendirse con esa query
MAX_RATE_LIMIT_RETRIES = 5


# ── Clasificación de errores ───────────────────────────────────

def classify_error(error: Exception) -> str:
    """
    Clasifica el error para decidir qué hacer:
    - 'credit'      → 402, sin créditos → rotar key
    - 'rate_limit'  → 429, demasiadas requests → esperar y reintentar misma key
    - 'generic'     → otro error → reintentar misma key con pausa corta
    """
    msg = str(error).lower()

    # Error de créditos agotados
    if "402" in msg or "payment required" in msg:
        return "credit"

    # Error de rate limit
    if (
        "429" in msg
        or "too many" in msg
        or "rate limit" in msg
        or "too many 429" in msg
        or "responseError('too many 429" in str(error)
    ):
        return "rate_limit"

    return "generic"


# ── Gestión de API keys ────────────────────────────────────────

class KeyRotator:
    """
    Pool de API keys. Solo rota cuando hay error 402 (sin créditos).
    Los errores 429 (rate limit) NO causan rotación.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError(
                "No se encontraron API keys en .env\n"
                "Añade DUNE_API_KEY_1=..., DUNE_API_KEY_2=..., etc."
            )
        self.keys        = keys
        self.current_idx = 0
        self.exhausted   = set()  # keys que dieron 402
        log.info(f"Pool de API keys: {len(keys)} keys disponibles")

    @property
    def current_key(self) -> str:
        return self.keys[self.current_idx]

    def get_client(self) -> DuneClient:
        return DuneClient(api_key=self.current_key)

    def rotate_on_credit_exhaustion(self) -> bool:
        """
        Llama SOLO cuando hay error 402 (créditos agotados).
        Busca la siguiente key no agotada.
        Devuelve True si encontró una, False si todas están agotadas.
        """
        self.exhausted.add(self.current_idx)
        log.warning(
            f"  Key {self.current_idx + 1} sin créditos (402) → "
            f"buscando key alternativa..."
        )

        for i in range(len(self.keys)):
            if i not in self.exhausted:
                self.current_idx = i
                log.warning(f"  Rotando a key {i + 1}")
                return True

        log.error("Todas las API keys están sin créditos.")
        return False

    @property
    def all_exhausted(self) -> bool:
        return len(self.exhausted) >= len(self.keys)


# ── Generación de días ─────────────────────────────────────────

def generate_days(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current < end:
        days.append(current)
        current += timedelta(days=1)
    return days


def block_month_for(d: date) -> str:
    return d.replace(day=1).strftime("%Y-%m-%d")


# ── Ejecución de queries ───────────────────────────────────────

def run_query(
    rotator: KeyRotator,
    query_id: int,
    params: dict,
) -> pd.DataFrame | None:
    """
    Ejecuta una query de Dune con gestión diferenciada de errores:
    - 402 → rota la key
    - 429 → espera PAUSE_ON_RATE_LIMIT segundos y reintenta misma key
    - Otros → reintenta con pausa corta
    Devuelve DataFrame o None si no se puede recuperar.
    """
    dune_params = [
        QueryParameter.text_type(k, v)
        for k, v in params.items()
    ]

    rate_limit_retries = 0

    while True:
        try:
            client = rotator.get_client()
            query  = QueryBase(query_id=query_id, params=dune_params)
            df     = client.run_query_dataframe(query)
            return df

        except Exception as e:
            error_type = classify_error(e)

            if error_type == "credit":
                # 402: esta key no tiene créditos → rotar
                log.warning(f"  Error 402 (sin créditos): rotando key...")
                if not rotator.rotate_on_credit_exhaustion():
                    return None  # todas las keys agotadas
                # Reintentar inmediatamente con la nueva key
                continue

            elif error_type == "rate_limit":
                # 429: rate limit → esperar y reintentar MISMA key
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RATE_LIMIT_RETRIES:
                    log.error(
                        f"  429 rate limit: superado el máximo de reintentos "
                        f"({MAX_RATE_LIMIT_RETRIES}). Marcando como fallo."
                    )
                    return None
                log.warning(
                    f"  Error 429 (rate limit) — reintento {rate_limit_retries}/"
                    f"{MAX_RATE_LIMIT_RETRIES} en {PAUSE_ON_RATE_LIMIT}s "
                    f"(misma key, no rotamos)"
                )
                time.sleep(PAUSE_ON_RATE_LIMIT)
                continue

            else:
                # Error genérico: reintentar con pausa corta
                log.warning(f"  Error genérico: {e}")
                log.warning(f"  Reintentando en {PAUSE_ON_GENERIC_ERROR}s...")
                time.sleep(PAUSE_ON_GENERIC_ERROR)
                continue


# ── Procesamiento por día ──────────────────────────────────────

def filepath_trades(d: date) -> str:
    return os.path.join(DIR_TRADES, f"trades_{d.strftime('%Y%m%d')}.csv")


def filepath_prices(d: date) -> str:
    return os.path.join(DIR_PRICES, f"prices_{d.strftime('%Y%m%d')}.csv")


# CAMBIO TEMPORAL
def process_day_trades(rotator: KeyRotator, d: date) -> int | None:
    fp = filepath_trades(d)
    if os.path.exists(fp):
        n = 1
        #n = len(pd.read_csv(fp))
        log.info(f"  TRADES {d} | YA EXISTE ({n:,} filas)")
        return n

    end_extended = (d + timedelta(days=2)).strftime("%Y-%m-%d")
    params = {
        "block_month":       block_month_for(d),
        "start_date":        d.strftime("%Y-%m-%d"),
        "end_date":          (d + timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date_extended": end_extended,
    }

    df = run_query(rotator, QUERY_ID_TRADES, params)

    if df is None:
        log.error(f"  TRADES {d} | FALLO — se reintentará en la próxima ejecución")
        return None

    if df.empty:
        df.to_csv(fp, index=False)
        log.info(f"  TRADES {d} | 0 filas")
        return 0

    df.to_csv(fp, index=False)
    log.info(f"  TRADES {d} | {len(df):,} filas → {fp}")
    return len(df)


def process_day_prices(rotator: KeyRotator, d: date) -> int | None:
    fp = filepath_prices(d)
    if os.path.exists(fp):
        n = len(pd.read_csv(fp))
        log.info(f"  PRICES {d} | YA EXISTE ({n:,} filas)")
        return n

    end_extended = (d + timedelta(days=2)).strftime("%Y-%m-%d")
    params = {
        "block_month":       block_month_for(d),
        "start_date":        d.strftime("%Y-%m-%d"),
        "end_date":          (d + timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date_extended": end_extended,
    }

    df = run_query(rotator, QUERY_ID_PRICES, params)

    if df is None:
        log.error(f"  PRICES {d} | FALLO — se reintentará en la próxima ejecución")
        return None

    if df.empty:
        df.to_csv(fp, index=False)
        log.info(f"  PRICES {d} | 0 filas")
        return 0

    df.to_csv(fp, index=False)
    log.info(f"  PRICES {d} | {len(df):,} filas → {fp}")
    return len(df)


# ── Consolidación ──────────────────────────────────────────────

def consolidate(directory: str, prefix: str) -> str:
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".csv")
        and f != f"{prefix}_complete.csv"
    ])

    if not files:
        return ""

    log.info(f"Consolidando {len(files)} archivos de {prefix}...")
    dfs = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]

    if not dfs:
        return ""

    combined = pd.concat(dfs, ignore_index=True)
    out = os.path.join(directory, f"{prefix}_complete.csv")
    combined.to_csv(out, index=False)
    return out


# ── Pipeline principal ─────────────────────────────────────────

def main():
    if not API_KEYS:
        log.error("No se encontraron API keys en .env")
        return
    if QUERY_ID_TRADES == 0:
        log.error("DUNE_QUERY_ID_TRADES no configurado en .env")
        return
    if QUERY_ID_PRICES == 0:
        log.error("DUNE_QUERY_ID_PRICES no configurado en .env")
        return

    days    = generate_days(START_DATE, END_DATE)
    rotator = KeyRotator(API_KEYS)

    log.info("=" * 60)
    log.info("DUNE HISTORICAL EXTRACTOR — pump.fun")
    log.info(f"Rango     : {START_DATE} → {END_DATE}")
    log.info(f"Días      : {len(days)}")
    log.info(f"API keys  : {len(API_KEYS)}")
    log.info(f"Query trades : {QUERY_ID_TRADES}")
    log.info(f"Query prices : {QUERY_ID_PRICES}")
    log.info("=" * 60)

    total_trades = 0
    total_prices = 0
    dias_ok      = 0
    dias_error   = []

    for i, d in enumerate(days):
        log.info(f"\n[Día {i+1:>3}/{len(days)}] {d}")

        # ── Trades ──
        n_trades = process_day_trades(rotator, d)
        if n_trades is None:
            dias_error.append(str(d))
            if rotator.all_exhausted:
                log.error("Todas las keys sin créditos. Deteniéndose.")
                break
        else:
            total_trades += n_trades

        # Pausa entre las dos queries del mismo día
        time.sleep(5)

        # ── Prices ──
        n_prices = process_day_prices(rotator, d)
        if n_prices is None:
            if str(d) not in dias_error:
                dias_error.append(str(d))
            if rotator.all_exhausted:
                log.error("Todas las keys sin créditos. Deteniéndose.")
                break
        else:
            total_prices += n_prices
            dias_ok += 1

        if i < len(days) - 1:
            time.sleep(PAUSE_BETWEEN_DAYS)

    log.info("\nConsolidando archivos...")
    out_trades = consolidate(DIR_TRADES, "trades")
    out_prices = consolidate(DIR_PRICES, "prices")

    # Resumen
    log.info("\n" + "=" * 60)
    log.info("RESUMEN FINAL")
    log.info(f"  Días OK          : {dias_ok}/{len(days)}")
    log.info(f"  Total trades     : {total_trades:,}")
    log.info(f"  Total prices     : {total_prices:,}")

    if out_trades:
        df_t = pd.read_csv(out_trades)
        log.info(f"  Trades únicos    : {len(df_t):,}")
        log.info(f"  Archivo trades   : {out_trades}")
    if out_prices:
        df_p = pd.read_csv(out_prices)
        log.info(f"  Prices únicos    : {len(df_p):,}")
        log.info(f"  Archivo prices   : {out_prices}")
    if dias_error:
        log.warning(f"  Días con error   : {dias_error}")
        log.warning("  Vuelve a ejecutar para reintentar.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()