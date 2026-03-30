"""
splitter.py
-----------
Módulo compartido que define el split temporal train/test.

Todos los scripts (wallet_scoring, features, model) importan
get_train_test_coins() para garantizar que usan exactamente
el mismo split sin hardcodear fechas ni archivos.

El split se basa en ordenar las coins por created_at y tomar
el 80% más antiguo como train. Funciona con cualquier volumen
de datos.
"""

import pandas as pd
from sqlalchemy.engine import Engine

TRAIN_RATIO = 0.80


def get_train_test_coins(engine: Engine) -> tuple[set, set]:
    """
    Devuelve (train_coins, test_coins) basado en orden temporal.

    Solo incluye coins que tienen label calculada en coin_prices,
    ya que son las únicas utilizables para entrenar y evaluar.

    El split es siempre:
      - train: TRAIN_RATIO más antiguas (por created_at)
      - test:  el resto más recientes
    """
    from sqlalchemy import text

    df = pd.read_sql(
        text("""
            SELECT cp.coin_address, c.created_at
            FROM coin_prices cp
            INNER JOIN coins c ON cp.coin_address = c.coin_address
            WHERE cp.label IS NOT NULL
            ORDER BY c.created_at ASC
        """),
        engine
    )

    if df.empty:
        return set(), set()

    split_idx   = int(len(df) * TRAIN_RATIO)
    train_coins = set(df.iloc[:split_idx]["coin_address"])
    test_coins  = set(df.iloc[split_idx:]["coin_address"])

    return train_coins, test_coins