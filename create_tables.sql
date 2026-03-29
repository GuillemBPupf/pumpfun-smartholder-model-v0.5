-- =============================================================
-- create_tables.sql
-- Ejecutar con:
--   psql -U postgres -d pumpfun -f create_tables.sql
-- =============================================================

-- ── Tabla coins ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coins (
    coin_address        TEXT PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL,
    creator_wallet      TEXT,
    price_at_5min       NUMERIC,
    price_max_4h        NUMERIC,
    price_max_sustained NUMERIC,
    max_sustained_start TIMESTAMPTZ,
    label               SMALLINT,
    rug_detected        BOOLEAN DEFAULT FALSE,
    prices_loaded       BOOLEAN DEFAULT FALSE,
    label_calculated    BOOLEAN DEFAULT FALSE,
    created_in_db_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Tabla early_buyers ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS early_buyers (
    id                   BIGSERIAL PRIMARY KEY,
    coin_address         TEXT NOT NULL REFERENCES coins(coin_address),
    wallet_address       TEXT NOT NULL,
    first_entry_seconds  NUMERIC,
    total_sol_spent      NUMERIC,
    total_usd_spent      NUMERIC,
    n_trades             INTEGER,
    tier                 SMALLINT,
    UNIQUE (coin_address, wallet_address)
);

CREATE INDEX IF NOT EXISTS idx_eb_coin    ON early_buyers(coin_address);
CREATE INDEX IF NOT EXISTS idx_eb_wallet  ON early_buyers(wallet_address);
CREATE INDEX IF NOT EXISTS idx_eb_tier    ON early_buyers(tier);
CREATE INDEX IF NOT EXISTS idx_eb_entry   ON early_buyers(first_entry_seconds);

-- ── Tabla coin_prices ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coin_prices (
    coin_address      TEXT PRIMARY KEY REFERENCES coins(coin_address),
    price_at_5min     NUMERIC,
    price_max_4h      NUMERIC,
    price_min_5min    NUMERIC,
    price_max_5min    NUMERIC,
    max_multiple      NUMERIC,
    label             SMALLINT,
    label_raw         SMALLINT,
    rug_detected      BOOLEAN,
    sustained_10min   SMALLINT,
    loaded_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ── Tabla wallet_metrics ──────────────────────────────────────
-- Score de cada wallet calculado sobre el histórico completo.
CREATE TABLE IF NOT EXISTS wallet_metrics (
    wallet_address      TEXT PRIMARY KEY,
    appearances_total   INTEGER DEFAULT 0,   -- apariciones totales en el dataset
    win_rate            NUMERIC,             -- % coins exitosas (label=1) sobre total
    avg_roi             NUMERIC,             -- ROI medio sobre coins con precio calculado
    negative_rate       NUMERIC,             -- % coins con rug_detected=TRUE sobre total
    performance_score   NUMERIC,             -- score final entre 0 y 1
    score_reliable      BOOLEAN DEFAULT FALSE, -- TRUE si appearances_total >= 5
    last_calculated_at  TIMESTAMPTZ,
    first_seen_at       TIMESTAMPTZ
);

-- ── Tabla coin_features ───────────────────────────────────────
-- Features calculadas por coin para el modelo de ML.
CREATE TABLE IF NOT EXISTS coin_features (
    coin_address             TEXT PRIMARY KEY REFERENCES coins(coin_address),
    calculated_at            TIMESTAMPTZ DEFAULT NOW(),

    -- Composición de wallets
    n_early_buyers           INTEGER,
    n_reliable_wallets       INTEGER,
    avg_wallet_score         NUMERIC,
    max_wallet_score         NUMERIC,
    pct_high_score_wallets   NUMERIC,   -- % wallets con score > 0.6
    pct_negative_wallets     NUMERIC,   -- % wallets con negative_rate > 0.4
    pct_new_wallets          NUMERIC,   -- % wallets sin historial (appearances_total < 3)

    -- Co-ocurrencia
    avg_cooccurrence_score   NUMERIC,   -- media de veces que pares de wallets han
                                        -- coincidido en coins exitosas previas

    -- Comportamiento de compra
    total_volume_sol         NUMERIC,
    avg_buy_size_sol         NUMERIC,
    std_buy_size_sol         NUMERIC,   -- desviación estándar (baja = posible bot)
    concentration_top5       NUMERIC,   -- % volumen en top 5 wallets por SOL gastado
    n_tier1_buyers           INTEGER,   -- wallets que entraron en primeros 20s
    creator_is_buyer         BOOLEAN,   -- TRUE si creator_wallet aparece en early_buyers

    -- Velocidad de acumulación
    buys_in_first_20s        INTEGER,
    buys_20s_to_60s          INTEGER,
    buys_60s_to_180s         INTEGER,
    acceleration_ratio       NUMERIC,   -- buys_first_90s / max(buys_last_90s, 1)
    time_to_5th_buy          NUMERIC,   -- segundos hasta la quinta compra

    -- Contexto temporal
    hour_utc                 SMALLINT,  -- hora UTC del lanzamiento (0-23)
    day_of_week              SMALLINT   -- día semana (0=lunes, 6=domingo)
);

-- ── Tabla signals ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id                  BIGSERIAL PRIMARY KEY,
    coin_address        TEXT REFERENCES coins(coin_address),
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    model_score         NUMERIC,          -- probabilidad de éxito (0-1)
    expected_multiple   NUMERIC,          -- múltiplo esperado (modelo de regresión)
    signal_tier         TEXT,             -- 'high', 'medium', 'low'
    outcome_label       SMALLINT,         -- resultado real (se rellena después)
    outcome_verified_at TIMESTAMPTZ
);

SELECT 'Tablas creadas correctamente' AS status;