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
-- Columnas originales: price_at_5min → sustained_10min
-- Columnas nuevas (query extendida de Dune):
--   sustained_score    → nº de buckets de 10 min con precio >= 1.5x (rango 0-23)
--   price_t5…t240      → snapshots de precio en intervalos estratégicos
--                        NULL si sin actividad en ventana de ±30s
--   seconds_to_2x      → segundos hasta 2x desde price_at_5min (NULL si no llega)
--   seconds_to_2_5x    → ídem para 2.5x (take profit objetivo)
--   max_drawdown_1h    → mayor caída peak-to-trough en primera hora (0-1)
--   max_drawdown_to_tp → mayor caída desde price_5min hasta alcanzar 2.5x (0-1)
--                        NULL si nunca llegó a 2.5x
--                        > 0.20 → stop-loss 20% habría saltado antes del TP
CREATE TABLE IF NOT EXISTS coin_prices (
    coin_address        TEXT PRIMARY KEY REFERENCES coins(coin_address),

    -- ── Campos originales ─────────────────────────────────────
    price_at_5min       NUMERIC,
    price_max_4h        NUMERIC,
    price_min_5min      NUMERIC,
    price_max_5min      NUMERIC,
    max_multiple        NUMERIC,
    label               SMALLINT,
    label_raw           SMALLINT,
    rug_detected        BOOLEAN,
    sustained_10min     SMALLINT,

    -- ── Sostenibilidad continua ───────────────────────────────
    sustained_score     SMALLINT,

    -- ── Snapshots de precio en intervalos estratégicos ────────
    price_t5            NUMERIC,
    price_t10           NUMERIC,
    price_t15           NUMERIC,
    price_t20           NUMERIC,
    price_t25           NUMERIC,
    price_t30           NUMERIC,
    price_t45           NUMERIC,
    price_t60           NUMERIC,
    price_t90           NUMERIC,
    price_t120          NUMERIC,
    price_t180          NUMERIC,
    price_t240          NUMERIC,

    -- ── Velocidad de apreciación ──────────────────────────────
    seconds_to_2x       NUMERIC,
    seconds_to_2_5x     NUMERIC,

    -- ── Drawdowns ─────────────────────────────────────────────
    max_drawdown_1h     NUMERIC,
    max_drawdown_to_tp  NUMERIC,

    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ── Tabla wallet_metrics ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS wallet_metrics (
    wallet_address      TEXT PRIMARY KEY,
    appearances_total   INTEGER DEFAULT 0,
    win_rate            NUMERIC,
    avg_roi             NUMERIC,
    negative_rate       NUMERIC,
    performance_score   NUMERIC,
    score_reliable      BOOLEAN DEFAULT FALSE,
    last_calculated_at  TIMESTAMPTZ,
    first_seen_at       TIMESTAMPTZ
);

-- ── Tabla coin_features ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS coin_features (
    coin_address             TEXT PRIMARY KEY REFERENCES coins(coin_address),
    calculated_at            TIMESTAMPTZ DEFAULT NOW(),
    n_early_buyers           INTEGER,
    n_reliable_wallets       INTEGER,
    avg_wallet_score         NUMERIC,
    max_wallet_score         NUMERIC,
    pct_high_score_wallets   NUMERIC,
    pct_negative_wallets     NUMERIC,
    pct_new_wallets          NUMERIC,
    avg_cooccurrence_score   NUMERIC,
    total_volume_sol         NUMERIC,
    avg_buy_size_sol         NUMERIC,
    std_buy_size_sol         NUMERIC,
    concentration_top5       NUMERIC,
    n_tier1_buyers           INTEGER,
    creator_is_buyer         BOOLEAN,
    buys_in_first_20s        INTEGER,
    buys_20s_to_60s          INTEGER,
    buys_60s_to_180s         INTEGER,
    acceleration_ratio       NUMERIC,
    time_to_5th_buy          NUMERIC,
    hour_utc                 SMALLINT,
    day_of_week              SMALLINT
);

-- ── Tabla signals ─────────────────────────────────────────────
-- model_score:       probabilidad calibrada (0-1)
-- expected_multiple: múltiplo esperado (regresor XGBoost)
-- ev_score:          Expected Value por unidad apostada
--                    EV = P × 1.395 + (1-P) × (-0.52)
--                    Break-even: P ≈ 27.2% de precisión
-- signal_tier:       'high' (EV>0.30) / 'medium' (EV>0.10) /
--                    'low' (EV>0.00) / NULL (no señal)
-- outcome_label:     resultado real (se rellena tras la coin)
CREATE TABLE IF NOT EXISTS signals (
    id                  BIGSERIAL PRIMARY KEY,
    coin_address        TEXT REFERENCES coins(coin_address),
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    model_score         NUMERIC,
    expected_multiple   NUMERIC,
    ev_score            NUMERIC,
    signal_tier         TEXT,
    outcome_label       SMALLINT,
    outcome_verified_at TIMESTAMPTZ
);

-- ── Índices signals ───────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin_address);
CREATE INDEX IF NOT EXISTS idx_signals_tier ON signals(signal_tier);
CREATE INDEX IF NOT EXISTS idx_signals_ev   ON signals(ev_score);
CREATE INDEX IF NOT EXISTS idx_signals_gen  ON signals(generated_at);

SELECT 'Tablas creadas correctamente' AS status;