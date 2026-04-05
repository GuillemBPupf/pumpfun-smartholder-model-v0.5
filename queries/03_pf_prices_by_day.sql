-- ================================================================
-- pumpfun_prices_by_day  (versión extendida)
-- Parámetros: {{block_month}}, {{start_date}}, {{end_date}}, {{end_date_extended}}
--
-- Novedades respecto a la versión anterior:
--   1. price_snapshots   → precio mediana en 12 intervalos estratégicos
--                          (t=5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 180, 240 min)
--   2. seconds_to_2x     → segundos desde lanzamiento hasta 2x el precio_5min
--   3. seconds_to_2_5x   → ídem para 2.5x (take profit objetivo)
--   4. max_drawdown_1h   → mayor caída peak-to-trough en la primera hora
--                          (0 = sin caída, 1 = caída total desde el pico)
--   5. max_drawdown_to_tp → mayor caída desde price_5min hasta alcanzar 2.5x
--                           Solo disponible para coins que llegan a 2.5x.
--                           Si > 0.20 con stop-loss del 20%: el stop se habría
--                           activado antes de llegar al take profit.
--   6. sustained_score   → nº de buckets de 10 min (entre t=5min y t=4h) donde
--                           el precio mínimo >= 1.5x price_5min. Rango: 0-23.
--                           Sustituye el binario sustained_10min (se mantiene
--                           por compatibilidad con código existente).
--
-- El número de filas de output es idéntico al anterior (~2k/día).
-- El coste extra en créditos Dune es bajo: todos los nuevos cálculos
-- operan sobre price_series (ya en memoria), sin re-escanear la tabla base.
-- ================================================================

WITH coin_launches AS (
    SELECT
        token_bought_mint_address    AS coin_address,
        MIN(block_time)              AS created_at
    FROM dex_solana.trades
    WHERE project = 'pumpdotfun'
      AND token_sold_mint_address = 'So11111111111111111111111111111111111111112'
      AND block_month >= DATE '{{block_month}}'
      AND block_time  >= TIMESTAMP '{{start_date}}'
      AND block_time  <  TIMESTAMP '{{end_date}}'
    GROUP BY token_bought_mint_address
),

price_series AS (
    SELECT
        t.token_bought_mint_address                              AS coin_address,
        DATE_DIFF('second', l.created_at, t.block_time)         AS seconds_since_launch,
        t.amount_usd / NULLIF(t.token_bought_amount, 0)         AS price_usd
    FROM dex_solana.trades t
    INNER JOIN coin_launches l
        ON t.token_bought_mint_address = l.coin_address
    WHERE t.project = 'pumpdotfun'
      AND t.token_sold_mint_address = 'So11111111111111111111111111111111111111112'
      AND t.block_month >= DATE '{{block_month}}'
      AND t.block_time  >= TIMESTAMP '{{start_date}}'
      AND t.block_time  <  TIMESTAMP '{{end_date_extended}}'
      AND t.amount_usd > 0
      AND t.token_bought_amount > 0
      AND DATE_DIFF('second', l.created_at, t.block_time) BETWEEN 0 AND 14400
),

-- ── Precio de referencia: mediana entre minuto 4 y 6 ──────────
price_at_5min_calc AS (
    SELECT
        coin_address,
        APPROX_PERCENTILE(price_usd, 0.5) AS price_5min
    FROM price_series
    WHERE seconds_since_launch BETWEEN 240 AND 360
    GROUP BY coin_address
),

-- ── Estadísticas base por coin ────────────────────────────────
base_stats AS (
    SELECT
        ps.coin_address,
        p5.price_5min                                           AS price_at_5min,
        MAX(ps.price_usd)
            FILTER (WHERE ps.seconds_since_launch BETWEEN 300 AND 14400)
                                                                AS price_max_4h,
        MIN(ps.price_usd)
            FILTER (WHERE ps.seconds_since_launch BETWEEN 0 AND 300)
                                                                AS price_min_5min,
        MAX(ps.price_usd)
            FILTER (WHERE ps.seconds_since_launch BETWEEN 0 AND 300)
                                                                AS price_max_5min
    FROM price_series ps
    INNER JOIN price_at_5min_calc p5 ON ps.coin_address = p5.coin_address
    GROUP BY ps.coin_address, p5.price_5min
),

-- ── Snapshots de precio en intervalos estratégicos ────────────
-- Ventana de ±30s por intervalo para capturar suficientes trades.
-- Todos los valores son medianas (APPROX_PERCENTILE 0.5) para
-- minimizar el impacto de trades outlier.
-- Intervalos elegidos para capturar:
--   - Dinámica temprana (5-30 min): donde ocurre la mayor parte del movimiento
--   - Evolución posterior (45-240 min): para distinguir pumps sostenidos
price_snapshots AS (
    SELECT
        coin_address,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN  270 AND  330) AS price_t5,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN  570 AND  630) AS price_t10,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN  870 AND  930) AS price_t15,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 1170 AND 1230) AS price_t20,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 1470 AND 1530) AS price_t25,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 1770 AND 1830) AS price_t30,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 2670 AND 2730) AS price_t45,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 3570 AND 3630) AS price_t60,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 5370 AND 5430) AS price_t90,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 7170 AND 7230) AS price_t120,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 10770 AND 10830) AS price_t180,
        APPROX_PERCENTILE(price_usd, 0.5)
            FILTER (WHERE seconds_since_launch BETWEEN 14340 AND 14400) AS price_t240
    FROM price_series
    GROUP BY coin_address
),

-- ── Tiempo hasta múltiplos clave desde price_at_5min ─────────
-- Solo considera trades DESPUÉS de t=5min (la referencia de precio).
-- NULL si la coin nunca alcanza ese múltiplo en la ventana de 4h.
time_to_multiples AS (
    SELECT
        ps.coin_address,
        MIN(
            CASE WHEN ps.price_usd >= p5.price_5min * 2.0
            THEN ps.seconds_since_launch END
        )                                                       AS seconds_to_2x,
        MIN(
            CASE WHEN ps.price_usd >= p5.price_5min * 2.5
            THEN ps.seconds_since_launch END
        )                                                       AS seconds_to_2_5x
    FROM price_series ps
    INNER JOIN price_at_5min_calc p5 ON ps.coin_address = p5.coin_address
    WHERE ps.seconds_since_launch >= 300
      AND p5.price_5min > 0
    GROUP BY ps.coin_address
),

-- ── Max drawdown en la primera hora ───────────────────────────
-- Paso 1: running max por coin ordenado cronológicamente
drawdown_1h_series AS (
    SELECT
        coin_address,
        price_usd,
        MAX(price_usd) OVER (
            PARTITION BY coin_address
            ORDER BY seconds_since_launch
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                       AS running_max
    FROM price_series
    WHERE seconds_since_launch BETWEEN 0 AND 3600
),

-- Paso 2: mayor caída desde el pico local hasta cualquier punto posterior
-- Valor entre 0 (sin drawdown) y 1 (pérdida total desde el pico)
-- Ejemplo: 0.35 = cayó un 35% desde su máximo en la primera hora
max_drawdown_1h_calc AS (
    SELECT
        coin_address,
        MAX(1.0 - price_usd / NULLIF(running_max, 0))          AS max_drawdown_1h
    FROM drawdown_1h_series
    GROUP BY coin_address
),

-- ── Max drawdown entre t=5min y el take profit ────────────────
-- Mide cuánto cayó el precio (respecto a price_5min) en el camino
-- hacia 2.5x. Solo existe para coins que llegaron a 2.5x.
--
-- Interpretación directa para backtesting:
--   max_drawdown_to_tp > 0.20 → con stop-loss del 20% habríamos
--   salido antes de que la coin llegara al take profit objetivo.
--   Permite calcular la tasa real de "stop-loss activado" sobre
--   las coins que el modelo habría marcado como éxito (label=1).
max_drawdown_to_tp_calc AS (
    SELECT
        ps.coin_address,
        1.0 - MIN(ps.price_usd) / NULLIF(p5.price_5min, 0)     AS max_drawdown_to_tp
    FROM price_series ps
    INNER JOIN price_at_5min_calc p5  ON ps.coin_address = p5.coin_address
    INNER JOIN time_to_multiples ttm  ON ps.coin_address = ttm.coin_address
    WHERE ps.seconds_since_launch >= 300
      AND ttm.seconds_to_2_5x IS NOT NULL
      AND ps.seconds_since_launch <= ttm.seconds_to_2_5x
      AND p5.price_5min > 0
    GROUP BY ps.coin_address
),

-- ── Buckets de 10 minutos entre t=5min y t=4h ─────────────────
buckets AS (
    SELECT
        ps.coin_address,
        FLOOR((ps.seconds_since_launch - 300) / 600)           AS bucket_10min,
        MIN(ps.price_usd)                                       AS min_price_in_bucket,
        p5.price_5min * 1.5                                     AS threshold_price
    FROM price_series ps
    INNER JOIN price_at_5min_calc p5 ON ps.coin_address = p5.coin_address
    WHERE ps.seconds_since_launch BETWEEN 300 AND 14400
      AND p5.price_5min > 0
    GROUP BY
        ps.coin_address,
        FLOOR((ps.seconds_since_launch - 300) / 600),
        p5.price_5min
),

-- ── Sostenibilidad ────────────────────────────────────────────
-- sustained_score : nº de buckets donde precio mínimo >= 1.5x price_5min
--                   Rango 0-23. Más informativo que el binario anterior.
-- is_sustained    : binario equivalente al campo original sustained_10min
--                   Se mantiene para compatibilidad con loader.py y model.py
sustained_score_calc AS (
    SELECT
        coin_address,
        SUM(
            CASE WHEN min_price_in_bucket >= threshold_price THEN 1 ELSE 0 END
        )                                                       AS sustained_score,
        MAX(
            CASE WHEN min_price_in_bucket >= threshold_price THEN 1 ELSE 0 END
        )                                                       AS is_sustained
    FROM buckets
    GROUP BY coin_address
)

-- ── Resultado final: una fila por coin ───────────────────────
SELECT
    b.coin_address,

    -- ── Campos originales (sin cambios) ──────────────────────
    b.price_at_5min,
    b.price_max_4h,
    b.price_min_5min,
    b.price_max_5min,

    CASE
        WHEN b.price_at_5min IS NULL OR b.price_at_5min = 0 THEN NULL
        WHEN b.price_max_4h / b.price_at_5min >= 2.5
             AND s.is_sustained = 1                             THEN 1
        ELSE 0
    END                                                         AS label,

    CASE
        WHEN b.price_at_5min IS NULL OR b.price_at_5min = 0 THEN NULL
        WHEN b.price_max_4h / b.price_at_5min >= 2.5           THEN 1
        ELSE 0
    END                                                         AS label_raw,

    CASE
        WHEN b.price_at_5min > 0
        THEN ROUND(CAST(b.price_max_4h / b.price_at_5min AS DOUBLE), 2)
        ELSE NULL
    END                                                         AS max_multiple,

    CASE
        WHEN b.price_max_5min > 0
         AND b.price_min_5min / b.price_max_5min < 0.2         THEN TRUE
        ELSE FALSE
    END                                                         AS rug_detected,

    -- Compatibilidad con código existente
    s.is_sustained                                              AS sustained_10min,

    -- ── Campos nuevos ─────────────────────────────────────────

    -- Sostenibilidad continua (0-23 buckets de 10 min)
    s.sustained_score,

    -- Snapshots de precio en intervalos estratégicos
    -- NULL si la coin no tenía actividad en esa ventana de ±30s
    ps.price_t5,
    ps.price_t10,
    ps.price_t15,
    ps.price_t20,
    ps.price_t25,
    ps.price_t30,
    ps.price_t45,
    ps.price_t60,
    ps.price_t90,
    ps.price_t120,
    ps.price_t180,
    ps.price_t240,

    -- Velocidad de apreciación desde price_5min
    -- NULL si nunca alcanzó ese múltiplo en las 4h
    ttm.seconds_to_2x,
    ttm.seconds_to_2_5x,

    -- Max drawdown en primera hora (peak-to-trough, 0-1)
    -- NULL si no hay suficientes trades en la primera hora
    ROUND(CAST(dd1h.max_drawdown_1h  AS DOUBLE), 4)             AS max_drawdown_1h,

    -- Max drawdown desde price_5min hasta alcanzar el TP (0-1)
    -- NULL si la coin nunca llegó a 2.5x
    -- > 0.20 → stop-loss del 20% se habría activado antes del TP
    ROUND(CAST(ddtp.max_drawdown_to_tp AS DOUBLE), 4)           AS max_drawdown_to_tp

FROM base_stats b
LEFT JOIN sustained_score_calc    s    ON b.coin_address = s.coin_address
LEFT JOIN price_snapshots         ps   ON b.coin_address = ps.coin_address
LEFT JOIN time_to_multiples       ttm  ON b.coin_address = ttm.coin_address
LEFT JOIN max_drawdown_1h_calc    dd1h ON b.coin_address = dd1h.coin_address
LEFT JOIN max_drawdown_to_tp_calc ddtp ON b.coin_address = ddtp.coin_address