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

-- Precio de referencia: mediana entre minuto 4 y 6
price_at_5min_calc AS (
    SELECT
        coin_address,
        APPROX_PERCENTILE(price_usd, 0.5) AS price_5min
    FROM price_series
    WHERE seconds_since_launch BETWEEN 240 AND 360
    GROUP BY coin_address
),

-- Estadísticas base por coin
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

-- Buckets de 10 minutos entre t=5min y t=4h
-- Cada bucket recoge el precio mínimo observado en esa ventana
-- Si el mínimo de un bucket >= 1.5x precio_5min → esa ventana fue sostenida
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

-- Por coin: ¿existe algún bucket donde el mínimo esté por encima del umbral?
-- Si sí → la coin mantuvo precio >= 1.5x durante al menos 10 minutos completos
sustainability AS (
    SELECT
        coin_address,
        MAX(
            CASE WHEN min_price_in_bucket >= threshold_price THEN 1 ELSE 0 END
        )                                                       AS is_sustained
    FROM buckets
    GROUP BY coin_address
)

-- Resultado final: una fila por coin con todos los datos necesarios
SELECT
    b.coin_address,
    b.price_at_5min,
    b.price_max_4h,
    b.price_min_5min,
    b.price_max_5min,

    -- Label definitiva (incorpora condición de sostenibilidad)
    CASE
        WHEN b.price_at_5min IS NULL OR b.price_at_5min = 0 THEN NULL
        WHEN b.price_max_4h / b.price_at_5min >= 2.5
             AND s.is_sustained = 1                             THEN 1
        ELSE 0
    END                                                         AS label,

    -- Label sin filtro de sostenibilidad (útil para comparar)
    CASE
        WHEN b.price_at_5min IS NULL OR b.price_at_5min = 0 THEN NULL
        WHEN b.price_max_4h / b.price_at_5min >= 2.5           THEN 1
        ELSE 0
    END                                                         AS label_raw,

    -- Múltiplo máximo alcanzado (útil para análisis)
    CASE
        WHEN b.price_at_5min > 0
        THEN ROUND(CAST(b.price_max_4h / b.price_at_5min AS DOUBLE), 2)
        ELSE NULL
    END                                                         AS max_multiple,

    -- Detección de rug
    CASE
        WHEN b.price_max_5min > 0
         AND b.price_min_5min / b.price_max_5min < 0.2         THEN TRUE
        ELSE FALSE
    END                                                         AS rug_detected,

    s.is_sustained                                              AS sustained_10min

FROM base_stats b
LEFT JOIN sustainability s ON b.coin_address = s.coin_address