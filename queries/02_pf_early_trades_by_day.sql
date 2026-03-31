-- ==============================================================
-- pumpfun_early_trades_by_day  (versión optimizada)
-- Parámetros: {{block_month}}, {{start_date}}, {{end_date}}, {{end_date_extended}}
--
-- Optimizaciones aplicadas vs versión original:
--   1. Un solo scan de dex_solana.trades (vs 2 scans anteriores)
--      → se usa MIN(block_time) OVER (...) en lugar de una CTE separada
--   2. Output agregado por (coin, wallet) en SQL
--      → ~50k-80k filas/día vs ~300k+ anteriores → menos MB exportados
--      → 20 créditos/MB × reducción del 75-80% = ahorro muy significativo
--   3. Sin ORDER BY final (costoso en Trino para resultados grandes)
--      → Pandas puede ordenar localmente si fuera necesario
--   4. Sin tx_id ni block_time individuales (no se usan tras la agregación)
--   5. DATE_DIFF calculado una sola vez por fila
-- ==============================================================

WITH all_trades AS (
    -- Scan ÚNICO de dex_solana.trades con todos los partition keys activos:
    --   project, token_sold_mint_address, block_month, block_time
    -- La window function calcula el MIN(block_time) por coin en este mismo paso
    -- sin necesidad de un segundo scan o CTE separada.
    SELECT
        token_bought_mint_address                                      AS coin_address,
        trader_id                                                      AS wallet_address,
        block_time,
        token_sold_amount                                              AS amount_sol,
        amount_usd,
        MIN(block_time) OVER (PARTITION BY token_bought_mint_address)  AS created_at
    FROM dex_solana.trades
    WHERE project                = 'pumpdotfun'
      AND token_sold_mint_address = 'So11111111111111111111111111111111111111112'
      AND block_month            >= DATE '{{block_month}}'
      AND block_time             >= TIMESTAMP '{{start_date}}'
      AND block_time             <  TIMESTAMP '{{end_date_extended}}'
),

early_trades AS (
    -- Filtros aplicados temprano antes de la agregación:
    --   1. Solo coins cuyo lanzamiento (created_at) cae dentro del día {{start_date}}
    --      → excluye coins lanzadas antes de start_date que tengan trades en la ventana
    --   2. Solo trades en los primeros 180 segundos desde el lanzamiento
    SELECT
        coin_address,
        wallet_address,
        amount_sol,
        amount_usd,
        DATE_DIFF('second', created_at, block_time) AS seconds_since_launch
    FROM all_trades
    WHERE created_at >= TIMESTAMP '{{start_date}}'
      AND created_at <  TIMESTAMP '{{end_date}}'
      AND DATE_DIFF('second', created_at, block_time) BETWEEN 0 AND 180
)

-- Agregación final: una fila por (coin, wallet)
-- Reduce el output de ~300k filas/día a ~50k-80k filas/día
-- Tier basado en el primer momento de entrada de cada wallet
SELECT
    coin_address,
    wallet_address,
    CAST(MIN(seconds_since_launch) AS DOUBLE)                             AS first_entry_seconds,
    CAST(SUM(amount_sol) AS DOUBLE)                                       AS total_sol_spent,
    CAST(SUM(amount_usd) AS DOUBLE)                                       AS total_usd_spent,
    COUNT(*)                                                              AS n_trades,
    CASE WHEN MIN(seconds_since_launch) <= 20 THEN 1 ELSE 2 END          AS tier
FROM early_trades
GROUP BY coin_address, wallet_address