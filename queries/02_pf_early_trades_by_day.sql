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

early_trades AS (
    SELECT
        t.token_bought_mint_address                              AS coin_address,
        t.trader_id                                              AS wallet_address,
        t.block_time,
        CAST(
            DATE_DIFF('second', l.created_at, t.block_time)
        AS DOUBLE)                                               AS seconds_since_launch,
        t.token_sold_amount                                      AS amount_sol,
        t.amount_usd,
        t.tx_id,
        CASE
            WHEN DATE_DIFF('second', l.created_at, t.block_time) <= 20
            THEN 1 ELSE 2
        END                                                      AS tier
    FROM dex_solana.trades t
    INNER JOIN coin_launches l
        ON t.token_bought_mint_address = l.coin_address
    WHERE t.project = 'pumpdotfun'
      AND t.token_sold_mint_address = 'So11111111111111111111111111111111111111112'
      AND t.block_month >= DATE '{{block_month}}'
      AND t.block_time  >= TIMESTAMP '{{start_date}}'
      AND t.block_time  <  TIMESTAMP '{{end_date_extended}}'
)

SELECT *
FROM early_trades
WHERE seconds_since_launch BETWEEN 0 AND 180
ORDER BY coin_address, block_time ASC