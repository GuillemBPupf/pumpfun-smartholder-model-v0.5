WITH first_trades AS (
    SELECT
        token_bought_mint_address                    AS coin_address,
        MIN(block_time)                              AS created_at,
        MIN_BY(trader_id, block_time)                AS creator_wallet
    FROM dex_solana.trades
    WHERE project = 'pumpdotfun'
      AND token_sold_mint_address = 'So11111111111111111111111111111111111111112'
      AND block_month >= DATE '{{block_month}}'
      AND block_time  >= TIMESTAMP '{{start_date}}'
      AND block_time  <  TIMESTAMP '{{end_date}}'
    GROUP BY token_bought_mint_address
)
SELECT
    coin_address,
    created_at,
    creator_wallet
FROM first_trades
ORDER BY created_at ASC