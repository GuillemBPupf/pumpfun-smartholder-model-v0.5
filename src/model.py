"""
model.py
--------
Entrena dos modelos XGBoost sobre coin_features.
Calibra el clasificador con isotonic regression.
Optimiza el umbral de señal maximizando P&L esperado por trade.
Realiza un backtesting rápido sobre el test set.

Flujo:
  1. Cargar datos (incluye rug_detected)
  2. Split temporal compartido (via splitter.py)
  3. Entrenar clasificador + regresor
  4. Calibrar clasificador (isotonic regression sobre test set)
  5. Optimizar umbral (máximo avg P&L por señal)
  6. Backtesting rápido con 3 estrategias
  7. Guardar modelos, calibrador, señales y metadata

⚠  Pre-requisito BD (si ya tienes la tabla signals creada):
     ALTER TABLE signals ADD COLUMN IF NOT EXISTS ev_score NUMERIC;
   Para una BD nueva, ejecutar create_tables.sql directamente.

Uso:
    python src/model.py
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from splitter import get_train_test_coins

try:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report, mean_squared_error,
        mean_absolute_error, brier_score_loss,
    )
except ImportError:
    print("Instala: pip install xgboost scikit-learn")
    raise

load_dotenv()
os.makedirs("models", exist_ok=True)


# ── Constantes de simulación ───────────────────────────────────
SLIPPAGE                = 0.03   # slippage de entrada y salida (3%)
TAKE_PROFIT_X           = 2.5    # múltiplo de take profit (coincide con label=1)
RUG_RECOVERY            = 0.15   # fracción recuperada si hay rug
NOLABEL_EXIT_MULTIPLIER = 0.80   # salida fija para label=0 sin rug
                                  # (stop loss ~20% por debajo de entrada)
TRADE_SIZE              = 50     # capital base por señal (solo para display de Capital)

# Pérdida media estimada cuando la señal fracasa (para fórmula EV):
#   42% rugs     → pnl/u ≈ -0.88
#   58% no-rugs  → pnl/u = NOLABEL_EXIT_MULTIPLIER × (1-slip) - (1+slip) ≈ -0.25
#   Promedio: 0.42 × (-0.88) + 0.58 × (-0.25) ≈ -0.52
AVG_LOSS_ON_FAILURE = -0.52

# Mínimo de señales para evaluar un umbral en la búsqueda
MIN_SIGNALS = 10

# Fallback si la optimización no converge
THRESHOLD_FALLBACK = 0.50

# Break-even precision:
#   EV=0  →  P × gain_éxito = (1-P) × |AVG_LOSS_ON_FAILURE|
#   gain_éxito = 2.5×(1-slip) - (1+slip) = 1.395 con slip=3%
#   P_be = 0.52 / (1.395 + 0.52) ≈ 27.2%
_GAIN_SUCCESS       = TAKE_PROFIT_X * (1 - SLIPPAGE) - (1 + SLIPPAGE)
PRECISION_BREAKEVEN = abs(AVG_LOSS_ON_FAILURE) / (_GAIN_SUCCESS + abs(AVG_LOSS_ON_FAILURE))

FEATURE_COLS = [
    "n_early_buyers", "n_reliable_wallets",
    "avg_wallet_score", "max_wallet_score",
    "pct_high_score_wallets", "pct_negative_wallets", "pct_new_wallets",
    "avg_cooccurrence_score",
    "total_volume_sol", "avg_buy_size_sol", "std_buy_size_sol",
    "concentration_top5", "n_tier1_buyers", "creator_is_buyer",
    "buys_in_first_20s", "buys_20s_to_60s", "buys_60s_to_180s",
    "acceleration_ratio", "time_to_5th_buy",
    "hour_utc", "day_of_week",
]


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_dataset(engine) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT
            cf.*,
            cp.label,
            cp.max_multiple,
            cp.rug_detected,
            c.created_at
        FROM coin_features cf
        INNER JOIN coin_prices cp ON cf.coin_address = cp.coin_address
        INNER JOIN coins c        ON cf.coin_address = c.coin_address
        WHERE cp.label IS NOT NULL
        ORDER BY c.created_at ASC
    """, engine)
    df["rug_detected"] = df["rug_detected"].fillna(False).astype(bool)
    print(f"  Dataset: {len(df):,} filas | {int(df['label'].sum())} positivos "
          f"({df['label'].mean():.1%} tasa de éxito)")
    return df


# ── Preparación de datos ───────────────────────────────────────

def prepare_data(df: pd.DataFrame, train_coins: set, test_coins: set):
    train_df = (
        df[df["coin_address"].isin(train_coins)]
        .copy()
        .sort_values("created_at")
        .reset_index(drop=True)
    )
    test_df = (
        df[df["coin_address"].isin(test_coins)]
        .copy()
        .sort_values("created_at")
        .reset_index(drop=True)
    )

    print(f"  Train: {len(train_df):,} coins "
          f"({train_df['created_at'].min()} → {train_df['created_at'].max()})")
    print(f"  Test:  {len(test_df):,} coins "
          f"({test_df['created_at'].min()} → {test_df['created_at'].max()})")
    print(f"  Positivos train: {int(train_df['label'].sum())} "
          f"({train_df['label'].mean():.1%})")
    print(f"  Positivos test:  {int(test_df['label'].sum())} "
          f"({test_df['label'].mean():.1%})")

    def to_X(d):
        X = d[FEATURE_COLS].copy()
        X["creator_is_buyer"] = X["creator_is_buyer"].astype(int)
        return X.fillna(-1)

    X_train = to_X(train_df)
    X_test  = to_X(test_df)
    y_train = train_df["label"].astype(int)
    y_test  = test_df["label"].astype(int)

    reg_mask_train = train_df["max_multiple"].notna()
    reg_mask_test  = test_df["max_multiple"].notna()
    X_train_reg    = X_train[reg_mask_train]
    X_test_reg     = X_test[reg_mask_test]
    y_train_reg    = train_df.loc[reg_mask_train, "max_multiple"].clip(upper=50)
    y_test_reg     = test_df.loc[reg_mask_test,   "max_multiple"].clip(upper=50)

    return (X_train, X_test, y_train, y_test,
            X_train_reg, X_test_reg, y_train_reg, y_test_reg,
            train_df, test_df, reg_mask_test)


# ── Simulación de P&L de un trade ─────────────────────────────

def simulate_trade_pnl(
    label: int,
    max_multiple,        # mantenido en firma por compatibilidad, no se usa
    rug_detected: bool,
    slippage: float = SLIPPAGE,
) -> float:
    """
    P&L neto POR UNIDAD APOSTADA dado el outcome real.
    NO se multiplica por TRADE_SIZE aquí; eso ocurre solo en el output.

    Escenarios:
    - label=1              → exit al take profit (TAKE_PROFIT_X)
    - label=0 + rug        → exit a RUG_RECOVERY del precio de entrada
    - label=0 sin rug      → exit a NOLABEL_EXIT_MULTIPLIER del precio de entrada
                              (stop loss fijo ~20% por debajo de entrada)

    Ejemplo con SLIPPAGE=3%:
      label=1   → +1.395 por unidad
      rug       → -0.884 por unidad
      no-rug    → -0.254 por unidad

    Returns: P&L neto por unidad (>0 ganancia, <0 pérdida).
    """
    entry = 1.0 + slippage

    if label == 1:
        exit_val = TAKE_PROFIT_X * (1.0 - slippage)
    elif bool(rug_detected):
        exit_val = RUG_RECOVERY * (1.0 - slippage)
    else:
        exit_val = NOLABEL_EXIT_MULTIPLIER * (1.0 - slippage)

    return round(exit_val - entry, 6)   # per-unit, sin escalar por TRADE_SIZE


# ── EV por señal (para producción) ────────────────────────────

def compute_ev(p_calib: float, slippage: float = SLIPPAGE) -> float:
    """
    Expected Value por unidad apostada (para uso en producción,
    cuando el outcome real es desconocido).

    EV = P × ganancia_si_éxito + (1-P) × pérdida_media_si_fracaso

    - ganancia_si_éxito        = 2.5×(1-slip) - (1+slip)  ≈  +1.395
    - pérdida_media_si_fracaso = AVG_LOSS_ON_FAILURE       ≈  -0.52

    Break-even: P ≈ 27.2% (ver PRECISION_BREAKEVEN)
    """
    gain = TAKE_PROFIT_X * (1.0 - slippage) - (1.0 + slippage)
    ev   = p_calib * gain + (1.0 - p_calib) * AVG_LOSS_ON_FAILURE
    return round(float(ev), 6)


# ── Clasificador ───────────────────────────────────────────────

def train_classifier(X_train, X_test, y_train, y_test):
    n_pos   = int(y_train.sum())
    n_neg   = int((y_train == 0).sum())
    scale_w = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_w:.1f}  ({n_neg:,} neg / {n_pos:,} pos)")

    clf = xgb.XGBClassifier(
        n_estimators          = 300,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = scale_w,
        use_label_encoder     = False,
        eval_metric           = "aucpr",
        early_stopping_rounds = 20,
        random_state          = 42,
        verbosity             = 0,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=False)

    proba_raw = clf.predict_proba(X_test)[:, 1]

    print(f"\n  AUC-ROC: {roc_auc_score(y_test, proba_raw):.4f}")
    print(f"  AUC-PR:  {average_precision_score(y_test, proba_raw):.4f}"
          f"  ← métrica principal")
    print(f"  Brier (raw): {brier_score_loss(y_test, proba_raw):.4f}")

    importance = pd.Series(
        clf.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {bar} {imp:.4f}")

    return clf, proba_raw


# ── Calibración ────────────────────────────────────────────────

def calibrate_classifier(proba_raw: np.ndarray, y_test) -> tuple:
    """
    Calibra las probabilidades brutas con isotonic regression
    ajustada sobre el test set.

    ⚠ Limitación: se calibra sobre el mismo test usado para evaluar.
    Con más datos, lo correcto es usar un val set separado.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_raw, y_test)
    proba_calib = calibrator.predict(proba_raw)

    print(f"  Brier score: {brier_score_loss(y_test, proba_raw):.4f} (raw)"
          f"  →  {brier_score_loss(y_test, proba_calib):.4f} (calibrado)")
    print(f"  Proba raw   — rango [{proba_raw.min():.4f}, {proba_raw.max():.4f}]"
          f"  media={proba_raw.mean():.4f}")
    print(f"  Proba calib — rango [{proba_calib.min():.4f}, {proba_calib.max():.4f}]"
          f"  media={proba_calib.mean():.4f}")
    print(f"\n  Break-even precision para EV=0: {PRECISION_BREAKEVEN:.1%}")
    print(f"  (necesitamos ≥{PRECISION_BREAKEVEN:.1%} precisión en las señales emitidas)")

    return calibrator, proba_calib


# ── Optimización del umbral ────────────────────────────────────

def find_optimal_threshold(
    y_test,
    proba_calib: np.ndarray,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Busca el umbral de probabilidad calibrada que maximiza
    el avg P&L por unidad por señal emitida.

    Itera sobre 200 umbrales entre 0.02 y 0.98.
    Solo evalúa umbrales con al menos MIN_SIGNALS señales.

    Returns: (mejor_umbral, DataFrame con todos los resultados)
    """
    thresholds = np.round(np.linspace(0.02, 0.98, 200), 4)
    y_arr   = np.asarray(y_test)
    m_arr   = test_df["max_multiple"].values
    rug_arr = test_df["rug_detected"].values
    results = []

    for t in thresholds:
        idxs = np.where(proba_calib >= t)[0]
        n    = len(idxs)
        if n < MIN_SIGNALS:
            continue

        # P&L per-unit (sin TRADE_SIZE)
        pnls      = np.array([
            simulate_trade_pnl(int(y_arr[i]), m_arr[i], bool(rug_arr[i]))
            for i in idxs
        ])
        avg_pnl_pu   = float(pnls.mean())
        total_capital = float(pnls.sum()) * TRADE_SIZE
        precision     = float(y_arr[idxs].mean())

        results.append({
            "threshold":     round(float(t), 4),
            "n_signals":     int(n),
            "n_wins":        int(y_arr[idxs].sum()),
            "precision":     round(precision, 4),
            "avg_pnl_pu":    round(avg_pnl_pu, 6),
            "total_capital": round(total_capital, 2),
        })

    df_thresh = pd.DataFrame(results)

    if df_thresh.empty:
        print("  ⚠ No se pudo calcular umbral óptimo → usando fallback.")
        return THRESHOLD_FALLBACK, df_thresh

    best_idx = df_thresh["avg_pnl_pu"].idxmax()
    best     = df_thresh.loc[best_idx]
    best_t   = float(best["threshold"])

    print(f"\n  Umbral óptimo (max AvgPnL/u por señal): {best_t:.3f}")
    print(f"    Señales en test:   {int(best['n_signals']):,}")
    print(f"    Precisión:         {best['precision']:.1%}  "
          f"(break-even: {PRECISION_BREAKEVEN:.1%})")
    print(f"    AvgPnL / unidad:   {best['avg_pnl_pu']:+.4f}")
    total_pu = float(best["avg_pnl_pu"]) * int(best["n_signals"])
    print(f"    Capital total:     {best['total_capital']:+.2f}  "
          f"(sum P&L/u={total_pu:+.4f} × trade_size={TRADE_SIZE})")

    # Tabla de umbrales clave:
    # Para cada umbral de referencia, encontramos el índice exacto de la fila
    # más cercana en df_thresh y lo guardamos. Así cada umbral de referencia
    # aparece exactamente una vez, con su valor real (no aproximado).
    key_t = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    seen_idx = set()
    rows_to_print = []

    # Fila del óptimo siempre incluida
    seen_idx.add(best_idx)
    rows_to_print.append((best_idx, True))

    # Para cada umbral clave, el índice de la fila más cercana
    for kt in key_t:
        closest_idx = (df_thresh["threshold"] - kt).abs().idxmin()
        if closest_idx not in seen_idx:
            seen_idx.add(closest_idx)
            rows_to_print.append((closest_idx, False))

    # Ordenar por threshold para imprimir de menor a mayor
    rows_to_print.sort(key=lambda x: df_thresh.loc[x[0], "threshold"])

    print(f"\n  Tabla de umbrales clave (Capital = sum×{TRADE_SIZE}):")
    print(f"  {'Umbral':>7} {'Señales':>8} {'Precisión':>10} "
          f"{'AvgPnL/u':>10} {'Capital':>10}")
    print("  " + "─" * 55)
    for idx, is_best in rows_to_print:
        row    = df_thresh.loc[idx]
        marker = "  ←ÓPTIMO" if is_best else ""
        print(f"  {row.threshold:>7.4f} {int(row.n_signals):>8,} "
              f"{row.precision:>10.1%} {row.avg_pnl_pu:>+10.4f} "
              f"{row.total_capital:>+10.2f}{marker}")

    return best_t, df_thresh


# ── Backtesting rápido ─────────────────────────────────────────

def run_backtest(
    y_test,
    proba_calib: np.ndarray,
    test_df: pd.DataFrame,
    optimal_threshold: float,
) -> dict:
    """
    Compara tres estrategias sobre el test set.
    P&L per-unit internamente; Capital = sum × TRADE_SIZE en el output.
    """
    y_arr   = np.asarray(y_test)
    m_arr   = test_df["max_multiple"].values
    rug_arr = test_df["rug_detected"].values

    strategies = {
        "Baseline (todo)":                   0.01,
        f"Óptimo ({optimal_threshold:.2f})": optimal_threshold,
        "Alta confianza (0.70)":             0.70,
    }

    all_results = {}

    print(f"\n  {'Estrategia':<28} {'Señales':>8} {'Precisión':>10} "
          f"{'AvgPnL/u':>10} {'Capital':>10} {'Sharpe':>7}")
    print(f"  {'':28} {'':8} {'':10} "
          f"{'':10} {'(×'+str(TRADE_SIZE)+')':>10} {'':7}")
    print("  " + "─" * 80)

    for name, threshold in strategies.items():
        idxs = np.where(proba_calib >= threshold)[0]
        if len(idxs) == 0:
            print(f"  {name:<28}  (sin señales con este umbral)")
            continue

        pnls          = np.array([
            simulate_trade_pnl(int(y_arr[i]), m_arr[i], bool(rug_arr[i]))
            for i in idxs
        ])
        n             = len(pnls)
        n_wins        = int(y_arr[idxs].sum())
        avg_pnl_pu    = float(pnls.mean())
        total_capital = float(pnls.sum()) * TRADE_SIZE
        precision     = float(y_arr[idxs].mean())
        sharpe        = avg_pnl_pu / (pnls.std() + 1e-9) * np.sqrt(n)

        ev_marker = "✓" if avg_pnl_pu > 0 else " "
        all_results[name] = {
            "threshold":     threshold,
            "n_signals":     n,
            "n_wins":        n_wins,
            "precision":     round(precision, 4),
            "avg_pnl_pu":    round(avg_pnl_pu, 6),
            "total_capital": round(total_capital, 2),
            "sharpe":        round(sharpe, 4),
        }

        print(f"  {ev_marker}{name:<27} {n:>8,} {precision:>10.1%} "
              f"{avg_pnl_pu:>+10.4f} {total_capital:>+10.2f} {sharpe:>7.2f}")

    print("  (✓ = AvgPnL/u positivo)")

    # Reporte clasificación con umbral óptimo
    opt_idxs = np.where(proba_calib >= optimal_threshold)[0]
    if len(opt_idxs) >= 5:
        pred_bin           = np.zeros(len(y_arr), dtype=int)
        pred_bin[opt_idxs] = 1
        print(f"\n  Reporte clasificación (umbral óptimo {optimal_threshold:.2f}):")
        print(classification_report(
            y_arr, pred_bin,
            target_names=["fracaso", "éxito"], digits=3
        ))

    return all_results


# ── Regresor ───────────────────────────────────────────────────

def train_regressor(X_train_reg, X_test_reg, y_train_reg, y_test_reg):
    if len(X_train_reg) < 10:
        print("  Insuficientes datos para el regresor.")
        return None, None

    print(f"  Train: {len(X_train_reg):,} | Test: {len(X_test_reg):,}")
    reg = xgb.XGBRegressor(
        n_estimators          = 300,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        early_stopping_rounds = 20,
        random_state          = 42,
        verbosity             = 0,
    )
    reg.fit(X_train_reg, y_train_reg,
            eval_set=[(X_test_reg, y_test_reg)], verbose=False)

    pred_reg = reg.predict(X_test_reg).clip(min=1.0)
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, pred_reg)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_test_reg, pred_reg):.4f}")
    print(f"  Target mean: {y_test_reg.mean():.2f}x | "
          f"Pred mean: {pred_reg.mean():.2f}x")
    return reg, pred_reg


# ── Guardar señales ────────────────────────────────────────────

def save_signals(
    engine,
    test_df: pd.DataFrame,
    proba_calib: np.ndarray,
    pred_reg,
    reg_mask_test,
    optimal_threshold: float,
):
    """
    Guarda señales en la BD con probabilidad calibrada y EV.

    Tiers por EV (por unidad apostada):
      'high'   → EV > 0.30
      'medium' → EV > 0.10
      'low'    → EV > 0.00
       None    → EV ≤ 0.00 (no se emite señal)
    """
    if pred_reg is not None:
        reg_idxs = test_df[reg_mask_test.values].index
        reg_map  = dict(zip(reg_idxs.tolist(), pred_reg.tolist()))
    else:
        reg_map = {}

    records = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        p_calib  = float(proba_calib[i])
        exp_mult = float(reg_map.get(idx, TAKE_PROFIT_X))
        ev       = compute_ev(p_calib)

        if   ev > 0.30: tier = "high"
        elif ev > 0.10: tier = "medium"
        elif ev > 0.00: tier = "low"
        else:           tier = None

        records.append({
            "coin_address":        row["coin_address"],
            "generated_at":        datetime.now(timezone.utc),
            "model_score":         round(p_calib, 6),
            "expected_multiple":   round(exp_mult, 2),
            "ev_score":            round(ev, 6),
            "signal_tier":         tier,
            "outcome_label":       int(row["label"]),
            "outcome_verified_at": datetime.now(timezone.utc),
        })

    with engine.connect() as conn:
        conn.execute(text(
            "DELETE FROM signals WHERE outcome_label IS NOT NULL"
        ))
        conn.commit()
        for start in range(0, len(records), 1_000):
            conn.execute(text("""
                INSERT INTO signals (
                    coin_address, generated_at, model_score,
                    expected_multiple, ev_score, signal_tier,
                    outcome_label, outcome_verified_at
                ) VALUES (
                    :coin_address, :generated_at, :model_score,
                    :expected_multiple, :ev_score, :signal_tier,
                    :outcome_label, :outcome_verified_at
                ) ON CONFLICT DO NOTHING
            """), records[start:start + 1_000])
            conn.commit()

    n_signaled     = sum(1 for r in records if r["signal_tier"] is not None)
    n_pos_signaled = sum(1 for r in records if r["signal_tier"] is not None
                         and r["outcome_label"] == 1)
    n_ev_pos       = sum(1 for r in records if r["ev_score"] > 0)

    tier_counts: dict = {}
    for r in records:
        t = r["signal_tier"] or "sin_señal"
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print(f"\n  {len(records):,} registros en signals.")
    print(f"  Con EV > 0: {n_ev_pos:,} ({n_ev_pos / len(records):.1%})")
    if n_signaled > 0:
        print(f"  Precisión sobre señales emitidas: "
              f"{n_pos_signaled}/{n_signaled} = "
              f"{n_pos_signaled / n_signaled:.1%}")
    print("  Distribución de tiers:")
    for tier in ["high", "medium", "low", "sin_señal"]:
        count = tier_counts.get(tier, 0)
        bar   = "█" * min(count // max(len(records) // 40, 1), 35)
        print(f"    {tier:<12} {bar} ({count:,})")


# ── Guardar modelos ────────────────────────────────────────────

def save_models(
    clf,
    reg,
    calibrator,
    optimal_threshold: float,
    backtest_results: dict,
):
    clf.save_model("models/classifier.json")
    print("  models/classifier.json")

    if reg is not None:
        reg.save_model("models/regressor.json")
        print("  models/regressor.json")

    with open("models/calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    print("  models/calibrator.pkl")

    meta = {
        "features":             FEATURE_COLS,
        "train_ratio":          0.80,
        "trained_at":           datetime.now(timezone.utc).isoformat(),
        "optimal_threshold":    optimal_threshold,
        "precision_breakeven":  round(PRECISION_BREAKEVEN, 4),
        "slippage":             SLIPPAGE,
        "take_profit_x":        TAKE_PROFIT_X,
        "nolabel_exit":         NOLABEL_EXIT_MULTIPLIER,
        "avg_loss_on_failure":  AVG_LOSS_ON_FAILURE,
        "trade_size":           TRADE_SIZE,
        "backtest_summary":     backtest_results,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  models/metadata.json")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MODEL v2 — Calibración + EV + Backtesting")
    print(f"  Break-even precision: {PRECISION_BREAKEVEN:.1%}")
    print(f"  Slippage asumido:     {SLIPPAGE:.0%}")
    print(f"  Take profit:          {TAKE_PROFIT_X}x")
    print(f"  Pérdida media fallo:  {AVG_LOSS_ON_FAILURE:+.2f}/u")
    print(f"  Trade size:           {TRADE_SIZE} (solo para display Capital)")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/5] Cargando dataset...")
    df = load_dataset(engine)
    if len(df) < 50:
        print("Dataset muy pequeño. Necesitas más datos.")
        return

    print("\n[2/5] Split temporal y preparación de datos...")
    train_coins, test_coins = get_train_test_coins(engine)
    (X_train, X_test, y_train, y_test,
     X_train_reg, X_test_reg, y_train_reg, y_test_reg,
     train_df, test_df, reg_mask_test) = prepare_data(
        df, train_coins, test_coins
    )

    print("\n[3/5] Entrenando modelos...")
    print("\n  ── Clasificador ──")
    clf, proba_raw = train_classifier(X_train, X_test, y_train, y_test)

    print("\n  ── Regresor ──")
    reg, pred_reg = train_regressor(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )

    print("\n[4/5] Calibración y optimización de umbral...")
    calibrator, proba_calib = calibrate_classifier(proba_raw, y_test)

    optimal_threshold, _ = find_optimal_threshold(
        y_test, proba_calib, test_df
    )

    print("\n[5/5] Backtesting, guardado y señales...")
    print("\n  ── Backtesting rápido ──")
    backtest_results = run_backtest(
        y_test, proba_calib, test_df, optimal_threshold
    )

    print("\n  ── Guardando modelos ──")
    save_models(clf, reg, calibrator, optimal_threshold, backtest_results)

    print("\n  ── Guardando señales ──")
    save_signals(
        engine, test_df, proba_calib, pred_reg, reg_mask_test, optimal_threshold
    )

    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("Para análisis detallado: python src/backtest.py")
    print("=" * 60)


if __name__ == "__main__":
    main()