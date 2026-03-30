"""
model.py
--------
Entrena dos modelos XGBoost sobre coin_features.
Usa splitter.py para el split temporal, igual que wallet_scoring
y features.py, garantizando consistencia sin hardcodear fechas.

Uso:
    python src/model.py

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
pip install xgboost scikit-learn
"""

import os
import sys
import json
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
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report, mean_squared_error, mean_absolute_error
    )
except ImportError:
    print("Instala: pip install xgboost scikit-learn")
    raise

load_dotenv()
os.makedirs("models", exist_ok=True)

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
            c.created_at
        FROM coin_features cf
        INNER JOIN coin_prices cp ON cf.coin_address = cp.coin_address
        INNER JOIN coins c        ON cf.coin_address = c.coin_address
        WHERE cp.label IS NOT NULL
        ORDER BY c.created_at ASC
    """, engine)
    print(f"  Dataset: {len(df):,} filas, {df['label'].sum():.0f} positivos "
          f"({df['label'].mean():.1%} tasa de éxito)")
    return df


# ── Preparación de datos ───────────────────────────────────────

def prepare_data(df: pd.DataFrame, train_coins: set, test_coins: set):
    train_df = df[df["coin_address"].isin(train_coins)].copy()
    test_df  = df[df["coin_address"].isin(test_coins)].copy()

    # Ordenar por fecha para que el log sea informativo
    train_df = train_df.sort_values("created_at")
    test_df  = test_df.sort_values("created_at")

    print(f"  Train: {len(train_df):,} coins "
          f"({train_df['created_at'].min()} → {train_df['created_at'].max()})")
    print(f"  Test:  {len(test_df):,} coins "
          f"({test_df['created_at'].min()} → {test_df['created_at'].max()})")
    print(f"  Positivos train: {train_df['label'].sum():.0f} "
          f"({train_df['label'].mean():.1%})")
    print(f"  Positivos test:  {test_df['label'].sum():.0f} "
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


# ── Clasificador ───────────────────────────────────────────────

def train_classifier(X_train, X_test, y_train, y_test):
    n_pos   = int(y_train.sum())
    n_neg   = int((y_train == 0).sum())
    scale_w = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_w:.1f} ({n_neg} neg / {n_pos} pos)")

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

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)

    print(f"\n  AUC-ROC: {roc_auc_score(y_test, proba_test):.4f}")
    print(f"  AUC-PR:  {average_precision_score(y_test, proba_test):.4f}"
          f"  ← métrica principal con desbalanceo")
    print(f"\n  Reporte (umbral 0.5):")
    print(classification_report(y_test, pred_test,
                                 target_names=["fracaso", "éxito"], digits=3))

    importance = pd.Series(
        clf.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("  Top 10 features:")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {bar} {imp:.4f}")

    return clf, proba_test


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

def save_signals(engine, test_df, proba_test, pred_reg, reg_mask_test):
    records = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        score = float(proba_test[i])
        if   score >= 0.70: tier = "high"
        elif score >= 0.50: tier = "medium"
        elif score >= 0.35: tier = "low"
        else:               tier = None

        records.append({
            "coin_address":      row["coin_address"],
            "generated_at":      datetime.now(timezone.utc),
            "model_score":       round(score, 6),
            "expected_multiple": None,
            "signal_tier":       tier,
            "outcome_label":     int(row["label"]),
            "outcome_verified_at": datetime.now(timezone.utc),
        })

    if pred_reg is not None:
        reg_indices = test_df[reg_mask_test.values].index
        reg_preds   = dict(zip(reg_indices, pred_reg))
        for rec, (idx, _) in zip(records, test_df.iterrows()):
            if idx in reg_preds:
                rec["expected_multiple"] = round(float(reg_preds[idx]), 2)

    with engine.connect() as conn:
        conn.execute(text(
            "DELETE FROM signals WHERE outcome_label IS NOT NULL"
        ))
        conn.commit()
        for start in range(0, len(records), 1000):
            conn.execute(text("""
                INSERT INTO signals (
                    coin_address, generated_at, model_score,
                    expected_multiple, signal_tier,
                    outcome_label, outcome_verified_at
                ) VALUES (
                    :coin_address, :generated_at, :model_score,
                    :expected_multiple, :signal_tier,
                    :outcome_label, :outcome_verified_at
                ) ON CONFLICT DO NOTHING
            """), records[start:start + 1000])
            conn.commit()

    tier_counts = {}
    for r in records:
        t = r["signal_tier"] or "sin_señal"
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  {len(records):,} señales guardadas.")
    print("  Distribución:")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier:<12} {count:>6,}")


# ── Guardar modelos ────────────────────────────────────────────

def save_models(clf, reg):
    clf.save_model("models/classifier.json")
    print("  models/classifier.json")
    if reg is not None:
        reg.save_model("models/regressor.json")
        print("  models/regressor.json")
    meta = {
        "features":     FEATURE_COLS,
        "train_ratio":  0.80,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  models/metadata.json")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MODEL TRAINING (sin data leakage)")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/4] Cargando dataset...")
    df = load_dataset(engine)
    if len(df) < 50:
        print("Dataset muy pequeño. Necesitas más datos.")
        return

    print("\n[2/4] Preparando datos (split temporal compartido)...")
    train_coins, test_coins = get_train_test_coins(engine)
    (X_train, X_test, y_train, y_test,
     X_train_reg, X_test_reg, y_train_reg, y_test_reg,
     train_df, test_df, reg_mask_test) = prepare_data(
        df, train_coins, test_coins
    )

    print("\n[3/4] Entrenando modelos...")
    print("\n  ── Clasificador ──")
    clf, proba_test = train_classifier(X_train, X_test, y_train, y_test)

    print("\n  ── Regresor ──")
    reg, pred_reg = train_regressor(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )

    print("\n[4/4] Guardando...")
    save_models(clf, reg)
    save_signals(engine, test_df, proba_test, pred_reg, reg_mask_test)

    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()