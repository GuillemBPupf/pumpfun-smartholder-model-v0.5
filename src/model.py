"""
model.py
--------
Entrena dos modelos XGBoost sobre coin_features:

  Modelo 1 — Clasificador (score 0-1):
    Target: label (1 = hizo 2.5x sostenido, 0 = no)
    Output: model_score en tabla signals

  Modelo 2 — Regresor (múltiplo esperado):
    Target: max_multiple (solo coins con max_multiple >= 1)
    Output: expected_multiple en tabla signals

Validación temporal:
    80% coins más antiguas → entrenamiento
    20% coins más recientes → test
    (nunca split aleatorio para evitar leakage temporal)

Modelos guardados en models/
    models/classifier.json
    models/regressor.json

Uso:
    python src/model.py

Requisitos en .env:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
pip install xgboost scikit-learn
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report, mean_squared_error, mean_absolute_error
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("Instala las dependencias: pip install xgboost scikit-learn")
    raise

load_dotenv()

os.makedirs("models", exist_ok=True)

# ── Features usadas por el modelo ─────────────────────────────

FEATURE_COLS = [
    "n_early_buyers",
    "n_reliable_wallets",
    "avg_wallet_score",
    "max_wallet_score",
    "pct_high_score_wallets",
    "pct_negative_wallets",
    "pct_new_wallets",
    "avg_cooccurrence_score",
    "total_volume_sol",
    "avg_buy_size_sol",
    "std_buy_size_sol",
    "concentration_top5",
    "n_tier1_buyers",
    "creator_is_buyer",
    "buys_in_first_20s",
    "buys_20s_to_60s",
    "buys_60s_to_180s",
    "acceleration_ratio",
    "time_to_5th_buy",
    "hour_utc",
    "day_of_week",
]

# Columnas categóricas (XGBoost las trata de forma especial)
CATEGORICAL_COLS = ["hour_utc", "day_of_week", "creator_is_buyer"]


# ── Conexión ───────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# ── Carga de datos ─────────────────────────────────────────────

def load_dataset(engine) -> pd.DataFrame:
    """
    Carga coin_features unido con coin_prices y coins (para el timestamp).
    Solo coins que tienen label Y features calculadas.
    """
    query = """
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
    """
    df = pd.read_sql(query, engine)
    print(f"  Dataset: {len(df):,} filas, {df['label'].sum():.0f} positivos "
          f"({df['label'].mean():.1%} tasa de éxito)")
    return df


# ── Preparación de datos ───────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Split temporal 80/20, prepara matrices X e y.
    """
    # Split temporal estricto
    split_idx   = int(len(df) * 0.80)
    train_df    = df.iloc[:split_idx].copy()
    test_df     = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df):,} coins "
          f"({train_df['created_at'].min()} → {train_df['created_at'].max()})")
    print(f"  Test:  {len(test_df):,} coins "
          f"({test_df['created_at'].min()} → {test_df['created_at'].max()})")
    print(f"  Positivos en train: {train_df['label'].sum():.0f} "
          f"({train_df['label'].mean():.1%})")
    print(f"  Positivos en test:  {test_df['label'].sum():.0f} "
          f"({test_df['label'].mean():.1%})")

    def to_X(d):
        X = d[FEATURE_COLS].copy()
        # Booleanos a int para XGBoost
        X["creator_is_buyer"] = X["creator_is_buyer"].astype(int)
        # Rellenar nulos con -1 (XGBoost los maneja bien)
        X = X.fillna(-1)
        return X

    X_train = to_X(train_df)
    X_test  = to_X(test_df)
    y_train = train_df["label"].astype(int)
    y_test  = test_df["label"].astype(int)

    # Para el regresor, solo coins con max_multiple disponible
    reg_mask_train = train_df["max_multiple"].notna()
    reg_mask_test  = test_df["max_multiple"].notna()
    y_train_reg    = train_df.loc[reg_mask_train, "max_multiple"].clip(upper=50)
    y_test_reg     = test_df.loc[reg_mask_test,   "max_multiple"].clip(upper=50)
    X_train_reg    = X_train[reg_mask_train]
    X_test_reg     = X_test[reg_mask_test]

    return (X_train, X_test, y_train, y_test,
            X_train_reg, X_test_reg, y_train_reg, y_test_reg,
            train_df, test_df)


# ── Entrenamiento Modelo 1: Clasificador ──────────────────────

def train_classifier(X_train, X_test, y_train, y_test):
    """
    XGBoost clasificador con class_weight automático para
    manejar el desbalanceo de clases.
    """
    print("\n  Entrenando clasificador XGBoost...")

    # Ratio para scale_pos_weight: negatives / positives
    n_pos   = int(y_train.sum())
    n_neg   = int((y_train == 0).sum())
    scale_w = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_w:.1f} ({n_neg} neg / {n_pos} pos)")

    clf = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_w,
        use_label_encoder = False,
        eval_metric       = "aucpr",   # AUC-PR mejor que AUC-ROC con desbalanceo
        early_stopping_rounds = 20,
        random_state      = 42,
        verbosity         = 0,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Métricas
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, proba_test)
    auc_pr  = average_precision_score(y_test, proba_test)

    print(f"\n  AUC-ROC:  {auc_roc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}  (métrica principal con desbalanceo)")
    print(f"\n  Reporte de clasificación (umbral 0.5):")
    print(classification_report(y_test, pred_test,
                                 target_names=["fracaso", "éxito"],
                                 digits=3))

    # Importancia de features (top 10)
    importance = pd.Series(
        clf.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("  Top 10 features más importantes:")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {bar} {imp:.4f}")

    return clf, proba_test


# ── Entrenamiento Modelo 2: Regresor ──────────────────────────

def train_regressor(X_train_reg, X_test_reg, y_train_reg, y_test_reg):
    """
    XGBoost regresor para predecir el múltiplo esperado.
    """
    if len(X_train_reg) < 10:
        print("\n  Insuficientes datos para el regresor. Saltando.")
        return None, None

    print(f"\n  Entrenando regresor XGBoost...")
    print(f"  Train reg: {len(X_train_reg):,} filas | "
          f"Test reg: {len(X_test_reg):,} filas")

    reg = xgb.XGBRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        early_stopping_rounds = 20,
        random_state     = 42,
        verbosity        = 0,
    )

    reg.fit(
        X_train_reg, y_train_reg,
        eval_set=[(X_test_reg, y_test_reg)],
        verbose=False,
    )

    pred_reg = reg.predict(X_test_reg).clip(min=1.0)
    rmse     = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
    mae      = mean_absolute_error(y_test_reg, pred_reg)

    print(f"\n  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  Target mean:  {y_test_reg.mean():.2f}x")
    print(f"  Pred mean:    {pred_reg.mean():.2f}x")

    return reg, pred_reg


# ── Guardar señales en la BD ───────────────────────────────────

def save_signals(
    engine,
    test_df: pd.DataFrame,
    proba_test: np.ndarray,
    pred_reg,
    reg_mask_test: pd.Series
):
    """
    Inserta las predicciones del conjunto de test en la tabla signals.
    """
    print("\n  Guardando señales en tabla signals...")

    # Score de clasificación
    records = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        score = float(proba_test[i])

        # Signal tier
        if score >= 0.70:
            tier = "high"
        elif score >= 0.50:
            tier = "medium"
        elif score >= 0.35:
            tier = "low"
        else:
            tier = None  # por debajo del umbral mínimo, no señal

        records.append({
            "coin_address":     row["coin_address"],
            "generated_at":     datetime.now(timezone.utc),
            "model_score":      round(score, 6),
            "expected_multiple": None,
            "signal_tier":      tier,
            "outcome_label":    int(row["label"]),
            "outcome_verified_at": datetime.now(timezone.utc),
        })

    # Añadir expected_multiple donde está disponible
    if pred_reg is not None:
        reg_indices = test_df[reg_mask_test.values].index
        reg_preds   = dict(zip(reg_indices, pred_reg))
        for rec, (idx, _) in zip(records, test_df.iterrows()):
            if idx in reg_preds:
                rec["expected_multiple"] = round(float(reg_preds[idx]), 2)

    with engine.connect() as conn:
        # Limpiar señales previas de test
        conn.execute(text("DELETE FROM signals WHERE outcome_label IS NOT NULL"))
        conn.commit()

        for batch_start in range(0, len(records), 1000):
            batch = records[batch_start:batch_start + 1000]
            conn.execute(text("""
                INSERT INTO signals (
                    coin_address, generated_at, model_score,
                    expected_multiple, signal_tier,
                    outcome_label, outcome_verified_at
                )
                VALUES (
                    :coin_address, :generated_at, :model_score,
                    :expected_multiple, :signal_tier,
                    :outcome_label, :outcome_verified_at
                )
                ON CONFLICT DO NOTHING
            """), batch)
            conn.commit()

    print(f"  {len(records):,} señales guardadas.")

    # Distribución de tiers
    tier_counts = {}
    for r in records:
        t = r["signal_tier"] or "sin_señal"
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print("\n  Distribución de señales en test:")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier:<12} {count:>6,}")


# ── Guardar modelos ────────────────────────────────────────────

def save_models(clf, reg):
    clf.save_model("models/classifier.json")
    print("\n  Modelo guardado: models/classifier.json")

    if reg is not None:
        reg.save_model("models/regressor.json")
        print("  Modelo guardado: models/regressor.json")

    # Guardar metadatos
    meta = {
        "features":          FEATURE_COLS,
        "categorical_cols":  CATEGORICAL_COLS,
        "trained_at":        datetime.now(timezone.utc).isoformat(),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Metadatos guardados: models/metadata.json")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/4] Cargando dataset...")
    df = load_dataset(engine)

    if len(df) < 50:
        print("Dataset muy pequeño para entrenar. Necesitas más datos.")
        return

    print("\n[2/4] Preparando datos (split temporal 80/20)...")
    (X_train, X_test, y_train, y_test,
     X_train_reg, X_test_reg, y_train_reg, y_test_reg,
     train_df, test_df) = prepare_data(df)

    print("\n[3/4] Entrenando modelos...")

    print("\n  ── Modelo 1: Clasificador ──")
    clf, proba_test = train_classifier(X_train, X_test, y_train, y_test)

    print("\n  ── Modelo 2: Regresor ──")
    reg_mask_test = test_df["max_multiple"].notna()
    reg, pred_reg = train_regressor(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )

    print("\n[4/4] Guardando modelos y señales...")
    save_models(clf, reg)
    save_signals(engine, test_df, proba_test, pred_reg, reg_mask_test)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("  Modelos en: models/")
    print("  Señales en: tabla signals")
    print("=" * 60)


if __name__ == "__main__":
    main()