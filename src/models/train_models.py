"""
src/models/train_models.py

Trains and compares Random Forest, XGBoost, and CatBoost on three
targets: Duration_Hours, Total_Cost_USD, NPT_Hours.

Why these three models (and not "everything"):
  - Linear Regression: tested as a sanity baseline only (drilling
    relationships are non-linear, so it's expected to lose).
  - Random Forest: stable, hard-to-overfit benchmark.
  - XGBoost: industry-standard gradient boosting, usually wins on
    structured tabular data.
  - CatBoost: handles raw categorical columns (Basin, Phase, Formation,
    Well_Type) natively without manual encoding, and often performs
    very well when categoricals are important -- as they are here.

The script picks the BEST model PER TARGET based on lowest MAE on a
held-out test set, and saves it + its metrics for the dashboard.

IMPORTANT - train/test split strategy:
  We split by Well_ID, not randomly by row. All 25 phase-rows of a
  well stay together in either train or test. Random row-splitting
  would leak information (other phases of the same well) between
  train and test and give falsely optimistic metrics.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

DATA_PATH = os.path.join("data", "processed", "features_data.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Targets we predict
TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]

# Raw categorical columns (used directly by CatBoost)
CAT_COLS = ["Basin", "Well_Type", "Phase", "Formation"]

# Columns that must NEVER be used as features (target leakage / IDs)
LEAKAGE_COLS = {
    "Duration_Hours": [
        "Total_Cost_USD", "Cost_Per_Hour", "Cost_Per_Meter", "NPT_Hours",
        "NPT_Ratio", "Productive_Hours", "Cumulative_Duration",
        "Cumulative_NPT", "Sum_Cost_Components", "Failure_Density",
        "Weather_Impact", "Daily_Rig_Rate_USD", "Materials_Cost_USD",
        "Service_Cost_USD"
    ],
    "Total_Cost_USD": [
        "Cost_Per_Hour", "Cost_Per_Meter", "Sum_Cost_Components",
        "Daily_Rig_Rate_USD", "Materials_Cost_USD", "Service_Cost_USD"
    ],
    "NPT_Hours": [
        "NPT_Ratio", "Productive_Hours", "Cumulative_NPT",
        "NPT_Was_Capped", "Previous_Phase_NPT", "Failure_Density",
        "Total_Cost_USD", "Cost_Per_Hour", "Cost_Per_Meter",
        "Sum_Cost_Components", "Daily_Rig_Rate_USD", "Materials_Cost_USD",
        "Service_Cost_USD"
    ],
}

ID_COLS = ["Well_ID", "Basin_Formation", "Campaign_ID"]
ENCODED_COLS = [c for c in [
    "Basin_Encoded", "Well_Type_Encoded", "Phase_Encoded",
    "Formation_Encoded", "Basin_Formation_Encoded"
]]


def get_feature_columns(df: pd.DataFrame, target: str):
    """Return (numeric_features_for_RF_XGB, all_features_for_catboost, cat_cols_present)."""
    drop_cols = set(TARGETS) | set(ID_COLS) | set(LEAKAGE_COLS.get(target, []))
    drop_cols.add(target)

    all_cols = [c for c in df.columns if c not in drop_cols]

    # For RF / XGBoost: drop raw string categoricals, keep the *_Encoded versions
    raw_cat_cols = [c for c in CAT_COLS + ["Basin_Formation"] if c in all_cols]
    numeric_features = [c for c in all_cols if c not in raw_cat_cols]

    # For CatBoost: keep raw categoricals (drop the encoded duplicates to avoid redundancy)
    encoded_dupes = [c for c in all_cols if c.endswith("_Encoded")]
    catboost_features = [c for c in all_cols if c not in encoded_dupes]

    cat_cols_present = [c for c in raw_cat_cols if c in catboost_features]

    return numeric_features, catboost_features, cat_cols_present


def split_by_well(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["Well_ID"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE -- guard against division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


def train_for_target(df: pd.DataFrame, target: str):
    print(f"\n{'='*60}\nTARGET: {target}\n{'='*60}")

    numeric_features, catboost_features, cat_cols_present = get_feature_columns(df, target)

    train_df, test_df = split_by_well(df, target)
    print(f"Train wells: {train_df['Well_ID'].nunique()} ({len(train_df)} rows)")
    print(f"Test wells:  {test_df['Well_ID'].nunique()} ({len(test_df)} rows)")

    y_train = train_df[target].values
    y_test = test_df[target].values

    results = {}
    models = {}

    # ---- 1. Linear Regression (baseline) ----
    X_train = train_df[numeric_features].values
    X_test = test_df[numeric_features].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results["LinearRegression"] = evaluate(y_test, pred)
    models["LinearRegression"] = (lr, numeric_features, "numeric")

    # ---- 2. Random Forest ----
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["RandomForest"] = evaluate(y_test, pred)
    models["RandomForest"] = (rf, numeric_features, "numeric")

    # ---- 3. XGBoost ----
    xgb = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror", n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    results["XGBoost"] = evaluate(y_test, pred)
    models["XGBoost"] = (xgb, numeric_features, "numeric")

    # ---- 4. CatBoost (handles raw categoricals natively) ----
    X_train_cb = train_df[catboost_features].copy()
    X_test_cb = test_df[catboost_features].copy()
    for c in cat_cols_present:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    cb = CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.05,
        random_seed=42, verbose=False, cat_features=cat_cols_present
    )
    cb.fit(X_train_cb, y_train)
    pred = cb.predict(X_test_cb)
    results["CatBoost"] = evaluate(y_test, pred)
    models["CatBoost"] = (cb, catboost_features, "catboost")

    # Print comparison table
    print(f"\n{'Model':<18}{'MAE':>12}{'RMSE':>12}{'MAPE%':>10}{'R2':>8}")
    for name, m in results.items():
        print(f"{name:<18}{m['MAE']:>12.3f}{m['RMSE']:>12.3f}{m['MAPE']:>10.2f}{m['R2']:>8.3f}")

    # Pick best model = lowest MAE
    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n>>> BEST MODEL for {target}: {best_name} (lowest MAE)")

    best_model, feat_cols, mode = models[best_name]

    # Compute residuals on test set for Monte Carlo layer (next stage)
    if mode == "catboost":
        best_pred = best_model.predict(X_test_cb)
    else:
        best_pred = best_model.predict(X_test)
    residuals = (y_test - best_pred).tolist()

    return {
        "target": target,
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_columns": feat_cols,
        "cat_features": cat_cols_present,
        "mode": mode,
        "residuals": residuals,
        "test_well_ids": test_df["Well_ID"].unique().tolist(),
    }


def train_all(df: pd.DataFrame, save: bool = True):
    """
    Train + compare all models for all targets on the given dataframe.
    Returns the summary dict. If save=True, writes model .pkl files,
    per-target metadata.json, and reports/model_comparison.json --
    exactly like running this script from the command line.

    This function is what the Streamlit "Retrain" page calls, so the
    dashboard and the CLI script share one code path.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    summary = {}

    for target in TARGETS:
        out = train_for_target(df, target)

        if save:
            model_path = os.path.join(MODELS_DIR, f"{target}_best_model.pkl")
            joblib.dump(out["best_model"], model_path)

            meta = {
                "target": target,
                "best_model_name": out["best_model_name"],
                "feature_columns": out["feature_columns"],
                "cat_features": out["cat_features"],
                "mode": out["mode"],
                "metrics": out["results"],
                "residuals": out["residuals"],
            }
            meta_path = os.path.join(MODELS_DIR, f"{target}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        summary[target] = {
            "best_model": out["best_model_name"],
            "metrics": out["results"],
        }

    if save:
        report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def main():
    df = pd.read_csv(DATA_PATH)
    summary = train_all(df, save=True)

    print("\n" + "=" * 60)
    print("FINAL MODEL SELECTION")
    print("=" * 60)
    for target, info in summary.items():
        print(f"{target:<18} -> {info['best_model']}  (MAE={info['metrics'][info['best_model']]['MAE']:.3f})")


if __name__ == "__main__":
    main()
