"""
src/models/campaign_manager.py

This is the "improving and improving" engine.

Workflow:
  1. User uploads a CSV of NEW well/phase records (same raw columns as
     the original dataset -- this is "actual execution data" from a
     completed campaign).
  2. We append it to the master raw dataset (data/raw/drilling_dataset.csv)
     and re-run feature engineering on the COMBINED data.
  3. We retrain all models on the combined data (train_all) and
     re-run error_analysis (coverage/calibration).
  4. We log a "campaign snapshot" (timestamp, row counts, MAE/R2/coverage
     per target) to reports/campaign_history.json.
  5. The dashboard's Retrain page plots these snapshots over time --
     THIS is the proof that the system improves campaign over campaign.

Nothing here is destructive: the original uploaded dataset stays in
data/raw/, and every new upload is appended (with a Campaign_ID tag).
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime

from src.features.feature_engineering import build_features
from src.models.train_models import train_all, TARGETS
from src.models.error_analysis import run_error_analysis

RAW_PATH = os.path.join("data", "raw", "drilling_dataset.csv")
FEATURES_PATH = os.path.join("data", "processed", "features_data.csv")
HISTORY_PATH = os.path.join("reports", "campaign_history.json")
PLAN_VS_ACTUAL_PATH = os.path.join("reports", "plan_vs_actual_log.csv")
MODELS_DIR = "models"

REQUIRED_RAW_COLUMNS = [
    "Well_ID", "Basin", "Well_Type", "Phase", "Formation",
    "Depth_From_m", "Depth_To_m", "Meterage_Drilled_m", "Duration_Hours",
    "NPT_Hours", "ROP_m_per_hr", "Mud_Weight_ppg", "Formation_Hardness",
    "Equipment_Failures", "Weather_Severity", "Daily_Rig_Rate_USD",
    "Materials_Cost_USD", "Service_Cost_USD", "Total_Cost_USD",
]


def validate_new_data(new_df: pd.DataFrame):
    """
    Check that the uploaded CSV has the columns we need. Returns
    (is_valid, message). Extra columns are fine and ignored.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in new_df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    if new_df[REQUIRED_RAW_COLUMNS].isnull().any().any():
        return False, "New data contains missing values in required columns -- please fill or remove those rows first."
    return True, "OK"


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history):
    os.makedirs("reports", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def predict_with_current_models(raw_new_df: pd.DataFrame):
    """
    Before retraining, run the CURRENT (pre-update) saved models on the
    new rows to get "what the system would have predicted" for these
    actuals. This is the genuine Plan-vs-Actual comparison: the plan
    was made with the OLD model, the actual is what really happened.

    Returns a DataFrame with one row per new record: Well_ID, Phase,
    actual values, and predicted values for each target (or None if
    no saved model exists yet, e.g. very first run).
    """
    feats = build_features(raw_new_df.copy())

    predictions = {t: [None] * len(feats) for t in TARGETS}
    for target in TARGETS:
        meta_path = os.path.join(MODELS_DIR, f"{target}_metadata.json")
        model_path = os.path.join(MODELS_DIR, f"{target}_best_model.pkl")
        if not (os.path.exists(meta_path) and os.path.exists(model_path)):
            continue
        meta = json.load(open(meta_path))
        model = joblib.load(model_path)
        feature_cols = meta["feature_columns"]
        cat_features = meta.get("cat_features", [])
        model_name = meta["best_model_name"]

        missing = [c for c in feature_cols if c not in feats.columns]
        if missing:
            continue

        X = feats[feature_cols].copy()
        if model_name == "CatBoost":
            for c in cat_features:
                X[c] = X[c].astype(str)
        preds = model.predict(X)
        predictions[target] = [max(0, float(p)) for p in preds]

    out = feats[["Well_ID", "Phase"]].copy() if "Well_ID" in feats.columns else pd.DataFrame(index=feats.index)
    for target in TARGETS:
        out[f"Actual_{target}"] = feats[target].values
        out[f"Predicted_{target}"] = predictions[target]
    return out


def run_campaign_update(new_df: pd.DataFrame, campaign_label: str = None):
    """
    Main entry point called from the dashboard.

    Returns a dict:
      {
        "campaign_label": ...,
        "n_new_rows": ...,
        "n_total_rows": ...,
        "metrics": {target: {...}},          # from train_all
        "error_analysis": {target: {...}},   # from run_error_analysis
        "plan_vs_actual": DataFrame,         # NEW: per-row comparison using PRE-retrain models
        "history": [...]                      # full history incl. this run
      }
    """
    is_valid, msg = validate_new_data(new_df)
    if not is_valid:
        raise ValueError(msg)

    if campaign_label is None:
        campaign_label = f"Campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 0. BEFORE retraining: use the CURRENT models to predict on the new
    #    rows. This captures "what we planned" vs "what actually happened".
    plan_vs_actual = predict_with_current_models(new_df)
    plan_vs_actual["Campaign_ID"] = campaign_label

    # 1. Load existing raw data, append new rows
    base_df = pd.read_csv(RAW_PATH)

    new_df = new_df.copy()
    new_df["Campaign_ID"] = campaign_label
    if "Campaign_ID" not in base_df.columns:
        base_df["Campaign_ID"] = "Campaign_0_original"

    cols_to_keep = REQUIRED_RAW_COLUMNS.copy()
    if "Campaign_ID" in new_df.columns:
        cols_to_keep = cols_to_keep + ["Campaign_ID"]

    new_df_aligned = new_df[cols_to_keep]

    combined = pd.concat([base_df, new_df_aligned], ignore_index=True, sort=False)

    # 2. Save updated raw dataset (so it's persistent across sessions)
    combined.to_csv(RAW_PATH, index=False)

    # 3. Rebuild engineered features on the FULL combined dataset
    feats = build_features(combined)
    feats.to_csv(FEATURES_PATH, index=False)

    # 4. Retrain all models on the full combined dataset
    train_summary = train_all(feats, save=True)

    # 5. Re-run error/calibration analysis
    err_summary = run_error_analysis(feats, save=True)

    # 6. Log this campaign snapshot
    history = load_history()
    snapshot = {
        "campaign_label": campaign_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_new_rows": int(len(new_df)),
        "n_total_rows": int(len(combined)),
        "n_total_wells": int(combined["Well_ID"].nunique()) if "Well_ID" in combined.columns else None,
        "metrics": {
            target: {
                "best_model": info["best_model"],
                "MAE": round(info["metrics"][info["best_model"]]["MAE"], 4),
                "RMSE": round(info["metrics"][info["best_model"]]["RMSE"], 4),
                "MAPE": round(info["metrics"][info["best_model"]]["MAPE"], 2),
                "R2": round(info["metrics"][info["best_model"]]["R2"], 4),
            }
            for target, info in train_summary.items()
        },
        "coverage": {
            target: err_summary[target]["coverage_pct_actual_in_p10_p90"]
            for target in err_summary
        },
        "spread": {
            target: err_summary[target]["avg_p90_minus_p10_spread"]
            for target in err_summary
        },
    }
    history.append(snapshot)
    save_history(history)

    # 7. Persist the plan-vs-actual log (append, so it accumulates across campaigns)
    if os.path.exists(PLAN_VS_ACTUAL_PATH):
        existing_log = pd.read_csv(PLAN_VS_ACTUAL_PATH)
        full_log = pd.concat([existing_log, plan_vs_actual], ignore_index=True)
    else:
        full_log = plan_vs_actual
    full_log.to_csv(PLAN_VS_ACTUAL_PATH, index=False)

    return {
        "campaign_label": campaign_label,
        "n_new_rows": len(new_df),
        "n_total_rows": len(combined),
        "metrics": train_summary,
        "error_analysis": err_summary,
        "plan_vs_actual": plan_vs_actual,
        "history": history,
    }


if __name__ == "__main__":
    # Demo: re-run the pipeline on the existing data as "Campaign_0"
    # to seed reports/campaign_history.json with a baseline entry.
    df = pd.read_csv(RAW_PATH)
    feats = build_features(df)
    feats.to_csv(FEATURES_PATH, index=False)
    train_summary = train_all(feats, save=True)
    err_summary = run_error_analysis(feats, save=True)

    history = []
    snapshot = {
        "campaign_label": "Campaign_0_baseline",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_new_rows": 0,
        "n_total_rows": int(len(df)),
        "n_total_wells": int(df["Well_ID"].nunique()),
        "metrics": {
            target: {
                "best_model": info["best_model"],
                "MAE": round(info["metrics"][info["best_model"]]["MAE"], 4),
                "RMSE": round(info["metrics"][info["best_model"]]["RMSE"], 4),
                "MAPE": round(info["metrics"][info["best_model"]]["MAPE"], 2),
                "R2": round(info["metrics"][info["best_model"]]["R2"], 4),
            }
            for target, info in train_summary.items()
        },
        "coverage": {t: err_summary[t]["coverage_pct_actual_in_p10_p90"] for t in err_summary},
        "spread": {t: err_summary[t]["avg_p90_minus_p10_spread"] for t in err_summary},
    }
    history.append(snapshot)
    save_history(history)
    print("Seeded reports/campaign_history.json with baseline snapshot.")
    print(json.dumps(snapshot, indent=2))
