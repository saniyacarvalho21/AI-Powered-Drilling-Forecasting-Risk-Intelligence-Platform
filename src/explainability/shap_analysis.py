"""
src/explainability/shap_analysis.py

Computes SHAP values for the best model of each target, so the
dashboard can show "WHY" a prediction was made -- not just the number.

Saves:
  reports/shap_<target>.json  -> top feature importances (mean |SHAP|)
  reports/shap_values_<target>.npz -> raw shap values + feature matrix
                                       for waterfall plots in the dashboard
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import shap

DATA_PATH = os.path.join("data", "processed", "features_data.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"

TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]


def compute_shap_for_target(df: pd.DataFrame, target: str, sample_size: int = 500):
    meta_path = os.path.join(MODELS_DIR, f"{target}_metadata.json")
    model_path = os.path.join(MODELS_DIR, f"{target}_best_model.pkl")

    with open(meta_path) as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    feature_cols = meta["feature_columns"]
    cat_features = meta.get("cat_features", [])
    model_name = meta["best_model_name"]

    X = df[feature_cols].copy()
    if model_name == "CatBoost":
        for c in cat_features:
            X[c] = X[c].astype(str)

    # Sample for speed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X


    if model_name == "LinearRegression":
        # Use a small background sample for KernelExplainer-free linear SHAP
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    elif model_name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_name == "CatBoost":
        explainer = shap.TreeExplainer(model)
        # CatBoost needs the Pool for categorical handling, but
        # TreeExplainer on CatBoost works with raw dataframe too
        shap_values = explainer.shap_values(X_sample)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    shap_values = np.array(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_cols, mean_abs_shap.tolist()),
        key=lambda x: x[1], reverse=True
    )

    return {
        "target": target,
        "model": model_name,
        "feature_importance": importance[:15],  # top 15
    }, shap_values, X_sample


def main():
    df = pd.read_csv(DATA_PATH)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    all_importance = {}

    for target in TARGETS:
        print(f"\nComputing SHAP for {target}...")
        try:
            result, shap_values, X_sample = compute_shap_for_target(df, target)
            all_importance[target] = result

            print(f"  Model: {result['model']}")
            print("  Top 5 drivers:")
            for feat, val in result["feature_importance"][:5]:
                print(f"    {feat:<25} {val:.3f}")

            # Save raw shap values + feature matrix for dashboard waterfall plots
            np.savez(
                os.path.join(REPORTS_DIR, f"shap_values_{target}.npz"),
                shap_values=shap_values,
                feature_names=np.array(X_sample.columns),
                X_sample=X_sample.values.astype(object)
            )
        except Exception as e:
            print(f"  SHAP failed for {target}: {e}")
            all_importance[target] = {"target": target, "error": str(e)}

    with open(os.path.join(REPORTS_DIR, "shap_summary.json"), "w") as f:
        json.dump(all_importance, f, indent=2)

    print(f"\nSaved SHAP summary -> reports/shap_summary.json")


if __name__ == "__main__":
    main()
