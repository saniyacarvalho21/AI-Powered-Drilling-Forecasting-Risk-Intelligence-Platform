"""
src/models/error_analysis.py

Goes beyond MAE/RMSE/MAPE/R2 (already saved by train_models.py) to
answer the question that matters for a planning tool:

  "If the model says P10=X and P90=Y, do 80% of REAL outcomes actually
   fall between X and Y?"

This is the CALIBRATION / COVERAGE check. It is computed once here on
the held-out test set and saved for the dashboard's "Error Analysis"
page. It is also the metric you re-compute after every campaign to
prove the system is "improving and improving".
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings

warnings.filterwarnings("ignore", message="X has feature names")

from src.monte_carlo.monte_carlo_simulation import run_monte_carlo  # noqa

DATA_PATH = os.path.join("data", "processed", "features_data.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
TARGETS = ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]


def run_error_analysis(df: pd.DataFrame, save: bool = True):
    report = {}

    for target in TARGETS:
        meta_path = os.path.join(MODELS_DIR, f"{target}_metadata.json")
        model_path = os.path.join(MODELS_DIR, f"{target}_best_model.pkl")

        with open(meta_path) as f:
            meta = json.load(f)

        model = joblib.load(model_path)
        feature_cols = meta["feature_columns"]
        cat_features = meta.get("cat_features", [])
        model_name = meta["best_model_name"]
        residuals = np.array(meta["residuals"])

        # Use the same test wells as training for an honest re-check
        test_well_ids = meta.get("test_well_ids")
        if test_well_ids:
            test_df = df[df["Well_ID"].isin(test_well_ids)].copy()
        else:
            test_df = df.sample(frac=0.2, random_state=42)

        X_test = test_df[feature_cols].copy()
        if model_name == "CatBoost":
            for c in cat_features:
                X_test[c] = X_test[c].astype(str)

        y_true = test_df[target].values
        y_pred = model.predict(X_test)

        # For each test row, run Monte Carlo using the GLOBAL residual
        # distribution (leave-one-out would be more rigorous but this
        # is a reasonable v1 approximation) and check coverage.
        rng = np.random.default_rng(42)
        n_sim = 2000
        p10_arr, p90_arr = [], []
        for pred in y_pred:
            sampled = rng.choice(residuals, size=n_sim, replace=True)
            sim = np.clip(pred + sampled, a_min=0, a_max=None)
            p10, p90 = np.percentile(sim, [10, 90])
            p10_arr.append(p10)
            p90_arr.append(p90)
        p10_arr = np.array(p10_arr)
        p90_arr = np.array(p90_arr)

        coverage = float(np.mean((y_true >= p10_arr) & (y_true <= p90_arr)) * 100)
        avg_spread = float(np.mean(p90_arr - p10_arr))
        avg_pred = float(np.mean(y_pred))
        spread_pct_of_pred = (avg_spread / avg_pred * 100) if avg_pred != 0 else None

        report[target] = {
            "model": model_name,
            "metrics": meta["metrics"][model_name],
            "coverage_pct_actual_in_p10_p90": round(coverage, 2),
            "target_coverage_pct": 80,
            "avg_p90_minus_p10_spread": round(avg_spread, 2),
            "avg_spread_pct_of_prediction": round(spread_pct_of_pred, 2) if spread_pct_of_pred else None,
            "n_test_rows": int(len(test_df)),
        }

        print(f"\n{target}")
        print(f"  Model: {model_name}")
        print(f"  MAE: {report[target]['metrics']['MAE']:.3f}  "
              f"RMSE: {report[target]['metrics']['RMSE']:.3f}  "
              f"MAPE: {report[target]['metrics']['MAPE']:.2f}%  "
              f"R2: {report[target]['metrics']['R2']:.3f}")
        print(f"  Coverage (actual within P10-P90): {coverage:.1f}% (target: 80%)")
        print(f"  Avg P90-P10 spread: {avg_spread:.2f} ({spread_pct_of_pred:.1f}% of prediction)")

        if coverage < 70:
            verdict = "UNDERCONFIDENT-RISK: too many actuals fall outside the band -> widen residual sampling or retrain."
        elif coverage > 90:
            verdict = "OVERCONFIDENT-WASTE: band is wider than needed -> model is more accurate than the band suggests."
        else:
            verdict = "WELL-CALIBRATED: band width is appropriate for an 80% interval."
        report[target]["calibration_verdict"] = verdict
        print(f"  Verdict: {verdict}")

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(os.path.join(REPORTS_DIR, "error_analysis.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved -> reports/error_analysis.json")

    return report


def main():
    df = pd.read_csv(DATA_PATH)
    run_error_analysis(df, save=True)


if __name__ == "__main__":
    main()
