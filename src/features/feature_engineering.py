"""
src/features/feature_engineering.py

This is the highest-leverage file in the project. The raw dataset has
19 columns. This module turns it into a model-ready dataset with
engineered features that capture drilling-domain knowledge.

Run:
    python src/features/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = os.path.join("data", "processed", "validated_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "features_data.csv")


def fix_npt_violation(df: pd.DataFrame) -> pd.DataFrame:
    """
    The ONE real data issue found during audit: 155 rows (3.1%) have
    NPT_Hours > Duration_Hours, which is physically impossible.

    Fix: cap NPT_Hours at Duration_Hours, and keep a flag column so the
    fix is auditable, not silently hidden.
    """
    df = df.copy()
    df["NPT_Was_Capped"] = (df["NPT_Hours"] > df["Duration_Hours"]).astype(int)
    df["NPT_Hours"] = np.minimum(df["NPT_Hours"], df["Duration_Hours"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven engineered features.

    Each feature is explained inline -- this is the part that matters
    most to a reviewer asking "why does this feature exist?"
    """
    df = df.copy()

    # 1. NPT Ratio: what fraction of phase time was non-productive.
    #    Normalizes NPT across phases of very different lengths.
    df["NPT_Ratio"] = df["NPT_Hours"] / df["Duration_Hours"]

    # 2. Productive Hours: actual working time in the phase.
    df["Productive_Hours"] = df["Duration_Hours"] - df["NPT_Hours"]

    # 3. Cost per Hour: cost efficiency, comparable across phases.
    df["Cost_Per_Hour"] = df["Total_Cost_USD"] / df["Duration_Hours"]

    # 4. Cost per Meter: only meaningful when meterage > 0 (Drilling phase).
    #    Use a safe divide -> 0 when Meterage is 0 (non-drilling phases).
    df["Cost_Per_Meter"] = np.where(
        df["Meterage_Drilled_m"] > 0,
        df["Total_Cost_USD"] / df["Meterage_Drilled_m"],
        0.0
    )

    # 5. Failure Density: equipment failures normalized by time.
    #    Two wells can both have "1 failure" but if one took 5 hours and
    #    the other 20 hours, the failure RATE is very different.
    df["Failure_Density"] = df["Equipment_Failures"] / df["Duration_Hours"]

    # 6. Weather Impact Score: interaction term. A storm during a long
    #    phase matters more than a storm during a short phase.
    df["Weather_Impact"] = df["Weather_Severity"] * df["Duration_Hours"]

    # 7. Drilling Efficiency: ROP normalized by formation hardness.
    #    A high ROP in soft formation is "expected"; a high ROP in hard
    #    formation is genuinely impressive. Safe-divide for non-drilling
    #    rows where ROP=0.
    df["Drilling_Efficiency"] = np.where(
        df["Formation_Hardness"] > 0,
        df["ROP_m_per_hr"] / df["Formation_Hardness"],
        0.0
    )

    # 8. Depth Interval: more meaningful than raw absolute depth.
    df["Depth_Interval_m"] = df["Depth_To_m"] - df["Depth_From_m"]

    # 9. Is_Drilling_Phase: binary flag. Many features (ROP, Meterage,
    #    Cost_Per_Meter, Drilling_Efficiency) are ONLY non-zero here.
    df["Is_Drilling_Phase"] = (df["Phase"] == "Drilling").astype(int)

    # 10. Total Daily Cost (sum of the 3 cost components) -- sanity
    #     cross-check column. Should correlate strongly with Total_Cost_USD.
    df["Sum_Cost_Components"] = (
        df["Daily_Rig_Rate_USD"] + df["Materials_Cost_USD"] + df["Service_Cost_USD"]
    )

    # 11. Phase order within each well (1-25). Required if you later move
    #     to a sequence model (LSTM), and useful even for tree models as
    #     a "where in the campaign are we" signal.
    df = df.sort_values(["Well_ID"]).copy()
    df["Phase_Order"] = df.groupby("Well_ID").cumcount() + 1

    # 12. Cumulative duration and cumulative NPT up to (and including)
    #     this phase. Captures "how much trouble has this well already
    #     had by this point" -- a leading indicator for later phases.
    df["Cumulative_Duration"] = df.groupby("Well_ID")["Duration_Hours"].cumsum()
    df["Cumulative_NPT"] = df.groupby("Well_ID")["NPT_Hours"].cumsum()

    # 13. Previous phase NPT (lag feature). NPT tends to cluster --
    #     equipment problems in one phase often bleed into the next.
    df["Previous_Phase_NPT"] = df.groupby("Well_ID")["NPT_Hours"].shift(1).fillna(0)

    # 14. Basin_Formation combined categorical: captures interaction
    #     between geography and geology (e.g. "North Sea + Shale" behaves
    #     differently than "Permian + Shale").
    df["Basin_Formation"] = df["Basin"] + "_" + df["Formation"]

    return df


def encode_categoricals(df: pd.DataFrame, target_col: str = "Total_Cost_USD") -> pd.DataFrame:
    """
    Target-encode the main categorical columns (mean of target per category).
    This is simple, works very well with tree models (XGBoost/CatBoost/RF),
    and avoids the huge sparse one-hot columns you'd get from Basin_Formation.

    NOTE: For CatBoost you don't strictly need this (CatBoost handles raw
    categoricals natively) -- but having both raw and encoded versions lets
    you run a fair comparison across all three models.
    """
    df = df.copy()
    cat_cols = ["Basin", "Well_Type", "Phase", "Formation", "Basin_Formation"]

    for col in cat_cols:
        means = df.groupby(col)[target_col].mean()
        df[col + "_Encoded"] = df[col].map(means)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline as a single callable function:
    fix NPT violation -> add engineered features -> encode categoricals.
    Used by both the CLI script and the dashboard's retrain pipeline.
    """
    df = fix_npt_violation(df)
    df = add_engineered_features(df)
    df = encode_categoricals(df)
    return df


def run_pipeline():
    print("Loading validated data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nFixing NPT > Duration violation (155 rows expected)...")
    df = fix_npt_violation(df)
    print(f"  Rows flagged as capped: {df['NPT_Was_Capped'].sum()}")

    print("\nAdding engineered features...")
    df = add_engineered_features(df)
    new_cols = [
        "NPT_Ratio", "Productive_Hours", "Cost_Per_Hour", "Cost_Per_Meter",
        "Failure_Density", "Weather_Impact", "Drilling_Efficiency",
        "Depth_Interval_m", "Is_Drilling_Phase", "Sum_Cost_Components",
        "Phase_Order", "Cumulative_Duration", "Cumulative_NPT",
        "Previous_Phase_NPT", "Basin_Formation", "NPT_Was_Capped"
    ]
    print(f"  Added {len(new_cols)} new features:")
    for c in new_cols:
        print(f"    - {c}")

    print("\nEncoding categorical columns (target-encoding)...")
    df = encode_categoricals(df)

    print(f"\nFinal shape: {df.shape[0]} rows x {df.shape[1]} columns")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved feature-engineered data to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    run_pipeline()
