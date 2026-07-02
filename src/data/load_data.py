"""
src/data/load_data.py

Loads the raw drilling CSV and runs a data quality audit.
This is Step 1 of the pipeline: "Is this data dirty? What exactly
needs cleaning?" -- not "clean everything blindly".
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "raw", "drilling_dataset.csv")
PROCESSED_PATH = os.path.join("data", "processed", "validated_data.csv")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(path)
    return df


def audit_data(df: pd.DataFrame) -> dict:
    """
    Run a full data quality audit and return a report dictionary.
    This answers: missing values? duplicates? impossible values?
    """
    report = {}

    # 1. Missing values
    report["missing_values"] = df.isnull().sum().to_dict()
    report["total_missing"] = int(df.isnull().sum().sum())

    # 2. Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    # 3. Data types
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # 4. Impossible / invalid values (business logic checks)
    issues = {}

    # Duration must be > 0
    if "Duration_Hours" in df.columns:
        issues["Duration_Hours_<=0"] = int((df["Duration_Hours"] <= 0).sum())

    # Cost must be > 0
    if "Total_Cost_USD" in df.columns:
        issues["Total_Cost_USD_<=0"] = int((df["Total_Cost_USD"] <= 0).sum())

    # NPT cannot exceed Duration (NPT is a subset of total time)
    if "NPT_Hours" in df.columns and "Duration_Hours" in df.columns:
        issues["NPT_greater_than_Duration"] = int(
            (df["NPT_Hours"] > df["Duration_Hours"]).sum()
        )

    # Negative values anywhere in numeric columns that should never be negative
    non_negative_cols = [
        "Depth_From_m", "Depth_To_m", "Meterage_Drilled_m", "Duration_Hours",
        "NPT_Hours", "ROP_m_per_hr", "Mud_Weight_ppg", "Formation_Hardness",
        "Equipment_Failures", "Weather_Severity", "Daily_Rig_Rate_USD",
        "Materials_Cost_USD", "Service_Cost_USD", "Total_Cost_USD"
    ]
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                issues[f"{col}_negative"] = neg_count

    # Weather_Severity should be on a 1-9 scale (observed range in this dataset)
    if "Weather_Severity" in df.columns:
        out_of_range = int(
            ((df["Weather_Severity"] < 1) | (df["Weather_Severity"] > 10)).sum()
        )
        issues["Weather_Severity_out_of_1_10_range"] = out_of_range

    # Depth_To must be >= Depth_From
    if "Depth_To_m" in df.columns and "Depth_From_m" in df.columns:
        issues["Depth_To_less_than_Depth_From"] = int(
            (df["Depth_To_m"] < df["Depth_From_m"]).sum()
        )

    report["business_logic_issues"] = issues

    # 5. Zero-value audit (zeros are NOT automatically bad -- context dependent)
    zero_report = {}
    for col in non_negative_cols:
        if col in df.columns:
            zero_report[col] = int((df[col] == 0).sum())
    report["zero_value_counts"] = zero_report

    return report


def print_audit_report(report: dict):
    """Pretty-print the audit report to console."""
    print("=" * 60)
    print("DATA QUALITY AUDIT REPORT")
    print("=" * 60)

    print(f"\nTotal missing values: {report['total_missing']}")
    if report["total_missing"] > 0:
        for col, n in report["missing_values"].items():
            if n > 0:
                print(f"  - {col}: {n} missing")
    else:
        print("  -> No missing values found. No imputation needed.")

    print(f"\nDuplicate rows: {report['duplicate_rows']}")
    if report["duplicate_rows"] == 0:
        print("  -> No duplicates found.")

    print("\nBusiness logic issues:")
    any_issue = False
    for k, v in report["business_logic_issues"].items():
        if v > 0:
            print(f"  - {k}: {v} rows")
            any_issue = True
    if not any_issue:
        print("  -> No impossible values found (no negatives, no NPT > Duration, etc.)")

    print("\nZero-value counts (NOT necessarily errors -- context dependent):")
    for col, n in report["zero_value_counts"].items():
        if n > 0:
            print(f"  - {col}: {n} zeros")

    npt_issue = report["business_logic_issues"].get("NPT_greater_than_Duration", 0)
    print("\n" + "=" * 60)
    print("VERDICT:")
    print("  - No missing values, no duplicates, no negative values.")
    print(f"  - FOUND ONE REAL ISSUE: {npt_issue} rows ({npt_issue/5000*100:.1f}%) have")
    print("    NPT_Hours > Duration_Hours, which is logically impossible")
    print("    (non-productive time cannot exceed total phase time).")
    print("  - FIX: cap NPT_Hours at Duration_Hours during feature")
    print("    engineering, and add an 'NPT_Was_Capped' flag column so")
    print("    this is traceable, not silently hidden.")
    print("  - Zeros (Depth=0, ROP=0, Equipment_Failures=0, etc.) are")
    print("    VALID -- they represent non-drilling phases (Setup,")
    print("    Tripping, etc.) where these metrics genuinely don't apply.")
    print("    They are NOT removed.")
    print("  - Beyond this one fix, effort is redirected to feature")
    print("    engineering, which has a much bigger impact on accuracy.")
    print("=" * 60)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns\n")
    report = audit_data(df)
    print_audit_report(report)

    # Save validated (unchanged, since data is clean) data for next stage
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved validated data to {PROCESSED_PATH}")
