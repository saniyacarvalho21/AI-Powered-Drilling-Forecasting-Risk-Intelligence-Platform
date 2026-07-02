"""
src/monte_carlo/monte_carlo_simulation.py

Hybrid Monte Carlo: instead of sampling from a generic assumed
distribution (the "traditional" approach), we sample from the trained
model's ACTUAL residual distribution on held-out test data.

  point_prediction = model.predict(new_well_features)
  simulated_outcomes = point_prediction + sample(residuals, N times)
  P10, P50, P90 = percentiles of simulated_outcomes

Why this is better than traditional Monte Carlo:
  - Traditional MC assumes a shape (Normal/Lognormal) for the WHOLE
    population, ignoring the specific well's features.
  - This approach centers the distribution on a feature-aware ML
    prediction, then adds the model's *known* error pattern around it.
  - As the model improves (more campaign data -> retraining), the
    residual distribution narrows -> P90-P10 spread shrinks. This is
    the "self-improving" mechanism requested for the project.
"""

import numpy as np
import json
import os

N_SIMULATIONS = 10000


def run_monte_carlo(point_prediction: float, residuals: list, n_sim: int = N_SIMULATIONS, seed: int = 42):
    """
    Generate a simulated outcome distribution around a point prediction
    using bootstrapped residuals from the model's test-set errors.

    Returns dict with P10, P50, P90 and the full simulated array.
    """
    rng = np.random.default_rng(seed)
    residuals = np.array(residuals)

    sampled_residuals = rng.choice(residuals, size=n_sim, replace=True)
    simulated = point_prediction + sampled_residuals

    # Physical lower bound: outcomes (duration, cost, NPT) cannot be negative
    simulated = np.clip(simulated, a_min=0, a_max=None)

    p10, p50, p90 = np.percentile(simulated, [10, 50, 90])

    return {
        "point_prediction": float(point_prediction),
        "P10": float(p10),
        "P50": float(p50),
        "P90": float(p90),
        "spread": float(p90 - p10),
        "simulated": simulated,
    }


def coverage_check(actuals: np.ndarray, p10_array: np.ndarray, p90_array: np.ndarray):
    """
    Calibration check: what fraction of ACTUAL outcomes fell inside
    the [P10, P90] band? A well-calibrated 80% interval should cover
    ~80% of actuals. This is the metric you track campaign-over-campaign
    to prove the model is "improving and improving".
    """
    inside = (actuals >= p10_array) & (actuals <= p90_array)
    return float(np.mean(inside) * 100)


if __name__ == "__main__":
    # Demo: run Monte Carlo for one example using saved residuals
    for target in ["Duration_Hours", "Total_Cost_USD", "NPT_Hours"]:
        meta_path = os.path.join("models", f"{target}_metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        residuals = meta["residuals"]

        # Example: pretend a new well's point prediction equals the
        # mean of the test predictions (just for demonstration)
        example_pred = float(np.mean(residuals)) * 0  # placeholder center
        # Use a realistic center: median residual + typical value
        # (in real use, this comes from model.predict() on new data)
        center_lookup = {
            "Duration_Hours": 12.0,
            "Total_Cost_USD": 62000.0,
            "NPT_Hours": 1.9,
        }
        example_pred = center_lookup[target]

        result = run_monte_carlo(example_pred, residuals)
        print(f"\n{target}")
        print(f"  Point prediction: {result['point_prediction']:.2f}")
        print(f"  P10: {result['P10']:.2f}")
        print(f"  P50: {result['P50']:.2f}")
        print(f"  P90: {result['P90']:.2f}")
        print(f"  Spread (P90-P10): {result['spread']:.2f}")
