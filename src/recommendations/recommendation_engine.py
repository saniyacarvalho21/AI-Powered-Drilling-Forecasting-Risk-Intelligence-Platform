"""
src/recommendations/recommendation_engine.py

Simple, transparent rule-based recommendations. Version 1 is
deliberately rule-based (not ML) because:
  - It's instantly explainable to non-technical managers.
  - It's testable and auditable (no black box).
  - It can be built in hours, not days -- fits the 2-week timeline.
  - It can be replaced with a learned policy (Reinforcement Learning)
    in a future version, once enough campaign data exists.

Each rule maps a risk condition (derived from input features and/or
model predictions) to a concrete operational recommendation.
"""

from typing import Dict, List


def generate_recommendations(inputs: Dict) -> List[Dict[str, str]]:
    """
    inputs: dict with keys such as
        Weather_Severity, Equipment_Failures, Formation_Hardness,
        NPT_Ratio_Predicted, Phase, Cost_Per_Hour_Predicted, etc.

    Returns a list of {severity, message, action} dicts.
    """
    recs = []

    weather = inputs.get("Weather_Severity", 0)
    failures = inputs.get("Equipment_Failures", 0)
    hardness = inputs.get("Formation_Hardness", 0)
    npt_ratio = inputs.get("NPT_Ratio_Predicted", 0)
    phase = inputs.get("Phase", "")
    p90_p10_spread_pct = inputs.get("Spread_Pct_of_P50", None)

    # Rule 1: Weather risk
    if weather >= 7:
        recs.append({
            "severity": "High",
            "trigger": f"Weather_Severity = {weather} (>= 7)",
            "message": "Severe weather conditions expected.",
            "action": "Add 1-2 contingency days to the schedule for this phase "
                      "and pre-position weather-sensitive equipment early."
        })
    elif weather >= 5:
        recs.append({
            "severity": "Medium",
            "trigger": f"Weather_Severity = {weather} (5-6)",
            "message": "Moderate weather risk.",
            "action": "Monitor forecast closely; keep a flexible buffer of "
                      "0.5-1 day in the plan."
        })

    # Rule 2: Equipment failure risk
    if failures >= 2:
        recs.append({
            "severity": "High",
            "trigger": f"Equipment_Failures = {failures} (>= 2)",
            "message": "High equipment failure history for this phase type.",
            "action": "Schedule preventive maintenance before this phase begins "
                      "and ensure spare parts/backup equipment are on standby."
        })
    elif failures == 1:
        recs.append({
            "severity": "Medium",
            "trigger": "Equipment_Failures = 1",
            "message": "At least one equipment issue is likely.",
            "action": "Pre-check critical equipment (top drive, mud pumps, BOP) "
                      "before starting this phase."
        })

    # Rule 3: Formation hardness risk (mainly relevant for Drilling phase)
    if phase == "Drilling" and hardness >= 7:
        recs.append({
            "severity": "High",
            "trigger": f"Formation_Hardness = {hardness} (>= 7) during Drilling",
            "message": "Hard formation expected -- ROP will likely be low and "
                       "bit wear will increase.",
            "action": "Plan for a more aggressive bit (PDC with higher cutter "
                      "density) and budget extra time for potential bit trips."
        })

    # Rule 4: NPT ratio risk (predicted)
    if npt_ratio is not None and npt_ratio >= 0.20:
        recs.append({
            "severity": "High",
            "trigger": f"Predicted NPT Ratio = {npt_ratio:.1%} (>= 20%)",
            "message": "Non-productive time is predicted to be unusually high "
                       "for this phase.",
            "action": "Review crew shift handover procedures and ensure all "
                      "required materials/permits are pre-staged to avoid "
                      "waiting time."
        })

    # Rule 5: Forecast uncertainty risk (Monte Carlo spread)
    if p90_p10_spread_pct is not None and p90_p10_spread_pct >= 50:
        recs.append({
            "severity": "Medium",
            "trigger": f"P90-P10 spread = {p90_p10_spread_pct:.0f}% of P50",
            "message": "Forecast uncertainty for this phase is wide.",
            "action": "Treat the P50 estimate as indicative only for "
                      "scheduling commitments; use P90 for contractual "
                      "commitments to avoid penalty exposure."
        })

    if not recs:
        recs.append({
            "severity": "Low",
            "trigger": "No risk thresholds triggered",
            "message": "Conditions look within normal historical range.",
            "action": "Proceed with standard planning assumptions (P50)."
        })

    return recs


if __name__ == "__main__":
    example = {
        "Weather_Severity": 8,
        "Equipment_Failures": 2,
        "Formation_Hardness": 8,
        "NPT_Ratio_Predicted": 0.25,
        "Phase": "Drilling",
        "Spread_Pct_of_P50": 60,
    }
    for r in generate_recommendations(example):
        print(f"[{r['severity']}] {r['message']}")
        print(f"    Trigger: {r['trigger']}")
        print(f"    Action:  {r['action']}\n")
