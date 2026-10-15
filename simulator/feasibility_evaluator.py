"""V1 heuristic feasibility evaluator for RaceSimulator output.
Statuses are provisional screening labels, not proof of physical tyre feasibility.
"""

from typing import Any, Dict, List

DEFAULT_REVIEW_THRESHOLD = 5.0
DEFAULT_REJECT_THRESHOLD = 10.0


def evaluate_strategy(
    simulation_result: Dict[str, Any],
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate model predictions against race-distance allocations.

    FEASIBLE: all absolute gaps <= review threshold and lap budget is valid.
    REVIEW: any gap > review threshold, but none > reject threshold.
    INFEASIBLE: invalid race-lap budget or any gap > reject threshold.

    """
    if review_threshold < 0 or reject_threshold < 0:
        raise ValueError("Thresholds must be non-negative.")
    if review_threshold > reject_threshold:
        raise ValueError("review_threshold must be <= reject_threshold.")
    if not isinstance(simulation_result, dict):
        raise TypeError("simulation_result must be a dictionary.")

    for key in ("stints", "budget_valid", "race_laps", "total_allocated_laps"):
        if key not in simulation_result:
            raise ValueError(f"Simulation result is missing required key: {key}")

    stints = simulation_result["stints"]
    if not isinstance(stints, list) or not stints:
        raise ValueError("simulation_result['stints'] must be a non-empty list.")

    evaluated: List[Dict[str, Any]] = []
    gaps: List[float] = []
    reasons: List[str] = []

    for index, stint in enumerate(stints, start=1):
        if not isinstance(stint, dict):
            raise TypeError(f"Stint entry {index} must be a dictionary.")
        for key in ("stint_number", "compound", "predicted_stint_length", "required_stint_length"):
            if key not in stint:
                raise ValueError(f"Stint entry {index} is missing key: {key}")

        predicted = float(stint["predicted_stint_length"])
        required = float(stint["required_stint_length"])
        if predicted < 0 or required < 0:
            raise ValueError(f"Stint entry {index} contains a negative lap value.")

        signed_gap = required - predicted
        gap = abs(signed_gap)
        gaps.append(gap)

        if gap > reject_threshold:
            stint_status = "INFEASIBLE"
            reasons.append(
                f"Stint {stint['stint_number']} ({stint['compound']}): "
                f"{gap:.1f}-lap absolute gap exceeds the {reject_threshold:.1f}-lap reject threshold."
            )
        elif gap > review_threshold:
            stint_status = "REVIEW"
            reasons.append(
                f"Stint {stint['stint_number']} ({stint['compound']}): "
                f"{gap:.1f}-lap absolute gap exceeds the {review_threshold:.1f}-lap review threshold."
            )
        else:
            stint_status = "PASS"

        evaluated.append({
            "stint_number": int(stint["stint_number"]),
            "compound": str(stint["compound"]),
            "predicted_stint_length": round(predicted, 1),
            "required_stint_length": round(required, 1),
            "signed_gap": round(signed_gap, 1),
            "absolute_gap": round(gap, 1),
            "is_final_stint": bool(stint.get("is_final_stint", False)),
            "status": stint_status,
        })

    race_laps = int(simulation_result["race_laps"])
    allocated = int(simulation_result["total_allocated_laps"])
    budget_valid = bool(simulation_result["budget_valid"]) and allocated == race_laps

    if not budget_valid:
        status = "INFEASIBLE"
        reasons.insert(0, f"Race-distance budget failed: allocated {allocated} laps for a {race_laps}-lap race.")
    elif any(gap > reject_threshold for gap in gaps):
        status = "INFEASIBLE"
    elif any(gap > review_threshold for gap in gaps):
        status = "REVIEW"
    else:
        status = "FEASIBLE"

    final_stint = next((s for s in evaluated if s["is_final_stint"]), evaluated[-1])
    final_gap = final_stint["absolute_gap"]
    if final_gap > review_threshold:
        reasons.append(
            f"Final stint ({final_stint['compound']}): allocation is "
            f"{final_stint['required_stint_length']:.1f} laps versus a model prediction of "
            f"{final_stint['predicted_stint_length']:.1f} laps (absolute gap {final_gap:.1f})."
        )
    if not reasons:
        reasons.append("All stint gaps are within the provisional review threshold and the race-lap budget is valid.")

    return {
        "gp_name": simulation_result.get("gp_name"),
        "strategy": simulation_result.get("strategy"),
        "strategy_label": simulation_result.get("strategy_label"),
        "status": status,
        "race_laps": race_laps,
        "total_allocated_laps": allocated,
        "budget_valid": budget_valid,
        "pit_stops": simulation_result.get("pit_stops", max(0, len(stints) - 1)),
        "max_absolute_gap": round(max(gaps), 1),
        "mean_absolute_gap": round(sum(gaps) / len(gaps), 1),
        "final_stint_absolute_gap": round(final_gap, 1),
        "thresholds": {
            "review_threshold": float(review_threshold),
            "reject_threshold": float(reject_threshold),
            "calibration": "provisional_v1_heuristic",
        },
        "stints": evaluated,
        "reasons": reasons,
        "interpretation": (
            "Heuristic consistency screen comparing model predictions with race-distance allocations. "
            "It does not establish actual tyre life, degradation limits, or physical strategy feasibility."
        ),
    }


if __name__ == "__main__":
    sample = {
        "gp_name": "Bahrain",
        "strategy": ["MEDIUM", "HARD", "SOFT"],
        "strategy_label": "MEDIUM -> HARD -> SOFT",
        "stints": [
            {"stint_number": 1, "compound": "MEDIUM", "predicted_stint_length": 12.8,
             "required_stint_length": 13, "is_final_stint": False},
            {"stint_number": 2, "compound": "HARD", "predicted_stint_length": 27.9,
             "required_stint_length": 28, "is_final_stint": False},
            {"stint_number": 3, "compound": "SOFT", "predicted_stint_length": 15.5,
             "required_stint_length": 16, "is_final_stint": True},
        ],
        "race_laps": 57, "total_allocated_laps": 57, "budget_valid": True, "pit_stops": 2,
    }
    result = evaluate_strategy(sample)
    print("FEASIBILITY EVALUATION")
    print("=" * 45)
    print(f"Strategy: {result['strategy_label']}")
    print(f"Status: {result['status']}")
    print(f"Maximum absolute gap: {result['max_absolute_gap']} laps")
    print(f"Mean absolute gap: {result['mean_absolute_gap']} laps")
    print(f"Final-stint absolute gap: {result['final_stint_absolute_gap']} laps")
    print(f"Race-distance budget valid: {result['budget_valid']}")
    for reason in result["reasons"]:
        print(f"- {reason}")
