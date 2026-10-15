import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from simulator.race_simulator import RaceSimulator
from simulator.feasibility_evaluator import evaluate_strategy


def main():
    # 1. Define the race context
    race_context = {
        "gp_name": "Bahrain",
        "race_laps": 57,
        "air_temp": 25.0,
        "track_temp": 30.0,
        "season": 2025,
        "circuit_type": "Permanent",
        "track_length": 5.412,
        "track_abrasiveness": "Medium",
        "average_corner_speed": 180.0,
    }

    # 2. Define the strategy to test
    strategy = ["MEDIUM", "HARD", "SOFT"]

    # 3. Initialize the simulator
    simulator = RaceSimulator()

    # 4. Simulate the strategy
    simulation_result = simulator.simulate_strategy(
        strategy,
        race_context
    )

    # 5. Pass the simulation output to the evaluator
    evaluation = evaluate_strategy(simulation_result)

    # 6. Display the results
    print("\nSIMULATOR + EVALUATOR INTEGRATION TEST")
    print("=" * 45)

    print(f"Grand Prix: {evaluation['gp_name']}")
    print(f"Strategy: {evaluation['strategy_label']}")
    print(f"Overall status: {evaluation['status']}")
    print(f"Race distance: {evaluation['race_laps']} laps")
    print(f"Allocated laps: {evaluation['total_allocated_laps']}")
    print(f"Budget valid: {evaluation['budget_valid']}")
    print(f"Maximum absolute gap: {evaluation['max_absolute_gap']} laps")
    print(f"Mean absolute gap: {evaluation['mean_absolute_gap']} laps")
    print(
        f"Final-stint absolute gap: "
        f"{evaluation['final_stint_absolute_gap']} laps"
    )

    print("\nSTINT-BY-STINT RESULTS")
    print("-" * 45)

    for stint in evaluation["stints"]:
        print(
            f"Stint {stint['stint_number']} | "
            f"{stint['compound']} | "
            f"Predicted: {stint['predicted_stint_length']} | "
            f"Allocated: {stint['required_stint_length']} | "
            f"Absolute gap: {stint['absolute_gap']} | "
            f"Status: {stint['status']}"
        )

    print("\nREASONS")
    print("-" * 45)

    for reason in evaluation["reasons"]:
        print(f"- {reason}")

    # 7. Basic integration assertions
    assert evaluation["gp_name"] == "Bahrain"
    assert len(evaluation["stints"]) == len(strategy)
    assert evaluation["total_allocated_laps"] == race_context["race_laps"]
    assert evaluation["budget_valid"] is True
    # 8. Test large prediction discrepancy
    import copy

    print("\nTEST 2: Large Prediction Discrepancy")
    print("-" * 45)

    test_result = copy.deepcopy(simulation_result)

    # Force a 15-lap discrepancy in the final stint
    final_stint = test_result["stints"][-1]
    final_stint["required_stint_length"] = (
        final_stint["predicted_stint_length"] + 15
    )

    # Evaluate using the default 5- and 10-lap thresholds
    discrepancy_evaluation = evaluate_strategy(test_result)

    print(f"Status: {discrepancy_evaluation['status']}")
    print(
        f"Maximum absolute gap: "
        f"{discrepancy_evaluation['max_absolute_gap']} laps"
    )

    assert discrepancy_evaluation["status"] == "INFEASIBLE"
    assert discrepancy_evaluation["max_absolute_gap"] > 10

    print("Large discrepancy test passed.")
        # 9. Test invalid race-distance budget
    print("\nTEST 3: Invalid Race-Distance Budget")
    print("-" * 45)

    budget_test_result = copy.deepcopy(simulation_result)

    # Deliberately invalidate the reported budget
    budget_test_result["total_allocated_laps"] -= 1
    budget_test_result["budget_valid"] = False

    budget_evaluation = evaluate_strategy(budget_test_result)

    print(f"Status: {budget_evaluation['status']}")
    print(f"Budget valid: {budget_evaluation['budget_valid']}")

    assert budget_evaluation["status"] == "INFEASIBLE"
    assert budget_evaluation["budget_valid"] is False

    print("Invalid budget test passed.")
    print("\nIntegration test passed.")


if __name__ == "__main__":
    main()