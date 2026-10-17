import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path("C:/Users/VANSH/Formula1")
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.race_context_builder import RaceContextBuilder
from simulator.strategy_generator_ranker import StrategyGeneratorRanker


def main():
    print("=" * 65)
    print("F1 TYRE STRATEGY — FULL INTEGRATION TEST")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Build race context
    # ---------------------------------------------------------
    print("\n[1/4] Building race context...")

    builder = RaceContextBuilder()

    gp_name = input(
        "Grand Prix (use name from config/metadata): "
    ).strip()

    season = int(input("Season (e.g. 2027): ").strip())

    race_date = input(
        "Scheduled race date (YYYY-MM-DD): "
    ).strip()

    race_time = input(
        "Scheduled local start time (HH:MM; default 15:00): "
    ).strip() or "15:00"

    race_context = builder.build_context(
        gp_name=gp_name,
        season=season,
        race_date=race_date,
        race_time=race_time,
    )

    required_context_keys = [
        "gp_name",
        "air_temp",
        "track_temp",
        "season",
        "circuit_type",
        "track_length",
        "track_abrasiveness",
        "average_corner_speed",
        "race_laps",
    ]

    missing_keys = [
        key for key in required_context_keys
        if key not in race_context
    ]

    if missing_keys:
        raise AssertionError(
            f"Race context is missing required keys: {missing_keys}"
        )

    print("PASS: Race context built and required fields validated.")
    print(f"  Grand Prix: {race_context['gp_name']}")
    print(f"  Season: {race_context['season']}")
    print(f"  Race laps: {race_context['race_laps']}")
    print(f"  Circuit type: {race_context['circuit_type']}")
    print(f"  Air temperature: {race_context['air_temp']} °C")
    print(f"  Track temperature: {race_context['track_temp']} °C")
    print(f"  Weather source: {race_context['weather_source']}")
    print(
        f"  Historical fallback used: "
        f"{race_context['weather_fallback_used']}"
    )

    # ---------------------------------------------------------
    # 2. Initialize the strategy ranker
    # ---------------------------------------------------------
    print("\n[2/4] Initializing strategy generator and ranker...")

    ranker = StrategyGeneratorRanker()

    print("PASS: Ranker initialized.")

    # ---------------------------------------------------------
    # 3. Generate, simulate, evaluate, and rank strategies
    # ---------------------------------------------------------
    print("\n[3/4] Running all candidate strategies...")

    result = ranker.rank_strategies(
        race_context=race_context,
        top_n=None,
    )

    expected_candidates = 20

    if result["candidate_count"] != expected_candidates:
        raise AssertionError(
            f"Expected {expected_candidates} candidates, "
            f"received {result['candidate_count']}."
        )

    if len(result["ranked_candidates"]) != expected_candidates:
        raise AssertionError(
            "The number of ranked candidates does not match "
            "the number generated."
        )

    if result["candidate_count"] != (
        result["feasible_count"]
        + result["review_count"]
        + result["infeasible_count"]
    ):
        raise AssertionError(
            "Feasibility counts do not add up to the candidate count."
        )

    print("PASS: All 20 candidates were simulated and evaluated.")
    print(f"  FEASIBLE: {result['feasible_count']}")
    print(f"  REVIEW: {result['review_count']}")
    print(f"  INFEASIBLE: {result['infeasible_count']}")
    print(f"  Fallback mode: {result['fallback_mode']}")

    # ---------------------------------------------------------
    # 4. Validate and display recommendations
    # ---------------------------------------------------------
    print("\n[4/4] Validating ranked recommendations...")

    recommendations = result["recommendations"]

    if not recommendations:
        raise AssertionError(
            "The ranker returned no recommendations."
        )

    expected_statuses = {"FEASIBLE", "REVIEW", "INFEASIBLE"}

    for item in result["ranked_candidates"]:
        if item["feasibility_status"] not in expected_statuses:
            raise AssertionError(
                f"Unexpected feasibility status: "
                f"{item['feasibility_status']}"
            )

        if not item["budget_valid"]:
            if item["feasibility_status"] != "INFEASIBLE":
                raise AssertionError(
                    "A strategy with an invalid lap budget was not "
                    "classified as INFEASIBLE."
                )

        if len(item["predicted_stints"]) != len(item["strategy"]):
            raise AssertionError(
                f"Stint count mismatch for {item['strategy_label']}."
            )

        if len(item["required_stints"]) != len(item["strategy"]):
            raise AssertionError(
                f"Required stint count mismatch for "
                f"{item['strategy_label']}."
            )

    ranks = [
        item["rank"] for item in result["ranked_candidates"]
    ]

    if ranks != list(range(1, expected_candidates + 1)):
        raise AssertionError("Candidate ranks are missing or out of order.")

    print("PASS: Recommendation fields, statuses, and ranks validated.")

    print(f"\nGrand Prix: {result['gp_name']}")
    print(f"Candidates evaluated: {result['candidate_count']}")
    print(f"Recommendations returned: {len(recommendations)}")
    print(f"Recommendation note: {result['recommendation_note']}")

    print("\nTOP 3 RECOMMENDATIONS")
    print("-" * 65)

    for index, item in enumerate(recommendations[:3], start=1):
        print(f"\n#{index} — {item['strategy_label']}")
        print(f"  Feasibility: {item['feasibility_status']}")
        print(f"  Pit stops: {item['pit_stops']}")
        print(f"  Predicted stint lengths: {item['predicted_stints']}")
        print(f"  Required stint lengths: {item['required_stints']}")
        print(f"  Balance score: {item['balance_score']}")
        print(f"  Maximum absolute gap: {item['max_absolute_gap']}")
        print(f"  Mean absolute gap: {item['mean_absolute_gap']}")
        print(f"  Lap budget valid: {item['budget_valid']}")

    print("\n" + "=" * 65)
    print("FULL INTEGRATION TEST PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()