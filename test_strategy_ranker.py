"""
Integration tests for the V1 Strategy Generator and Ranker.
"""

from simulator.strategy_generator_ranker import (
    StrategyGeneratorRanker,
    generate_candidates,
)


RACE_CONTEXT = {
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


def test_candidate_generation():
    candidates = generate_candidates()

    assert len(candidates) == 20, f"Expected 20 candidates, got {len(candidates)}"
    assert len({tuple(candidate) for candidate in candidates}) == 20
    assert sum(len(candidate) == 2 for candidate in candidates) == 6
    assert sum(len(candidate) == 3 for candidate in candidates) == 14

    for candidate in candidates:
        assert set(candidate).issubset({"SOFT", "MEDIUM", "HARD"})
        assert len(set(candidate)) >= 2

    print("PASS: candidate generation (20 unique strategies)")


def test_full_ranker_with_model():
    ranker = StrategyGeneratorRanker()
    result = ranker.rank_strategies(RACE_CONTEXT)

    assert result["candidate_count"] == 20
    assert (
        result["feasible_count"]
        + result["review_count"]
        + result["infeasible_count"]
        == 20
    )
    assert len(result["ranked_candidates"]) == 20
    assert result["gp_name"] == "Bahrain"

    required_fields = {
        "strategy",
        "strategy_label",
        "pit_stops",
        "predicted_stints",
        "required_stints",
        "balance_score",
        "feasibility_status",
        "max_absolute_gap",
        "mean_absolute_gap",
        "final_stint_absolute_gap",
        "reasons",
        "rank",
    }
    for candidate in result["ranked_candidates"]:
        missing = required_fields - set(candidate)
        assert not missing, f"Candidate is missing output fields: {missing}"

        assert len(candidate["strategy"]) == len(candidate["predicted_stints"])
        assert len(candidate["strategy"]) == len(candidate["required_stints"])
        assert candidate["budget_valid"] is True

    # Verify status priority and balance-score ordering within each status.
    priority = {"FEASIBLE": 0, "REVIEW": 1, "INFEASIBLE": 2}
    ranked = result["ranked_candidates"]

    for previous, current in zip(ranked, ranked[1:]):
        assert priority[previous["feasibility_status"]] <= priority[current["feasibility_status"]], (
            "Feasibility status priority is not respected."
        )

    for status in priority:
        scores = [
            candidate["balance_score"]
            for candidate in ranked
            if candidate["feasibility_status"] == status
        ]
        assert scores == sorted(scores), f"Balance scores are not sorted for {status}"

    recommendations = result["recommendations"]
    assert recommendations, "Ranker returned no recommendations."

    if result["feasible_count"] > 0:
        assert result["fallback_mode"] is False
        assert all(x["feasibility_status"] == "FEASIBLE" for x in recommendations)
    elif result["review_count"] > 0:
        assert result["fallback_mode"] is True
        assert all(x["feasibility_status"] == "REVIEW" for x in recommendations)
    else:
        assert result["fallback_mode"] is True
        assert all(x["feasibility_status"] == "INFEASIBLE" for x in recommendations)
        assert "none is confirmed feasible" in result["recommendation_note"].lower()

    print("PASS: full ranker with the actual model")
    print(f"  Grand Prix: {result['gp_name']}")
    print(f"  Candidates: {result['candidate_count']}")
    print(f"  FEASIBLE: {result['feasible_count']}")
    print(f"  REVIEW: {result['review_count']}")
    print(f"  INFEASIBLE: {result['infeasible_count']}")
    print(f"  Fallback mode: {result['fallback_mode']}")
    print(f"  Recommendations returned: {len(recommendations)}")
    print(f"  Top strategy: {recommendations[0]['strategy_label']}")
    print(f"  Top status: {recommendations[0]['feasibility_status']}")
    print(f"  Top balance score: {recommendations[0]['balance_score']}")


def test_fallback_with_controlled_evaluator():
    # This evaluator deterministically labels every candidate REVIEW.
    # It tests fallback selection without depending on model predictions.
    def always_review_evaluator(simulation):
        return {
            "status": "REVIEW",
            "budget_valid": simulation["budget_valid"],
            "max_absolute_gap": 6.0,
            "mean_absolute_gap": 6.0,
            "final_stint_absolute_gap": 6.0,
            "reasons": ["Controlled test: force REVIEW status."],
        }

    ranker = StrategyGeneratorRanker(evaluator=always_review_evaluator)
    result = ranker.rank_strategies(RACE_CONTEXT)

    assert result["candidate_count"] == 20
    assert result["feasible_count"] == 0
    assert result["review_count"] == 20
    assert result["infeasible_count"] == 0
    assert result["fallback_mode"] is True
    assert len(result["recommendations"]) == 20
    assert all(
        candidate["feasibility_status"] == "REVIEW"
        for candidate in result["recommendations"]
    )
    assert "best REVIEW candidates" in result["recommendation_note"]

    print("PASS: fallback when no candidate is FEASIBLE")


def test_infeasible_fallback_with_controlled_evaluator():
    # This evaluator deterministically labels every candidate INFEASIBLE.
    def always_infeasible_evaluator(simulation):
        return {
            "status": "INFEASIBLE",
            "budget_valid": simulation["budget_valid"],
            "max_absolute_gap": 15.0,
            "mean_absolute_gap": 15.0,
            "final_stint_absolute_gap": 15.0,
            "reasons": ["Controlled test: force INFEASIBLE status."],
        }

    ranker = StrategyGeneratorRanker(evaluator=always_infeasible_evaluator)
    result = ranker.rank_strategies(RACE_CONTEXT)

    assert result["candidate_count"] == 20
    assert result["feasible_count"] == 0
    assert result["review_count"] == 0
    assert result["infeasible_count"] == 20
    assert result["fallback_mode"] is True
    assert len(result["recommendations"]) == 20
    assert all(
        candidate["feasibility_status"] == "INFEASIBLE"
        for candidate in result["recommendations"]
    )
    assert "none is confirmed feasible" in result["recommendation_note"].lower()

    print("PASS: fallback when every candidate is INFEASIBLE")


def main():
    print("STRATEGY GENERATOR + RANKER TESTS")
    print("=" * 48)

    test_candidate_generation()
    test_full_ranker_with_model()
    test_fallback_with_controlled_evaluator()
    test_infeasible_fallback_with_controlled_evaluator()

    print("\nAll Strategy Generator and Ranker tests passed.")


if __name__ == "__main__":
    main()
