"""V1 strategy generator and ranker.

Install as:
C:/Users/VANSH/Formula1/simulator/strategy_generator_ranker.py

Ranking: FEASIBLE before REVIEW before INFEASIBLE, then lower population
standard deviation of predicted stint lengths, then shorter final prediction.
The balance score is a heuristic, not predicted race time.
"""
from statistics import pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from simulator.race_simulator import RaceSimulator
from simulator.feasibility_evaluator import evaluate_strategy

ONE_STOP_CANDIDATES: Tuple[Tuple[str, ...], ...] = (
    ("SOFT", "HARD"),
    ("MEDIUM", "HARD"),
    ("HARD", "MEDIUM"),
    ("SOFT", "MEDIUM"),
    ("MEDIUM", "SOFT"),
    ("HARD", "SOFT"),
)

TWO_STOP_CANDIDATES: Tuple[Tuple[str, ...], ...] = (
    ("SOFT", "HARD", "HARD"),
    ("MEDIUM", "HARD", "HARD"),
    ("SOFT", "SOFT", "HARD"),
    ("MEDIUM", "MEDIUM", "HARD"),
    ("SOFT", "HARD", "SOFT"),
    ("MEDIUM", "HARD", "SOFT"),
    ("SOFT", "MEDIUM", "HARD"),
    ("MEDIUM", "SOFT", "HARD"),
    ("SOFT", "MEDIUM", "SOFT"),
    ("MEDIUM", "SOFT", "SOFT"),
    ("SOFT", "SOFT", "MEDIUM"),
    ("SOFT", "MEDIUM", "MEDIUM"),
    ("HARD", "SOFT", "MEDIUM"),
    ("HARD", "MEDIUM", "SOFT"),
)

STATUS_PRIORITY = {"FEASIBLE": 0, "REVIEW": 1, "INFEASIBLE": 2}


def generate_candidates(n_stops_list: Sequence[int] = (1, 2)) -> List[List[str]]:
    """Return the agreed 6 one-stop and 14 two-stop sequences."""
    candidates = []
    for n_stops in n_stops_list:
        if n_stops == 1:
            candidates.extend(ONE_STOP_CANDIDATES)
        elif n_stops == 2:
            candidates.extend(TWO_STOP_CANDIDATES)
        else:
            raise ValueError(
                f"V1 supports one-stop and two-stop strategies only; received {n_stops} stops."
            )
    return [list(seq) for seq in dict.fromkeys(candidates)]


class StrategyGeneratorRanker:
    """Generate, simulate, evaluate, and rank V1 candidate strategies."""

    def __init__(
        self,
        simulator: Optional[RaceSimulator] = None,
        evaluator: Callable[[Dict[str, Any]], Dict[str, Any]] = evaluate_strategy,
    ):
        self.simulator = simulator if simulator is not None else RaceSimulator()
        self.evaluator = evaluator

    @staticmethod
    def _balance_score(predicted_stints: Sequence[float]) -> float:
        values = [float(value) for value in predicted_stints]
        if not values:
            raise ValueError("predicted_stints cannot be empty.")
        return round(pstdev(values), 2)

    def rank_strategies(
        self,
        race_context: Dict[str, Any],
        n_stops_list: Sequence[int] = (1, 2),
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate every candidate and return ranked results plus fallback guidance."""
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be at least 1 when provided.")

        candidates = generate_candidates(n_stops_list)
        if not candidates:
            raise ValueError("No strategy candidates were generated.")

        ranked = []
        for strategy in candidates:
            simulation = self.simulator.simulate_strategy(strategy, race_context)
            evaluation = self.evaluator(simulation)

            predicted = simulation.get("predicted_stints")
            if predicted is None:
                predicted = [s["predicted_stint_length"] for s in simulation["stints"]]

            status = evaluation.get("status", "INFEASIBLE")
            if status not in STATUS_PRIORITY:
                status = "INFEASIBLE"

            ranked.append({
                "strategy": simulation["strategy"],
                "strategy_label": simulation.get("strategy_label", " -> ".join(strategy)),
                "pit_stops": simulation.get("pit_stops", len(strategy) - 1),
                "predicted_stints": list(predicted),
                "required_stints": simulation.get(
                    "required_stints",
                    [s["required_stint_length"] for s in simulation["stints"]],
                ),
                "balance_score": self._balance_score(predicted),
                "feasibility_status": status,
                "max_absolute_gap": evaluation.get("max_absolute_gap"),
                "mean_absolute_gap": evaluation.get("mean_absolute_gap"),
                "final_stint_absolute_gap": evaluation.get("final_stint_absolute_gap"),
                "reasons": evaluation.get("reasons", []),
                "budget_valid": evaluation.get("budget_valid", False),
                "final_predicted_stint_length": round(float(predicted[-1]), 1),
                "simulation_result": simulation,
                "evaluation_result": evaluation,
            })

        # Pit-loss is reported separately. It is not folded into balance_score,
        # because the model does not predict complete race time.
        pit_loss_per_stop = race_context.get("estimated_pit_loss_seconds")
        pit_speed_limit = race_context.get("pit_lane_speed_limit")
        for item in ranked:
            item["stop_group"] = "one_stop" if item["pit_stops"] == 1 else "two_stop"
            item["pit_loss_per_stop_seconds"] = (
                round(float(pit_loss_per_stop), 2) if pit_loss_per_stop is not None else None
            )
            item["estimated_total_pit_loss_seconds"] = (
                round(float(pit_loss_per_stop) * item["pit_stops"], 2)
                if pit_loss_per_stop is not None else None
            )
            item["pit_lane_speed_limit_kmh"] = (
                float(pit_speed_limit) if pit_speed_limit is not None else None
            )

        # Rank independently within each stop-count group so the balance score
        # does not inherently favour strategies with more stints.
        groups = {}
        for group_name, stop_count in (("one_stop", 1), ("two_stop", 2)):
            group_items = [item for item in ranked if item["pit_stops"] == stop_count]
            group_items.sort(key=lambda item: (
                STATUS_PRIORITY[item["feasibility_status"]],
                item["balance_score"],
                item["final_predicted_stint_length"],
                item["strategy_label"],
            ))
            for group_rank, item in enumerate(group_items, start=1):
                item["group_rank"] = group_rank
                # Backward-compatible rank field: rank within the stop-count group.
                item["rank"] = group_rank
            feasible_group = [x for x in group_items if x["feasibility_status"] == "FEASIBLE"]
            review_group = [x for x in group_items if x["feasibility_status"] == "REVIEW"]
            if feasible_group:
                group_pool, group_fallback = feasible_group, False
                group_note = "Returning strategies that pass the current V1 heuristic checks."
            elif review_group:
                group_pool, group_fallback = review_group, True
                group_note = "No FEASIBLE strategy in this stop group; showing REVIEW candidates."
            else:
                group_pool, group_fallback = group_items, True
                group_note = "No FEASIBLE or REVIEW strategy in this stop group; fallback candidates need inspection."
            groups[group_name] = {
                "pit_stops": stop_count,
                "candidate_count": len(group_items),
                "feasible_count": sum(x["feasibility_status"] == "FEASIBLE" for x in group_items),
                "review_count": sum(x["feasibility_status"] == "REVIEW" for x in group_items),
                "infeasible_count": sum(x["feasibility_status"] == "INFEASIBLE" for x in group_items),
                "fallback_mode": group_fallback,
                "recommendation_note": group_note,
                "recommendations": group_pool[:top_n] if top_n is not None else group_pool,
                "ranked_candidates": group_items,
            }

                # Keep a global debug list, ordered by stop group then group rank.
        ranked.sort(key=lambda item: (item["pit_stops"], item["group_rank"]))

        # Assign globally sequential ranks while preserving group_rank.
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        feasible = [x for x in ranked if x["feasibility_status"] == "FEASIBLE"]
        review = [x for x in ranked if x["feasibility_status"] == "REVIEW"]
        infeasible = [x for x in ranked if x["feasibility_status"] == "INFEASIBLE"]
        if feasible:
            pool, fallback_mode = feasible, False
            note = "Global compatibility list contains FEASIBLE candidates; use strategy_groups for separate one-stop/two-stop recommendations."
        elif review:
            pool, fallback_mode = review, True
            note = "No FEASIBLE candidates; returning best REVIEW candidates."
        else:
            pool, fallback_mode = infeasible, True
            note = "None is confirmed feasible; returning best INFEASIBLE candidates."

        return {
            "gp_name": race_context.get("gp_name"),
            "candidate_count": len(candidates),
            "feasible_count": len(feasible),
            "review_count": len(review),
            "infeasible_count": len(infeasible),
            "fallback_mode": fallback_mode,
            "recommendation_note": note,
            "recommendations": pool[:top_n] if top_n is not None else pool,
            "strategy_groups": groups,
            "ranked_candidates": ranked,
            "ranking_rules": {
                "grouping": "one-stop and two-stop strategies ranked independently",
                "status_priority": ["FEASIBLE", "REVIEW", "INFEASIBLE"],
                "primary_score": "population standard deviation of predicted stint lengths",
                "primary_score_direction": "lower is better within each stop group",
                "tie_breaker": "shorter predicted final stint, then strategy label",
                "pit_loss": "reported separately; not included in balance score or a full race-time prediction",
                "score_type": "heuristic; not predicted race time",
            },
        }


if __name__ == "__main__":
    generated = generate_candidates()
    print(f"One-stop candidates: {len(ONE_STOP_CANDIDATES)}")
    print(f"Two-stop candidates: {len(TWO_STOP_CANDIDATES)}")
    print(f"Total unique candidates: {len(generated)}")
    assert len(ONE_STOP_CANDIDATES) == 6
    assert len(TWO_STOP_CANDIDATES) == 14
    assert len(generated) == 20
    assert len({tuple(x) for x in generated}) == 20
    print("Candidate generation smoke test passed.")
