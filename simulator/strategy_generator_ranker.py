"""V1 strategy generator and ranker.
""" 
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)
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

        ranked.sort(key=lambda item: (
            STATUS_PRIORITY[item["feasibility_status"]],
            item["balance_score"],
            item["final_predicted_stint_length"],
            item["strategy_label"],
        ))
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        feasible = [x for x in ranked if x["feasibility_status"] == "FEASIBLE"]
        review = [x for x in ranked if x["feasibility_status"] == "REVIEW"]
        infeasible = [x for x in ranked if x["feasibility_status"] == "INFEASIBLE"]

        if feasible:
            pool = feasible
            fallback_mode = False
            note = "Returning candidates that pass the current V1 heuristic checks."
        elif review:
            pool = review
            fallback_mode = True
            note = (
                "No candidate passed the current V1 checks. These are the best REVIEW candidates; "
                "they require inspection and are not confirmed feasible."
            )
        else:
            pool = infeasible
            fallback_mode = True
            note = (
                "No candidate passed and no REVIEW candidates were available. These are fallback "
                "suggestions; none is confirmed feasible."
            )

        return {
            "gp_name": race_context.get("gp_name"),
            "candidate_count": len(candidates),
            "feasible_count": len(feasible),
            "review_count": len(review),
            "infeasible_count": len(infeasible),
            "fallback_mode": fallback_mode,
            "recommendation_note": note,
            "recommendations": pool[:top_n] if top_n is not None else pool,
            "ranked_candidates": ranked,
            "ranking_rules": {
                "status_priority": ["FEASIBLE", "REVIEW", "INFEASIBLE"],
                "primary_score": "population standard deviation of predicted stint lengths",
                "primary_score_direction": "lower is better",
                "tie_breaker": "shorter predicted final stint, then strategy label",
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
