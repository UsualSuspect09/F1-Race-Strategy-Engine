from simulator.stint_predictor import StintPredictor


class RaceSimulator:
    """Deterministic race-distance simulator for V1."""

    REQUIRED_CONTEXT_KEYS = (
        "gp_name",
        "air_temp",
        "track_temp",
        "season",
        "circuit_type",
        "track_length",
        "track_abrasiveness",
        "average_corner_speed",
        "race_laps",
    )

    ALLOWED_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

    def __init__(self, model_path=None):
        self.predictor = (
            StintPredictor(model_path=model_path)
            if model_path
            else StintPredictor()
        )

    def simulate_strategy(self, strategy, race_context):
        """Predict every stint and separately allocate the race-distance budget."""
        if not strategy:
            raise ValueError("Strategy cannot be empty.")

        strategy = [str(compound).upper() for compound in strategy]
        invalid = [c for c in strategy if c not in self.ALLOWED_COMPOUNDS]
        if invalid:
            raise ValueError(
                f"Unsupported compound(s): {invalid}. "
                "V1 supports SOFT, MEDIUM, and HARD."
            )

        missing = [key for key in self.REQUIRED_CONTEXT_KEYS if key not in race_context]
        if missing:
            raise KeyError(f"race_context is missing required keys: {missing}")

        raw_race_laps = race_context["race_laps"]
        if isinstance(raw_race_laps, bool) or int(raw_race_laps) != raw_race_laps:
            raise ValueError("race_laps must be a whole number.")
        race_laps = int(raw_race_laps)
        if race_laps < len(strategy):
            raise ValueError("Race distance must provide at least one lap per stint.")

        stints = []
        laps_allocated = 0
        n_stints = len(strategy)

        for index, compound in enumerate(strategy):
            stint_number = index + 1
            remaining_laps = race_laps - laps_allocated
            is_final = index == n_stints - 1
            race_progress = laps_allocated / race_laps

            # Always preserve the model's prediction, including for the final stint.
            predicted = float(self.predictor.predict_stint_length(
                GP=race_context["gp_name"],
                compound=compound,
                air_temp=race_context["air_temp"],
                track_temp=race_context["track_temp"],
                season=race_context["season"],
                race_progress=race_progress,
                circuit_type=race_context["circuit_type"],
                track_length=race_context["track_length"],
                track_abrasiveness=race_context["track_abrasiveness"],
                average_corner_speed=race_context["average_corner_speed"],
                fresh_tyre=1,
                yellow_laps=race_context.get("yellow_laps", 0),
                sc_laps=race_context.get("sc_laps", 0),
                vsc_laps=race_context.get("vsc_laps", 0),
                redflag_laps=race_context.get("redflag_laps", 0),
            ))

            if is_final:
                # Allocation closes the race-distance budget; prediction stays separate.
                required_laps = remaining_laps
            else:
                # Retain the existing V1 five-lap minimum and reserve at least
                # one lap for every stint that follows.
                predicted_laps = max(5, int(round(predicted)))
                later_stints = n_stints - stint_number
                max_allocation = remaining_laps - later_stints
                required_laps = min(predicted_laps, max_allocation)

            required_laps = int(required_laps)
            laps_allocated += required_laps

            stints.append({
                "stint_number": stint_number,
                "compound": compound,
                "race_progress": round(race_progress, 3),
                "predicted_stint_length": round(predicted, 1),
                "required_stint_length": required_laps,
                "prediction_gap": round(required_laps - predicted, 1),
                "is_final_stint": is_final,
            })

        total_allocated = sum(s["required_stint_length"] for s in stints)
        budget_valid = total_allocated == race_laps

        return {
            "gp_name": race_context["gp_name"],
            "strategy": strategy,
            "strategy_label": " -> ".join(strategy),
            "stints": stints,
            "predicted_stints": [s["predicted_stint_length"] for s in stints],
            "required_stints": [s["required_stint_length"] for s in stints],
            "race_laps": race_laps,
            "total_allocated_laps": total_allocated,
            "coverage_margin": total_allocated - race_laps,
            "pit_stops": len(strategy) - 1,
            "budget_valid": budget_valid,
            "final_compound": strategy[-1],
            "circuit_type": race_context["circuit_type"],
        }
