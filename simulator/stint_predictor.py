import joblib
import pandas as pd
import sys
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StintPredictor:

    def __init__(
        self,
        model_path="C:/Users/VANSH/Formula1/models/xgb_stint_model_3seasons.pkl"
    ):

        self.model = joblib.load(
            model_path
        )
        # Model loaded

    def predict_stint_length(
        self,
        GP,
        compound,
        air_temp,
        track_temp,
        season,
        race_progress,
        circuit_type,
        track_length,
        track_abrasiveness,
        average_corner_speed,
        fresh_tyre=1,
        yellow_laps=0,
        sc_laps=0,
        vsc_laps=0,
        redflag_laps=0
    ):

        features = pd.DataFrame(
            {
                "GP": [GP],
                "Compound": [compound],
                "CircuitType": [circuit_type],
                "TrackLength": [track_length],
                "TrackAbrasiveness": [track_abrasiveness],
                "AverageCornerSpeed": [average_corner_speed],
                "AirTemp": [air_temp],
                "TrackTemp": [track_temp],
                "Season": [season],
                "RaceProgress": [race_progress],
                "FreshTyre": [fresh_tyre],
                "YellowLaps": [yellow_laps],
                "SCLaps": [sc_laps],
                "VSCLaps": [vsc_laps],
                "RedFlagLaps": [redflag_laps],
            }
        )

        prediction = float(self.model.predict(features)[0])

        if not np.isfinite(prediction):
            raise ValueError(
                f"Model returned an invalid stint-length prediction: {prediction}"
    )

        return round(max(1.0, prediction), 1)