from simulator.stint_predictor import StintPredictor

MODEL_PATH = (
    "C:/Users/VANSH/Formula1/models/"
    "xgb_stint_model_3seasons.pkl"
)

predictor = StintPredictor(model_path=MODEL_PATH)

prediction = predictor.predict_stint_length(
    GP="Bahrain",
    compound="MEDIUM",
    air_temp=25.0,
    track_temp=30.0,
    season=2025,
    race_progress=0.0,
    circuit_type="Permanent",
    track_length=5.412,
    track_abrasiveness="Medium",
    average_corner_speed=180.0,
    fresh_tyre=1,
    yellow_laps=0,
    sc_laps=0,
    vsc_laps=0,
    redflag_laps=0,
)

print(f"Predicted stint length: {prediction} laps")

assert isinstance(prediction, float)
assert prediction >= 1.0

print("StintPredictor smoke test passed.")