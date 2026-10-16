"""Offline tests for race_context_builder.py. No network calls are made."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from race_context_builder import RaceContextBuilder


class RaceContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.metadata_path = root / "circuit_metadata.csv"
        self.history_path = root / "historical.csv"

        pd.DataFrame([{
            "GP": "Bahrain", "CircuitType": "Permanent", "TrackLength": 5.412,
            "TrackAbrasiveness": "Medium", "AverageCornerSpeed": 180.0
        }]).to_csv(self.metadata_path, index=False)

        pd.DataFrame([
            {"GP": "Bahrain", "AirTemp": 25.0, "TrackTemp": 31.0},
            {"GP": "Bahrain", "AirTemp": 27.0, "TrackTemp": 35.0},
            {"GP": "Bahrain", "AirTemp": 26.0, "TrackTemp": 33.0},
        ]).to_csv(self.history_path, index=False)

        self.builder_args = {
            "project_root": root,
            "race_laps": {"Bahrain Grand Prix": 57},
            "circuit_metadata_path": self.metadata_path,
            "historical_data_paths": [self.history_path],
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def test_forecast_success_and_context_contract(self):
        def fake_forecast(lat, lon, target_dt):
            return {
                "air_temp": 29.5, "humidity": 55.0, "pressure": 1008.0,
                "precipitation": 0.0, "wind_speed": 12.0,
                "forecast_timestamp": "2027-03-14T15:00",
                "weather_timezone": "Asia/Bahrain",
                "weather_source": "open_meteo_forecast",
            }

        builder = RaceContextBuilder(**self.builder_args, forecast_fetcher=fake_forecast)
        context = builder.build_context(
            "Bahrain", 2027, "2027-03-14", "15:00"
        )

        required = {
            "gp_name", "air_temp", "track_temp", "season", "circuit_type",
            "track_length", "track_abrasiveness", "average_corner_speed", "race_laps"
        }
        self.assertTrue(required.issubset(context))
        self.assertEqual(context["race_laps"], 57)
        self.assertEqual(context["air_temp"], 29.5)
        self.assertEqual(context["track_temp"], 33.0)  # historical median
        self.assertEqual(context["track_temp_source"], "historical_gp_median_fallback")
        self.assertFalse(context["weather_fallback_used"])
        self.assertEqual(context["humidity"], 55.0)

    def test_historical_fallback_when_forecast_fails(self):
        def failing_forecast(lat, lon, target_dt):
            raise RuntimeError("forecast out of range")

        builder = RaceContextBuilder(**self.builder_args, forecast_fetcher=failing_forecast)
        context = builder.build_context("Bahrain Grand Prix", 2027, "2027-12-12", "15:00")

        self.assertEqual(context["weather_source"], "historical_gp_median_fallback")
        self.assertTrue(context["weather_fallback_used"])
        self.assertEqual(context["air_temp"], 26.0)
        self.assertEqual(context["track_temp"], 33.0)
        self.assertIsNone(context["humidity"])

    def test_no_fallback_raises(self):
        def failing_forecast(lat, lon, target_dt):
            raise RuntimeError("API down")

        builder = RaceContextBuilder(**self.builder_args, forecast_fetcher=failing_forecast)
        with self.assertRaises(RuntimeError):
            builder.build_context(
                "Bahrain", 2027, "2027-12-12", "15:00",
                allow_historical_weather_fallback=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
