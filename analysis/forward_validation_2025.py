import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("C:/Users/VANSH/Formula1")

DATASET_2023 = PROJECT_ROOT / "datasets/2023/master_stint_2023_v4_targetfix.csv"
DATASET_2024 = PROJECT_ROOT / "datasets/2024/master_stint_2024_v4_targetfix.csv"
DATASET_2025 = PROJECT_ROOT / "datasets/2025/master_stint_2025_v2_targetfix.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis/outputs"

OUTPUT_FILE = OUTPUT_DIR / "2025_forward_validation_residuals.csv"


# ============================================================
# 2. LOAD DATASETS
# ============================================================

stints_2023 = pd.read_csv(DATASET_2023)
stints_2024 = pd.read_csv(DATASET_2024)
stints_2025 = pd.read_csv(DATASET_2025)

# Explicit season labels
stints_2023["Season"] = 2023
stints_2024["Season"] = 2024
stints_2025["Season"] = 2025

# Combine training seasons
train_df = pd.concat(
    [stints_2023, stints_2024],
    ignore_index=True
)

# Keep 2025 completely separate
test_df = stints_2025.copy()

print("========== DATASET SUMMARY ==========")

print("2023 samples:", len(stints_2023))
print("2024 samples:", len(stints_2024))
print("Training samples:", len(train_df))
print("2025 testing samples:", len(test_df))


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

categorical_features = [
    "Compound",
    "GP",
    "CircuitType",
    "TrackAbrasiveness"
]

numerical_features = [
    "AirTemp",
    "TrackTemp",
    "TrackLength",
    "Season",
    "RaceProgress",
    "FreshTyre",
    "YellowLaps",
    "SCLaps",
    "VSCLaps",
    "RedFlagLaps",
    "AverageCornerSpeed"
]

feature_columns = (
    categorical_features + numerical_features
)

target_column = "StintLength"


# ============================================================
# 4. VALIDATE DATA
# ============================================================

required_columns = feature_columns + [target_column]

for name, df in [
    ("2023", stints_2023),
    ("2024", stints_2024),
    ("2025", stints_2025)
]:
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

    if df[target_column].isna().any():
        raise ValueError(
            f"{name} dataset contains missing target values."
        )


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

X_train = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()

# Same categorical missing-value treatment
X_train[categorical_features] = (
    X_train[categorical_features].fillna("Unknown")
)

X_test[categorical_features] = (
    X_test[categorical_features].fillna("Unknown")
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# 8. TRAIN ON 2023-2024
# ============================================================

print("\n========== MODEL TRAINING ==========")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 9. PREDICT 2025
# ============================================================

train_predictions = pipeline.predict(X_train)

test_predictions = pipeline.predict(X_test)


# ============================================================
# 10. EVALUATION
# ============================================================

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

print("\n========== FORWARD VALIDATION RESULTS ==========")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print(f"Training MAE: {train_mae:.4f} laps")
print(f"2025 MAE:     {test_mae:.4f} laps")


# ============================================================
# 11. GENERATE PREDICTION RESIDUALS
# ============================================================

residuals = test_df[
    ["GP", "Driver", "Stint", "Compound", "Season"]
].copy()

residuals["ActualStintLength"] = y_test.values

residuals["PredictedStintLength"] = test_predictions

# Positive residual = model underestimated stint length
residuals["Residual"] = (
    residuals["ActualStintLength"]
    - residuals["PredictedStintLength"]
)

residuals["AbsoluteError"] = (
    residuals["Residual"].abs()
)


# ============================================================
# 12. COMPOUND-WISE ERROR ANALYSIS
# ============================================================

print("\n========== COMPOUND-WISE MAE ==========")

compound_mae = (
    residuals
    .groupby("Compound")
    .apply(
        lambda group: mean_absolute_error(
            group["ActualStintLength"],
            group["PredictedStintLength"]
        ),
        include_groups=False
    )
)

print(compound_mae)


# ============================================================
# 13. RESIDUAL DISTRIBUTION
# ============================================================

print("\n========== RESIDUAL DISTRIBUTION ==========")

print(residuals["Residual"].describe())

print("\nResidual standard deviation:")
print(residuals["Residual"].std())


# ============================================================
# 14. EXPORT RESIDUAL DATASET
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

residuals.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== EXPORT COMPLETE ==========")

print("Residual dataset saved to:")
print(OUTPUT_FILE)

print("\nForward validation completed successfully.")