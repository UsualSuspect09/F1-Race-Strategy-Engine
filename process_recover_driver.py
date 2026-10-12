import os
import shutil
import pandas as pd

from build_dataset import (
    clean_driver_data,
    apply_session_order,
    sort_driver_data,
    normalize_driver_data
)


# -----------------------------------
# File Paths
# -----------------------------------

BASE_PATH = "C:/Users/VANSH/Formula1/datasets/2025"

RECOVERED_RAW = (
    f"{BASE_PATH}/raw/driver_raw_recovered.csv"
)

RAW_PATH = (
    f"{BASE_PATH}/raw/driver_raw.csv"
)

BACKUP_PATH = (
    f"{BASE_PATH}/raw/driver_raw_before_recovery.csv"
)

WEATHER_PATH = (
    f"{BASE_PATH}/sorted/weather_sorted.csv"
)


# -----------------------------------
# Verify Input Files
# -----------------------------------

if not os.path.exists(RECOVERED_RAW):
    raise FileNotFoundError(
        f"Recovered dataset not found: {RECOVERED_RAW}"
    )

if not os.path.exists(WEATHER_PATH):
    raise FileNotFoundError(
        f"Weather dataset not found: {WEATHER_PATH}"
    )


# -----------------------------------
# Validate GP/Session Coverage
# -----------------------------------

driver_df = pd.read_csv(
    RECOVERED_RAW,
    low_memory=False
)

weather_df = pd.read_csv(
    WEATHER_PATH,
    usecols=["EventName", "Session"],
    low_memory=False
)

expected_pairs = set(
    weather_df[
        ["EventName", "Session"]
    ]
    .dropna()
    .astype(str)
    .apply(lambda col: col.str.strip())
    .itertuples(index=False, name=None)
)

existing_pairs = set(
    driver_df[
        ["GP", "Session"]
    ]
    .dropna()
    .astype(str)
    .apply(lambda col: col.str.strip())
    .itertuples(index=False, name=None)
)

missing_pairs = sorted(
    expected_pairs - existing_pairs
)

if missing_pairs:

    print("Missing GP/session combinations:")

    for gp, session in missing_pairs:
        print(f"  {gp} — {session}")

    raise SystemExit(
        "Recovery incomplete. Fix missing sessions before processing."
    )

print("GP/session coverage verified.")


# -----------------------------------
# Preserve Original Raw Dataset
# -----------------------------------

if os.path.exists(RAW_PATH) and not os.path.exists(BACKUP_PATH):

    shutil.copy2(
        RAW_PATH,
        BACKUP_PATH
    )

    print("Original raw dataset backed up.")

# Promote the recovered raw dataset to the standard path
shutil.copy2(
    RECOVERED_RAW,
    RAW_PATH
)

print("Recovered raw dataset installed.")


# -----------------------------------
# Clean Driver Data
# -----------------------------------

cleaned_df = clean_driver_data(
    driver_df
)

cleaned_path = (
    f"{BASE_PATH}/cleaned/driver_cleaned.csv"
)

cleaned_df.to_csv(
    cleaned_path,
    index=False
)

print("Cleaned data saved:", cleaned_path)


# -----------------------------------
# Apply Session Ordering
# -----------------------------------

sessioned_df = apply_session_order(
    cleaned_df
)

sessioned_path = (
    f"{BASE_PATH}/sessioned/driver_sessioned.csv"
)

sessioned_df.to_csv(
    sessioned_path,
    index=False
)

print("Sessioned data saved:", sessioned_path)


# -----------------------------------
# Sort Driver Data
# -----------------------------------

sorted_df = sort_driver_data(
    sessioned_df
)

sorted_path = (
    f"{BASE_PATH}/sorted/driver_sorted.csv"
)

sorted_df.to_csv(
    sorted_path,
    index=False
)

print("Sorted data saved:", sorted_path)


# -----------------------------------
# Normalize Driver Data
# -----------------------------------

normalized_df = normalize_driver_data(
    sorted_df
)

normalized_path = (
    f"{BASE_PATH}/normalized/driver_normalized.csv"
)

normalized_df.to_csv(
    normalized_path,
    index=False
)

print("Normalized data saved:", normalized_path)


# -----------------------------------
# Final Verification
# -----------------------------------

print("\n========== PROCESSING SUMMARY ==========")

print("Raw rows:", len(driver_df))
print("Cleaned rows:", len(cleaned_df))
print("Sorted rows:", len(sorted_df))

print(
    "Unique GPs:",
    sorted_df["GP"].nunique()
)

print(
    "Unique rounds:",
    sorted_df["RoundNumber"].nunique()
)

print(
    "Sessions:",
    sorted_df["Session"].astype(str).value_counts().to_dict()
)

print("\nDriver processing completed.")