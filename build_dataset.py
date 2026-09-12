import pandas as pd
import fastf1
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os

def create_folders(year):

    base_path = os.path.join(
        "datasets",
        str(year)
    )

    folders = [
        "raw",
        "cleaned",
        "sessioned",
        "sorted",
        "normalized"
    ]

    for folder in folders:

        os.makedirs(
            os.path.join(base_path, folder),
            exist_ok=True
        )

    print(
        f"Created dataset structure for {year}"
    )
    return base_path


def save_df(df, filepath):

    df.to_csv(
        filepath,
        index=False
    )

    print(f"Saved: {filepath}")


# Enable F1_cache to speed up data retrieval
fastf1.Cache.enable_cache("D:/F1-Tyre-Strategy-Engine data/f1_cache")

session_order = ['FP1', 'FP2', 'FP3', 'Q', 'R']



def time_to_seconds(x):

    if pd.isna(x):
        return None

    try:
        return pd.to_timedelta(x).total_seconds()

    except:
        return None


def collect_driver_data(year):

    all_laps = []

    schedule = fastf1.get_event_schedule(
        year,
        include_testing=False
    )

    sessions = ["FP1", "FP2", "FP3", "Q", "R"]

    for rnd, row in tqdm(
        schedule.iterrows(),
        total=len(schedule),
        desc=f"Collecting {year} race data"
    ):

        event = fastf1.get_event(year, rnd)

        for s in sessions:

            try:

                session = event.get_session(s)

                session.load()

                laps = session.laps.copy()

                # metadata
                laps["RoundNumber"] = rnd
                laps["GP"] = event.EventName
                laps["Session"] = s
                laps["Date"] = event.EventDate

                all_laps.append(laps)

                print(f"Loaded {event.EventName} - {s}")

            except Exception as e:

                print(f"Skipped round {rnd} {s}: {e}")

    driver_df = pd.concat(
        all_laps,
        ignore_index=True
    )

    return driver_df

# )
# ---------------Function to clean driver data ---------------
def clean_driver_data(driver_df):

    # Make copy
    cleaned_df = driver_df.copy()

    # -------------------------
    # Convert time columns
    # -------------------------

    time_cols = [
        'LapTime',
        'Sector1Time',
        'Sector2Time',
        'Sector3Time'
    ]

    for col in time_cols:

        cleaned_df[col] = cleaned_df[col].apply(
            time_to_seconds
        )

    # -------------------------
    # Drop missing LapTime/Compound
    # -------------------------

    before_drop = cleaned_df.shape[0]

    cleaned_df = cleaned_df.dropna(
        subset=['LapTime', 'Compound']
    )

    print(
        f"Dropped {before_drop - cleaned_df.shape[0]} rows "
        f"with missing LapTime/Compound"
    )

    # -------------------------
    # Handle TyreLife
    # -------------------------

    missing_tyre = cleaned_df['TyreLife'].isna().sum()

    cleaned_df['TyreLife'] = (
        cleaned_df['TyreLife']
        .fillna(0)
        .astype(int)
    )

    print(
        f"Filled {missing_tyre} missing TyreLife values with 0"
    )

    # -------------------------
    # Handle FreshTyre
    # -------------------------

    missing_fresh = cleaned_df['FreshTyre'].isna().sum()

    cleaned_df['FreshTyre'] = (
        cleaned_df['FreshTyre']
        .fillna(False)
        .astype(int)
    )

    print(
        f"Filled {missing_fresh} missing FreshTyre values with False"
    )

    # -------------------------
    # Remove outlier laps
    # -------------------------

    median_lap = cleaned_df['LapTime'].median()

    before_outlier = cleaned_df.shape[0]

    cleaned_df = cleaned_df[
        cleaned_df['LapTime'] < 1.5 * median_lap
    ]

    print(
        f"Removed "
        f"{before_outlier - cleaned_df.shape[0]} outlier laps"
    )

    # -------------------------
    # Reset index
    # -------------------------

    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


# raw_driver_2024 = pd.read_csv(
#     "datasets/2024/raw_driver_2024.csv"
# )
# cleaned_df_2024 = clean_driver_data(raw_driver_2024)

# cleaned_df_2024.to_csv("datasets/2024/cleaned_driver_2024.csv", index=False)

# print("Cleaned driver data!")

# -----------Function to collect weather data for a given year----------------

def collect_weather_data(year):

    all_weather = []

    schedule = fastf1.get_event_schedule(
        year,
        include_testing=False
    )

    sessions = ["FP1", "FP2", "FP3", "Q", "R"]

    for rnd, row in tqdm(
        schedule.iterrows(),
        total=len(schedule),
        desc=f"Collecting {year} weather"
    ):

        event = fastf1.get_event(year, rnd)

        for s in sessions:

            try:

                session = event.get_session(s)

                session.load()

                wd = session.weather_data.copy()

                wd["RoundNumber"] = rnd
                wd["EventName"] = event.EventName
                wd["Session"] = s
                wd["Date"] = event.EventDate

                all_weather.append(wd)

            except Exception as e:

                print(f"Skipped {rnd} {s}: {e}")

    weather_df = pd.concat(
        all_weather,
        ignore_index=True
    )

    weather_df.sort_values(
        ["RoundNumber", "Session", "Time"],
        inplace=True
    )

    return weather_df

# weather_2024 = collect_weather(2024)

# weather_2024.to_csv(
#     "datasets/2024/weather_2024.csv",
#     index=False
# )


#Function to clean weather data

def clean_weather_data(weather_df):

    cleaned_weather = weather_df.copy()

    # Remove duplicates
    cleaned_weather = cleaned_weather.drop_duplicates()

    # Forward fill missing values
    cleaned_weather = cleaned_weather.ffill()

    # Numeric weather columns
    num_cols = [
        'AirTemp',
        'Humidity',
        'Pressure',
        'TrackTemp',
        'WindDirection',
        'WindSpeed'
    ]

    cleaned_weather[num_cols] = (
        cleaned_weather[num_cols]
        .astype(float)
    )

    return cleaned_weather

# weather_2024 = pd.read_csv(
#     "datasets/2024/weather_2024.csv"
# )

# cleaned_weather_2024 = clean_weather_data(
#     weather_2024
# )

# cleaned_weather_2024.to_csv(
#     "datasets/2024/cleaned_weather_2024.csv",
#     index=False
# )

def apply_session_order(df):

    df['Session'] = pd.Categorical(
        df['Session'],
        categories=session_order,
        ordered=True
    )

    return df


def sort_driver_data(driver_df):

    return driver_df.sort_values(
        ['RoundNumber', 'Session','Driver', 'LapNumber']
    ).reset_index(drop=True)

def sort_weather_data(weather_df):

    return weather_df.sort_values(
        ['RoundNumber', 'Session', 'Time']
    ).reset_index(drop=True)

def normalize_driver_data(driver_df):

    normalized_df = driver_df.copy()

    scaler = MinMaxScaler()

    cols_to_normalize = [
        'LapTime',
        'Sector1Time',
        'Sector2Time',
        'Sector3Time',
        'TyreLife'
    ]

    normalized_df[cols_to_normalize] = scaler.fit_transform(
        normalized_df[cols_to_normalize]
    )

    return normalized_df

def normalize_weather_data(weather_df):

    normalized_weather = weather_df.copy()

    scaler = MinMaxScaler()

    cols_to_normalize = [
        'AirTemp',
        'Humidity',
        'Pressure',
        'TrackTemp',
        'WindDirection',
        'WindSpeed'
    ]

    normalized_weather[cols_to_normalize] = scaler.fit_transform(
        normalized_weather[cols_to_normalize]
    )

    return normalized_weather

def main():
    driver_df = None
    weather_df = None

    year = int(
        input("Enter season: ")
    )

    base_path = create_folders(year)

    run_driver = input(
        "Collect and clean driver data? (Y/N): "
    ).upper()

    if run_driver != "Y":

        driver_df = pd.read_csv(
            f"{base_path}/cleaned/driver_cleaned.csv"
        )

        print("Loaded existing driver data.")

    else:

        # -------------------------
        # Collect Driver Data
        # -------------------------

            driver_df = collect_driver_data(year)

            save_df(
            driver_df,
            f"{base_path}/raw/driver_raw.csv"
        )

        # -------------------------
        # Clean Driver Data
        # -------------------------

            driver_df = clean_driver_data(driver_df)

            save_df(
            driver_df,
            f"{base_path}/cleaned/driver_cleaned.csv"
        )

        # -------------------------
        # Apply Session Ordering
        # -------------------------

            driver_df = apply_session_order(driver_df)

            save_df(
            driver_df,
            f"{base_path}/sessioned/driver_sessioned.csv"
        )

            print("Driver pipeline completed.")

    run_weather = input(
        "Collect and clean weather data? (Y/N): "
    ).upper()

    if run_weather != "Y":
        print("Pipeline stopped.")
        return

    # -------------------------
    # Collect Weather Data
    # -------------------------

    weather_df = collect_weather_data(year)

    save_df(
        weather_df,
        f"{base_path}/raw/weather_raw.csv"
    )

    # -------------------------
    # Clean Weather Data
    # -------------------------

    weather_df = clean_weather_data(weather_df)

    save_df(
        weather_df,
        f"{base_path}/cleaned/weather_cleaned.csv"
    )

    # -------------------------
    # Apply Session Ordering
    # -------------------------

    weather_df = apply_session_order(weather_df)

    save_df(
        weather_df,
        f"{base_path}/sessioned/weather_sessioned.csv"
    )

    print("Weather pipeline completed.")

    run_sort = input(
        "Sort datasets? (Y/N): "
    ).upper()

    if run_sort != "Y":
        print("Pipeline stopped.")
        return

    # -------------------------
    # Sort Driver Data
    # -------------------------

    driver_df = sort_driver_data(driver_df)

    save_df(
        driver_df,
        f"{base_path}/sorted/driver_sorted.csv"
    )

    # -------------------------
    # Sort Weather Data
    # -------------------------

    weather_df = sort_weather_data(weather_df)

    save_df(
        weather_df,
        f"{base_path}/sorted/weather_sorted.csv"
    )

    print("Sorting completed.")

    run_normalize = input(
        "Normalize datasets? (Y/N): "
    ).upper()

    if run_normalize != "Y":
        print("Pipeline stopped.")
        return

    # -------------------------
    # Normalize Driver Data
    # -------------------------

    normalized_driver = normalize_driver_data(
        driver_df
    )

    save_df(
        normalized_driver,
        f"{base_path}/normalized/driver_normalized.csv"
    )

    # -------------------------
    # Normalize Weather Data
    # -------------------------

    normalized_weather = normalize_weather_data(
        weather_df
    )

    save_df(
        normalized_weather,
        f"{base_path}/normalized/weather_normalized.csv"
    )

    print("Normalization completed.")
    print("Dataset pipeline completed successfully.")

if __name__ == "__main__":
    main()