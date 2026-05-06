from pathlib import Path

import pandas as pd


DATASET_FOLDER = Path("Datasets")
OUTPUT_FILE = Path("Combined_Datasets_15min_avg.csv")


all_files = sorted(DATASET_FOLDER.glob("*.csv"))

if not all_files:
    raise FileNotFoundError(f"No CSV files found in {DATASET_FOLDER.resolve()}")

dataframes = []

for filename in all_files:
    df = pd.read_csv(filename, sep=";")
    df["Patient_ID"] = filename.stem
    dataframes.append(df)

combined_activity = pd.concat(dataframes, axis=0, ignore_index=True)
combined_activity["time"] = pd.to_datetime(combined_activity["time"], errors="coerce")
combined_activity = combined_activity.dropna(subset=["time"])

aggregation_rules = {
    "glucose": "mean",
    "calories": "sum",
    "heart_rate": "mean",
    "steps": "sum",
    "basal_rate": "mean",
    "bolus_volume_delivered": "sum",
    "carb_input": "sum",
}

aggregation_rules = {
    column: rule
    for column, rule in aggregation_rules.items()
    if column in combined_activity.columns
}

combined_activity_5min = (
    combined_activity.sort_values(["Patient_ID", "time"])
    .set_index("time")
    .groupby("Patient_ID")
    .resample("15min")
    .agg(aggregation_rules)
    .reset_index()
)

combined_activity_5min = combined_activity_5min.dropna(
    subset=list(aggregation_rules),
    how="all",
)
combined_activity_5min = combined_activity_5min[
    ["Patient_ID", "time", *aggregation_rules]
]

combined_activity_5min.to_csv(OUTPUT_FILE, index=False)

print(f"Input files: {len(all_files)}")
print(f"Total rows and columns: {combined_activity_5min.shape}")
print(f"Saved file: {OUTPUT_FILE}")
