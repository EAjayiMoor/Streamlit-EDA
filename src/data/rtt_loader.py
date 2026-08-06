import os
import pandas as pd
from src.utils.data_cleaning import remove_aggregate_rows


def load_all_rtt_files(folder_path="data/raw/rtt"):
    """
    Load all RTT CSV files from a folder and combine them into one dataframe.

    Why:
    - RTT data is published monthly in separate files
    - We want one scalable ingestion function for 3 months or 24 months
    - This replaces manual duplication and appending
    """

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    dataframes = []

    for file_name in sorted(all_files):
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)

        # Keep source file for traceability/debugging
        df["source_file"] = file_name

        # Extract readable month label from the NHS RTT filename
        # Example:
        # 20260131-RTT-January-2026-full-extract.csv -> "January 2026"
        parts = file_name.replace(".csv", "").split("-")
        if len(parts) >= 4:
            month_label = f"{parts[2]} {parts[3]}"
        else:
            month_label = file_name

        df["Month"] = month_label

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    # Apply global cleaning once so all downstream pages use the same clean RTT base
    combined_df = remove_aggregate_rows(combined_df)

    return combined_df