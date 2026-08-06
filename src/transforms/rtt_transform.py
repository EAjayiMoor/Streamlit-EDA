import pandas as pd


def filter_pah_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter RTT data to Princess Alexandra Hospital (PAH) and incomplete pathways only.

    Why:
    - Provider-level focus is PAH
    - Incomplete pathways represent the active waiting list
    - This gives the current backlog position rather than completed activity
    """
    filtered = df.copy()

    filtered = filtered[
        filtered["Provider Org Name"] == "THE PRINCESS ALEXANDRA HOSPITAL NHS TRUST"
    ]

    filtered = filtered[
        filtered["RTT Part Description"] == "Incomplete Pathways"
    ]

    return filtered


def filter_pah_admitted_backlog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter RTT data to PAH admitted backlog / DTA patients.

    Why:
    - In the RTT extracts, admitted backlog is represented by
      'Incomplete Pathways with DTA'
    - This should be treated as a subset of the wider incomplete pathway backlog
    """
    filtered = df.copy()

    filtered = filtered[
        filtered["Provider Org Name"] == "THE PRINCESS ALEXANDRA HOSPITAL NHS TRUST"
    ]

    filtered = filtered[
        filtered["RTT Part Description"] == "Incomplete Pathways with DTA"
    ]

    return filtered




def add_wait_band_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived RTT backlog metrics from weekly wait-band columns.

    Why:
    - 0–18 weeks reflects performance against the NHS constitutional standard
    - 52+ weeks reflects backlog severity / long-wait pressure
    """
    transformed = df.copy()

    transformed["0_18_total"] = (
        transformed["Gt 00 To 01 Weeks SUM 1"]
        + transformed["Gt 01 To 02 Weeks SUM 1"]
        + transformed["Gt 02 To 03 Weeks SUM 1"]
        + transformed["Gt 03 To 04 Weeks SUM 1"]
        + transformed["Gt 04 To 05 Weeks SUM 1"]
        + transformed["Gt 05 To 06 Weeks SUM 1"]
        + transformed["Gt 06 To 07 Weeks SUM 1"]
        + transformed["Gt 07 To 08 Weeks SUM 1"]
        + transformed["Gt 08 To 09 Weeks SUM 1"]
        + transformed["Gt 09 To 10 Weeks SUM 1"]
        + transformed["Gt 10 To 11 Weeks SUM 1"]
        + transformed["Gt 11 To 12 Weeks SUM 1"]
        + transformed["Gt 12 To 13 Weeks SUM 1"]
        + transformed["Gt 13 To 14 Weeks SUM 1"]
        + transformed["Gt 14 To 15 Weeks SUM 1"]
        + transformed["Gt 15 To 16 Weeks SUM 1"]
        + transformed["Gt 16 To 17 Weeks SUM 1"]
        + transformed["Gt 17 To 18 Weeks SUM 1"]
    )

    transformed["52_plus_total"] = (
        transformed["Gt 52 To 53 Weeks SUM 1"]
        + transformed["Gt 53 To 54 Weeks SUM 1"]
        + transformed["Gt 54 To 55 Weeks SUM 1"]
        + transformed["Gt 55 To 56 Weeks SUM 1"]
        + transformed["Gt 56 To 57 Weeks SUM 1"]
        + transformed["Gt 57 To 58 Weeks SUM 1"]
        + transformed["Gt 58 To 59 Weeks SUM 1"]
        + transformed["Gt 59 To 60 Weeks SUM 1"]
        + transformed["Gt 60 To 61 Weeks SUM 1"]
        + transformed["Gt 61 To 62 Weeks SUM 1"]
        + transformed["Gt 62 To 63 Weeks SUM 1"]
        + transformed["Gt 63 To 64 Weeks SUM 1"]
        + transformed["Gt 64 To 65 Weeks SUM 1"]
        + transformed["Gt 65 To 66 Weeks SUM 1"]
        + transformed["Gt 66 To 67 Weeks SUM 1"]
        + transformed["Gt 67 To 68 Weeks SUM 1"]
        + transformed["Gt 68 To 69 Weeks SUM 1"]
        + transformed["Gt 69 To 70 Weeks SUM 1"]
        + transformed["Gt 70 To 71 Weeks SUM 1"]
        + transformed["Gt 71 To 72 Weeks SUM 1"]
        + transformed["Gt 72 To 73 Weeks SUM 1"]
        + transformed["Gt 73 To 74 Weeks SUM 1"]
        + transformed["Gt 74 To 75 Weeks SUM 1"]
        + transformed["Gt 75 To 76 Weeks SUM 1"]
        + transformed["Gt 76 To 77 Weeks SUM 1"]
        + transformed["Gt 77 To 78 Weeks SUM 1"]
        + transformed["Gt 78 To 79 Weeks SUM 1"]
        + transformed["Gt 79 To 80 Weeks SUM 1"]
        + transformed["Gt 80 To 81 Weeks SUM 1"]
        + transformed["Gt 81 To 82 Weeks SUM 1"]
        + transformed["Gt 82 To 83 Weeks SUM 1"]
        + transformed["Gt 83 To 84 Weeks SUM 1"]
        + transformed["Gt 84 To 85 Weeks SUM 1"]
        + transformed["Gt 85 To 86 Weeks SUM 1"]
        + transformed["Gt 86 To 87 Weeks SUM 1"]
        + transformed["Gt 87 To 88 Weeks SUM 1"]
        + transformed["Gt 88 To 89 Weeks SUM 1"]
        + transformed["Gt 89 To 90 Weeks SUM 1"]
        + transformed["Gt 90 To 91 Weeks SUM 1"]
        + transformed["Gt 91 To 92 Weeks SUM 1"]
        + transformed["Gt 92 To 93 Weeks SUM 1"]
        + transformed["Gt 93 To 94 Weeks SUM 1"]
        + transformed["Gt 94 To 95 Weeks SUM 1"]
        + transformed["Gt 95 To 96 Weeks SUM 1"]
        + transformed["Gt 96 To 97 Weeks SUM 1"]
        + transformed["Gt 97 To 98 Weeks SUM 1"]
        + transformed["Gt 98 To 99 Weeks SUM 1"]
        + transformed["Gt 99 To 100 Weeks SUM 1"]
        + transformed["Gt 100 To 101 Weeks SUM 1"]
        + transformed["Gt 101 To 102 Weeks SUM 1"]
        + transformed["Gt 102 To 103 Weeks SUM 1"]
        + transformed["Gt 103 To 104 Weeks SUM 1"]
        + transformed["Gt 104 Weeks SUM 1"]
    )

    return transformed


def summarise_rtt_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate RTT backlog metrics to one row per month.

    Why:
    - We need one monthly hospital-level view for KPIs and trends
    - 'Total All' is the populated total field in this RTT extract
    """
    summary = (
        df.groupby("Month", as_index=False)
        .agg(
            Total=("Total All", "sum"),
            waiting_0_18_total=("0_18_total", "sum"),
            waiting_52_plus_total=("52_plus_total", "sum"),
        )
    )

    summary["pct_0_18"] = summary["waiting_0_18_total"] / summary["Total"]
    summary["pct_52_plus"] = summary["waiting_52_plus_total"] / summary["Total"]
    
    return summary

def summarise_wait_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise weekly RTT wait-band volumes by month.

    Why:
    - The monthly summary table shows total backlog, 18-week performance,
      and 52+ severity, but it does not show the shape of the waiting list.
    - This function creates a month-by-wait-band view so we can visualise
      where patient volumes sit across the full waiting-time distribution.
    """

    wait_band_columns = [
        "Gt 00 To 01 Weeks SUM 1",
        "Gt 01 To 02 Weeks SUM 1",
        "Gt 02 To 03 Weeks SUM 1",
        "Gt 03 To 04 Weeks SUM 1",
        "Gt 04 To 05 Weeks SUM 1",
        "Gt 05 To 06 Weeks SUM 1",
        "Gt 06 To 07 Weeks SUM 1",
        "Gt 07 To 08 Weeks SUM 1",
        "Gt 08 To 09 Weeks SUM 1",
        "Gt 09 To 10 Weeks SUM 1",
        "Gt 10 To 11 Weeks SUM 1",
        "Gt 11 To 12 Weeks SUM 1",
        "Gt 12 To 13 Weeks SUM 1",
        "Gt 13 To 14 Weeks SUM 1",
        "Gt 14 To 15 Weeks SUM 1",
        "Gt 15 To 16 Weeks SUM 1",
        "Gt 16 To 17 Weeks SUM 1",
        "Gt 17 To 18 Weeks SUM 1",
        "Gt 18 To 19 Weeks SUM 1",
        "Gt 19 To 20 Weeks SUM 1",
        "Gt 20 To 21 Weeks SUM 1",
        "Gt 21 To 22 Weeks SUM 1",
        "Gt 22 To 23 Weeks SUM 1",
        "Gt 23 To 24 Weeks SUM 1",
        "Gt 24 To 25 Weeks SUM 1",
        "Gt 25 To 26 Weeks SUM 1",
        "Gt 26 To 27 Weeks SUM 1",
        "Gt 27 To 28 Weeks SUM 1",
        "Gt 28 To 29 Weeks SUM 1",
        "Gt 29 To 30 Weeks SUM 1",
        "Gt 30 To 31 Weeks SUM 1",
        "Gt 31 To 32 Weeks SUM 1",
        "Gt 32 To 33 Weeks SUM 1",
        "Gt 33 To 34 Weeks SUM 1",
        "Gt 34 To 35 Weeks SUM 1",
        "Gt 35 To 36 Weeks SUM 1",
        "Gt 36 To 37 Weeks SUM 1",
        "Gt 37 To 38 Weeks SUM 1",
        "Gt 38 To 39 Weeks SUM 1",
        "Gt 39 To 40 Weeks SUM 1",
        "Gt 40 To 41 Weeks SUM 1",
        "Gt 41 To 42 Weeks SUM 1",
        "Gt 42 To 43 Weeks SUM 1",
        "Gt 43 To 44 Weeks SUM 1",
        "Gt 44 To 45 Weeks SUM 1",
        "Gt 45 To 46 Weeks SUM 1",
        "Gt 46 To 47 Weeks SUM 1",
        "Gt 47 To 48 Weeks SUM 1",
        "Gt 48 To 49 Weeks SUM 1",
        "Gt 49 To 50 Weeks SUM 1",
        "Gt 50 To 51 Weeks SUM 1",
        "Gt 51 To 52 Weeks SUM 1",
        "Gt 52 To 53 Weeks SUM 1",
        "Gt 53 To 54 Weeks SUM 1",
        "Gt 54 To 55 Weeks SUM 1",
        "Gt 55 To 56 Weeks SUM 1",
        "Gt 56 To 57 Weeks SUM 1",
        "Gt 57 To 58 Weeks SUM 1",
        "Gt 58 To 59 Weeks SUM 1",
        "Gt 59 To 60 Weeks SUM 1",
        "Gt 60 To 61 Weeks SUM 1",
        "Gt 61 To 62 Weeks SUM 1",
        "Gt 62 To 63 Weeks SUM 1",
        "Gt 63 To 64 Weeks SUM 1",
        "Gt 64 To 65 Weeks SUM 1",
        "Gt 65 To 66 Weeks SUM 1",
        "Gt 66 To 67 Weeks SUM 1",
        "Gt 67 To 68 Weeks SUM 1",
        "Gt 68 To 69 Weeks SUM 1",
        "Gt 69 To 70 Weeks SUM 1",
        "Gt 70 To 71 Weeks SUM 1",
        "Gt 71 To 72 Weeks SUM 1",
        "Gt 72 To 73 Weeks SUM 1",
        "Gt 73 To 74 Weeks SUM 1",
        "Gt 74 To 75 Weeks SUM 1",
        "Gt 75 To 76 Weeks SUM 1",
        "Gt 76 To 77 Weeks SUM 1",
        "Gt 77 To 78 Weeks SUM 1",
        "Gt 78 To 79 Weeks SUM 1",
        "Gt 79 To 80 Weeks SUM 1",
        "Gt 80 To 81 Weeks SUM 1",
        "Gt 81 To 82 Weeks SUM 1",
        "Gt 82 To 83 Weeks SUM 1",
        "Gt 83 To 84 Weeks SUM 1",
        "Gt 84 To 85 Weeks SUM 1",
        "Gt 85 To 86 Weeks SUM 1",
        "Gt 86 To 87 Weeks SUM 1",
        "Gt 87 To 88 Weeks SUM 1",
        "Gt 88 To 89 Weeks SUM 1",
        "Gt 89 To 90 Weeks SUM 1",
        "Gt 90 To 91 Weeks SUM 1",
        "Gt 91 To 92 Weeks SUM 1",
        "Gt 92 To 93 Weeks SUM 1",
        "Gt 93 To 94 Weeks SUM 1",
        "Gt 94 To 95 Weeks SUM 1",
        "Gt 95 To 96 Weeks SUM 1",
        "Gt 96 To 97 Weeks SUM 1",
        "Gt 97 To 98 Weeks SUM 1",
        "Gt 98 To 99 Weeks SUM 1",
        "Gt 99 To 100 Weeks SUM 1",
        "Gt 100 To 101 Weeks SUM 1",
        "Gt 101 To 102 Weeks SUM 1",
        "Gt 102 To 103 Weeks SUM 1",
        "Gt 103 To 104 Weeks SUM 1",
        "Gt 104 Weeks SUM 1",
    ]

def summarise_weekly_wait_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a full weekly wait-band distribution by month.

    Why:
    - 0-18 and 52+ are useful summary metrics, but they compress the distribution
    - this function preserves every weekly band so we can visualise the true shape
      and size of the waiting list across the full pathway
    - the output is reshaped into long format for easy charting in Streamlit/Plotly
    """

    wait_band_columns = [
        "Gt 00 To 01 Weeks SUM 1",
        "Gt 01 To 02 Weeks SUM 1",
        "Gt 02 To 03 Weeks SUM 1",
        "Gt 03 To 04 Weeks SUM 1",
        "Gt 04 To 05 Weeks SUM 1",
        "Gt 05 To 06 Weeks SUM 1",
        "Gt 06 To 07 Weeks SUM 1",
        "Gt 07 To 08 Weeks SUM 1",
        "Gt 08 To 09 Weeks SUM 1",
        "Gt 09 To 10 Weeks SUM 1",
        "Gt 10 To 11 Weeks SUM 1",
        "Gt 11 To 12 Weeks SUM 1",
        "Gt 12 To 13 Weeks SUM 1",
        "Gt 13 To 14 Weeks SUM 1",
        "Gt 14 To 15 Weeks SUM 1",
        "Gt 15 To 16 Weeks SUM 1",
        "Gt 16 To 17 Weeks SUM 1",
        "Gt 17 To 18 Weeks SUM 1",
        "Gt 18 To 19 Weeks SUM 1",
        "Gt 19 To 20 Weeks SUM 1",
        "Gt 20 To 21 Weeks SUM 1",
        "Gt 21 To 22 Weeks SUM 1",
        "Gt 22 To 23 Weeks SUM 1",
        "Gt 23 To 24 Weeks SUM 1",
        "Gt 24 To 25 Weeks SUM 1",
        "Gt 25 To 26 Weeks SUM 1",
        "Gt 26 To 27 Weeks SUM 1",
        "Gt 27 To 28 Weeks SUM 1",
        "Gt 28 To 29 Weeks SUM 1",
        "Gt 29 To 30 Weeks SUM 1",
        "Gt 30 To 31 Weeks SUM 1",
        "Gt 31 To 32 Weeks SUM 1",
        "Gt 32 To 33 Weeks SUM 1",
        "Gt 33 To 34 Weeks SUM 1",
        "Gt 34 To 35 Weeks SUM 1",
        "Gt 35 To 36 Weeks SUM 1",
        "Gt 36 To 37 Weeks SUM 1",
        "Gt 37 To 38 Weeks SUM 1",
        "Gt 38 To 39 Weeks SUM 1",
        "Gt 39 To 40 Weeks SUM 1",
        "Gt 40 To 41 Weeks SUM 1",
        "Gt 41 To 42 Weeks SUM 1",
        "Gt 42 To 43 Weeks SUM 1",
        "Gt 43 To 44 Weeks SUM 1",
        "Gt 44 To 45 Weeks SUM 1",
        "Gt 45 To 46 Weeks SUM 1",
        "Gt 46 To 47 Weeks SUM 1",
        "Gt 47 To 48 Weeks SUM 1",
        "Gt 48 To 49 Weeks SUM 1",
        "Gt 49 To 50 Weeks SUM 1",
        "Gt 50 To 51 Weeks SUM 1",
        "Gt 51 To 52 Weeks SUM 1",
        "Gt 52 To 53 Weeks SUM 1",
        "Gt 53 To 54 Weeks SUM 1",
        "Gt 54 To 55 Weeks SUM 1",
        "Gt 55 To 56 Weeks SUM 1",
        "Gt 56 To 57 Weeks SUM 1",
        "Gt 57 To 58 Weeks SUM 1",
        "Gt 58 To 59 Weeks SUM 1",
        "Gt 59 To 60 Weeks SUM 1",
        "Gt 60 To 61 Weeks SUM 1",
        "Gt 61 To 62 Weeks SUM 1",
        "Gt 62 To 63 Weeks SUM 1",
        "Gt 63 To 64 Weeks SUM 1",
        "Gt 64 To 65 Weeks SUM 1",
        "Gt 65 To 66 Weeks SUM 1",
        "Gt 66 To 67 Weeks SUM 1",
        "Gt 67 To 68 Weeks SUM 1",
        "Gt 68 To 69 Weeks SUM 1",
        "Gt 69 To 70 Weeks SUM 1",
        "Gt 70 To 71 Weeks SUM 1",
        "Gt 71 To 72 Weeks SUM 1",
        "Gt 72 To 73 Weeks SUM 1",
        "Gt 73 To 74 Weeks SUM 1",
        "Gt 74 To 75 Weeks SUM 1",
        "Gt 75 To 76 Weeks SUM 1",
        "Gt 76 To 77 Weeks SUM 1",
        "Gt 77 To 78 Weeks SUM 1",
        "Gt 78 To 79 Weeks SUM 1",
        "Gt 79 To 80 Weeks SUM 1",
        "Gt 80 To 81 Weeks SUM 1",
        "Gt 81 To 82 Weeks SUM 1",
        "Gt 82 To 83 Weeks SUM 1",
        "Gt 83 To 84 Weeks SUM 1",
        "Gt 84 To 85 Weeks SUM 1",
        "Gt 85 To 86 Weeks SUM 1",
        "Gt 86 To 87 Weeks SUM 1",
        "Gt 87 To 88 Weeks SUM 1",
        "Gt 88 To 89 Weeks SUM 1",
        "Gt 89 To 90 Weeks SUM 1",
        "Gt 90 To 91 Weeks SUM 1",
        "Gt 91 To 92 Weeks SUM 1",
        "Gt 92 To 93 Weeks SUM 1",
        "Gt 93 To 94 Weeks SUM 1",
        "Gt 94 To 95 Weeks SUM 1",
        "Gt 95 To 96 Weeks SUM 1",
        "Gt 96 To 97 Weeks SUM 1",
        "Gt 97 To 98 Weeks SUM 1",
        "Gt 98 To 99 Weeks SUM 1",
        "Gt 99 To 100 Weeks SUM 1",
        "Gt 100 To 101 Weeks SUM 1",
        "Gt 101 To 102 Weeks SUM 1",
        "Gt 102 To 103 Weeks SUM 1",
        "Gt 103 To 104 Weeks SUM 1",
        "Gt 104 Weeks SUM 1",
    ]

    # Sum all weekly bands by month
    grouped = df.groupby("Month", as_index=False)[wait_band_columns].sum()

    # Convert wide format to long format for charting
    long_df = grouped.melt(
        id_vars="Month",
        value_vars=wait_band_columns,
        var_name="Wait_Band",
        value_name="Volume",
    )

# Create labels that Plotly will treat as categories, not dates
# Example:
# "Gt 00 To 01 Weeks SUM 1" -> "0-1w"
# "Gt 17 To 18 Weeks SUM 1" -> "17-18w"
# "Gt 104 Weeks SUM 1" -> "104+w"
    long_df["Wait_Band_Label"] = (
    long_df["Wait_Band"]
    .str.replace("Gt ", "", regex=False)
    .str.replace(" Weeks SUM 1", "", regex=False)
    .str.replace(" To ", "-", regex=False)
    .str.replace("Weeks", "", regex=False)
    .str.strip()
    )

    long_df["Wait_Band_Label"] = long_df["Wait_Band_Label"].replace(
{"104": "104+"}
)

# Add suffix to stop Plotly parsing these as dates
    long_df["Wait_Band_Label"] = long_df["Wait_Band_Label"] + "w"

# Clean the final band label
    long_df["Wait_Band_Label"] = long_df["Wait_Band_Label"].replace(
{"104+w": "104+w"}
)
  

# Preserve band order for plotting
    band_order = {col: i for i, col in enumerate(wait_band_columns)}
    long_df["Band_Order"] = long_df["Wait_Band"].map(band_order)

    return long_df.sort_values(["Month", "Band_Order"]).reset_index(drop=True)
def summarise_rtt_by_month_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly RTT backlog summary by specialty.

    Why:
    - supports specialty backlog ranking
    - supports specialty trend views
    - forms the base for specialty heatmaps later
    """
    specialty_summary = (
        df.groupby(["Month", "Treatment Function Code", "Treatment Function Name"], as_index=False)
        .agg(
            Total=("Total All", "sum"),
            waiting_0_18_total=("0_18_total", "sum"),
            waiting_52_plus_total=("52_plus_total", "sum"),
        )
    )

    specialty_summary["waiting_18_52_total"] = (
        specialty_summary["Total"]
        - specialty_summary["waiting_0_18_total"]
        - specialty_summary["waiting_52_plus_total"]
    )

    specialty_summary["pct_0_18"] = specialty_summary["waiting_0_18_total"] / specialty_summary["Total"]
    specialty_summary["pct_52_plus"] = specialty_summary["waiting_52_plus_total"] / specialty_summary["Total"]
    specialty_summary["pct_18_52"] = specialty_summary["waiting_18_52_total"] / specialty_summary["Total"]

    return specialty_summary


def summarise_admitted_backlog_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate admitted backlog / DTA RTT patients to one row per month.
    """
    summary = (
        df.groupby("Month", as_index=False)
        .agg(
            admitted_backlog=("Total All", "sum"),
            admitted_0_18_total=("0_18_total", "sum"),
            admitted_52_plus_total=("52_plus_total", "sum"),
        )
    )

    return summary


def summarise_admitted_backlog_by_month_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate admitted backlog / DTA RTT patients by month and specialty.
    """
    specialty_summary = (
        df.groupby(
            ["Month", "Treatment Function Code", "Treatment Function Name"],
            as_index=False,
        )
        .agg(
            admitted_backlog=("Total All", "sum"),
            admitted_0_18_total=("0_18_total", "sum"),
            admitted_52_plus_total=("52_plus_total", "sum"),
        )
    )

    return specialty_summary


def get_latest_specialty_backlog(specialty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return latest month specialty backlog ranked largest to smallest.
    """
    latest_month = specialty_df["Month"].iloc[-1]
    latest_df = specialty_df[specialty_df["Month"] == latest_month].copy()
    latest_df = latest_df.sort_values("Total", ascending=False)
    return latest_df

def build_specialty_heatmap(
    specialty_df: pd.DataFrame,
    completion_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a specialty comparison table for heatmap visualisation.

    Why:
    - provides a comparative specialty view across backlog, performance,
      long-wait risk, trend, and throughput
    - uses relative ranking rather than absolute good/bad judgements
    """

    df = specialty_df.copy()

    if "Month_Date" not in df.columns:
        df["Month_Date"] = pd.to_datetime(df["Month"], format="%B %Y")

    df = df.sort_values(["Treatment Function Name", "Month_Date"])

    # Latest month per specialty
    latest = df.groupby("Treatment Function Name").tail(1).copy()

    # Previous month per specialty for trend comparison
    previous = df.groupby("Treatment Function Name").nth(-2).reset_index()

    latest = latest.merge(
        previous[["Treatment Function Name", "Total"]],
        on="Treatment Function Name",
        how="left",
        suffixes=("", "_prev"),
    )

    latest["backlog_change"] = latest["Total"] - latest["Total_prev"]

    # Prepare completions data using latest month only
    completion_df = completion_df.copy()
    if "Month_Date" not in completion_df.columns:
        completion_df["Month_Date"] = pd.to_datetime(completion_df["Month"], format="%B %Y")

    completion_df = completion_df.sort_values(["Treatment Function Name", "Month_Date"])
    latest_completions = completion_df.groupby("Treatment Function Name").tail(1).copy()

    latest = latest.merge(
        latest_completions[
            ["Treatment Function Name", "completed_total"]
        ],
        on="Treatment Function Name",
        how="left",
    )

    latest["completed_total"] = latest["completed_total"].fillna(0)
    latest["Total_prev"] = latest["Total_prev"].fillna(latest["Total"])
    latest["backlog_change"] = latest["backlog_change"].fillna(0)

    # Relative risk scores
    latest["backlog_score"] = latest["Total"].rank(pct=True)
    latest["perf_score"] = 1 - latest["pct_0_18"].rank(pct=True)
    latest["severity_score"] = latest["pct_52_plus"].rank(pct=True)
    latest["trend_score"] = latest["backlog_change"].rank(pct=True)

    # Higher completions = better throughput = lower risk, so invert
    latest["throughput_score"] = 1 - latest["completed_total"].rank(pct=True)

    # Aggregate comparative score
    latest["overall_score"] = (
        latest["backlog_score"]
        + latest["perf_score"]
        + latest["severity_score"]
        + latest["trend_score"]
        + latest["throughput_score"]
    ) / 5

    heatmap_df = latest[
        [
            "Treatment Function Name",
            "overall_score",
            "backlog_score",
            "perf_score",
            "severity_score",
            "trend_score",
            "throughput_score",
        ]
    ].copy()

    heatmap_df = heatmap_df.rename(
        columns={
            "Treatment Function Name": "Specialty",
            "overall_score": "Overall Score",
            "backlog_score": "Backlog Size",
            "perf_score": "Performance Risk",
            "severity_score": "Long Wait Risk",
            "trend_score": "Trend Risk",
            "throughput_score": "Throughput Risk",
        }
    )

    return heatmap_df.sort_values("Overall Score", ascending=False).reset_index(drop=True)

def summarise_specialty_weekly_wait_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekly wait-band distribution by specialty and month.
    """

    wait_band_columns = [col for col in df.columns if "Weeks SUM 1" in col]

    grouped = df.groupby(
        ["Month", "Treatment Function Name"], as_index=False
    )[wait_band_columns].sum()

    long_df = grouped.melt(
        id_vars=["Month", "Treatment Function Name"],
        value_vars=wait_band_columns,
        var_name="Wait_Band",
        value_name="Volume",
    )

    # Clean labels
    long_df["Wait_Band_Label"] = (
        long_df["Wait_Band"]
        .str.replace("Gt ", "", regex=False)
        .str.replace(" Weeks SUM 1", "", regex=False)
        .str.replace(" To ", "-", regex=False)
        .str.strip()
    )

    # Avoid Plotly treating as dates
    long_df["Wait_Band_Label"] = long_df["Wait_Band_Label"] + "w"

    # Preserve order
    band_order = {col: i for i, col in enumerate(wait_band_columns)}
    long_df["Band_Order"] = long_df["Wait_Band"].map(band_order)

    return long_df.sort_values(
        ["Month", "Treatment Function Name", "Band_Order"]
    ).reset_index(drop=True)
def summarise_rtt_completions_by_month_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly RTT completions summary by specialty.

    Why:
    - completed pathways for admitted and non-admitted patients act as a proxy
      for throughput
    - this lets us compare backlog size against the rate at which patients are
      coming off the list
    """

    completion_df = df.copy()

    completion_df = completion_df[
        completion_df["RTT Part Description"].isin(
            [
                "Completed Pathways For Admitted Patients",
                "Completed Pathways For Non-Admitted Patients",
            ]
        )
    ]

    grouped = (
        completion_df.groupby(
            ["Month", "Treatment Function Code", "Treatment Function Name", "RTT Part Description"],
            as_index=False,
        )
        .agg(completed_total=("Total All", "sum"))
    )

    pivoted = grouped.pivot_table(
        index=["Month", "Treatment Function Code", "Treatment Function Name"],
        columns="RTT Part Description",
        values="completed_total",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Flatten columns after pivot
    pivoted.columns.name = None

    # Ensure both completion columns exist even if one is missing in a period
    if "Completed Pathways For Admitted Patients" not in pivoted.columns:
        pivoted["Completed Pathways For Admitted Patients"] = 0

    if "Completed Pathways For Non-Admitted Patients" not in pivoted.columns:
        pivoted["Completed Pathways For Non-Admitted Patients"] = 0

    pivoted = pivoted.rename(
        columns={
            "Completed Pathways For Admitted Patients": "admitted_completed_total",
            "Completed Pathways For Non-Admitted Patients": "nonadmitted_completed_total",
        }
    )

    pivoted["completed_total"] = (
        pivoted["admitted_completed_total"] + pivoted["nonadmitted_completed_total"]
    )

    return pivoted
def summarise_rtt_additions_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly RTT demand summary.

    Why:
    - 'New RTT Periods' is the inflow into the waiting list
    - this is the clearest RTT measure of demand over time
    """

    demand_df = df.copy()

    demand_df = demand_df[
        demand_df["RTT Part Description"].str.contains(
            "New RTT Periods", case=False, na=False
        )
    ]

    summary = (
        demand_df.groupby("Month", as_index=False)
        .agg(additions_total=("Total All", "sum"))
    )

    return summary
def summarise_rtt_additions_by_month_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly RTT demand summary by specialty.
    """

    demand_df = df.copy()

    demand_df = demand_df[
        demand_df["RTT Part Description"].str.contains(
            "New RTT Periods", case=False, na=False
        )
    ]

    summary = (
        demand_df.groupby(
            ["Month", "Treatment Function Name"], as_index=False
        )
        .agg(additions_total=("Total All", "sum"))
    )

    return summary
def summarise_rtt_completions_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly RTT throughput summary.

    Why:
    - completed pathways represent patients coming off the waiting list
    - combines admitted and non-admitted pathways into one monthly throughput view
    """

    completion_df = df.copy()

    completion_df = completion_df[
        completion_df["RTT Part Description"].isin(
            [
                "Completed Pathways For Admitted Patients",
                "Completed Pathways For Non-Admitted Patients",
            ]
        )
    ]

    grouped = (
        completion_df.groupby(["Month", "RTT Part Description"], as_index=False)
        .agg(completed_total=("Total All", "sum"))
    )

    pivoted = grouped.pivot_table(
        index="Month",
        columns="RTT Part Description",
        values="completed_total",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivoted.columns.name = None

    if "Completed Pathways For Admitted Patients" not in pivoted.columns:
        pivoted["Completed Pathways For Admitted Patients"] = 0

    if "Completed Pathways For Non-Admitted Patients" not in pivoted.columns:
        pivoted["Completed Pathways For Non-Admitted Patients"] = 0

    pivoted = pivoted.rename(
        columns={
            "Completed Pathways For Admitted Patients": "admitted_completed_total",
            "Completed Pathways For Non-Admitted Patients": "nonadmitted_completed_total",
        }
    )

    pivoted["completed_total"] = (
        pivoted["admitted_completed_total"] + pivoted["nonadmitted_completed_total"]
    )

    return pivoted
def build_rtt_flow_summary(
    backlog_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    completion_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine backlog, demand, and throughput into one monthly flow summary.

    Why:
    - backlog shows current waiting list size
    - demand shows inflow (new RTT periods)
    - throughput shows patients coming off the list
    - net flow helps explain whether pressure is increasing or reducing
    """

    flow_df = backlog_df.merge(demand_df, on="Month", how="left")
    flow_df = flow_df.merge(completion_df, on="Month", how="left")

    fill_cols = [
        "additions_total",
        "admitted_completed_total",
        "nonadmitted_completed_total",
        "completed_total",
    ]

    for col in fill_cols:
        if col in flow_df.columns:
            flow_df[col] = flow_df[col].fillna(0)

    flow_df["net_flow"] = flow_df["additions_total"] - flow_df["completed_total"]

    return flow_df
def filter_pah_all_rtt_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter RTT data to Princess Alexandra Hospital / PAH provider rows,
    while keeping all RTT Part Descriptions.

    This is used for demand and throughput because those rows are not only
    'Incomplete Pathways' but still need to be PAH-only.
    """

    filtered = df.copy()

    # Prefer exact provider code match if available
    if "Provider Org Code" in filtered.columns:
        filtered = filtered[
            filtered["Provider Org Code"].astype(str).str.strip().str.upper() == "RQW"
        ].copy()

    # Fallback to provider name match if code is not available
    elif "Provider Org Name" in filtered.columns:
        filtered = filtered[
            filtered["Provider Org Name"]
            .astype(str)
            .str.upper()
            .str.contains("PRINCESS ALEXANDRA", na=False)
        ].copy()

    else:
        raise ValueError(
            "Could not find Provider Org Code or Provider Org Name for PAH filtering."
        )

    # Remove aggregate specialty rows if present
    if "Treatment Function Name" in filtered.columns:
        filtered = filtered[
            ~filtered["Treatment Function Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["total"])
        ].copy()

    if "Treatment Function Code" in filtered.columns:
        filtered = filtered[
            ~filtered["Treatment Function Code"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["C_999"])
        ].copy()

    return filtered
