from pathlib import Path

import pandas as pd
import streamlit as st


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
SESSION_STANDARD_MINUTES = 240
TIMESTAMP_COLUMNS = [
    "Scheduled start time(Session)",
    "Scheduled finish time(Session)",
    "Actual start time(Session)",
    "Actual finish time(Session)",
    "Operation Anaesthetic Start Datetime",
    "Operation Start Datetime",
    "Operation Procedure Start Datetime",
    "Operation Procedure Finish Datetime",
    "Operation Finish Datetime",
    "Operation Patient in Recovery Datetime",
    "Cancellation Date",
]
SESSION_TYPE_COLUMN = "Elective/Emergency"


@st.cache_data(show_spinner=False)
def load_theatre_activity_data(
    path: str = "data/raw/theatre utilisation",
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Theatre activity path not found: {path}")

    if path.is_file():
        return _clean_theatre_activity(_load_single_file(path))

    files = [
        file
        for file in path.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"No theatre activity files found in: {path}")

    dfs = []

    for file in sorted(files):
        df = _clean_theatre_activity(_load_single_file(file))
        df["Source_File"] = file.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def _load_single_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        for encoding in ["utf-8-sig", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(
                "utf-8-sig",
                b"",
                0,
                1,
                f"Could not decode theatre activity file: {file_path}",
            )
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported theatre activity file type: {file_path}")

    df.columns = df.columns.str.strip()
    return df


def _clean_theatre_activity(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Theatre session ID",
        "Booked Operation Date",
        "Scheduled start time(Session)",
        "Scheduled finish time(Session)",
        "Case Touch time (minutes)",
        "Number of cases completed",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required theatre activity columns: {missing}")

    cleaned = df.copy()

    for col in ["Booked Operation Date", *TIMESTAMP_COLUMNS]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(
                cleaned[col].replace("NULL", pd.NA),
                errors="coerce",
                dayfirst=True,
            )

    for col in [
        "Case Touch time (minutes)",
        "Number of cases completed",
        "Cancelled cases",
    ]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)

    cleaned["Theatre session ID"] = (
        cleaned["Theatre session ID"].astype(str).str.strip()
    )

    if "Specialty (standardised)" in cleaned.columns:
        cleaned["Specialty (standardised)"] = (
            cleaned["Specialty (standardised)"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    if SESSION_TYPE_COLUMN not in cleaned.columns:
        cleaned[SESSION_TYPE_COLUMN] = "Unknown"

    cleaned[SESSION_TYPE_COLUMN] = (
        cleaned[SESSION_TYPE_COLUMN]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
    )

    start_cols = [
        col
        for col in [
            "Operation Anaesthetic Start Datetime",
            "Operation Start Datetime",
            "Operation Procedure Start Datetime",
            "Actual start time(Session)",
        ]
        if col in cleaned.columns
    ]
    end_cols = [
        col
        for col in [
            "Operation Patient in Recovery Datetime",
            "Operation Finish Datetime",
            "Operation Procedure Finish Datetime",
            "Actual finish time(Session)",
        ]
        if col in cleaned.columns
    ]

    if start_cols and end_cols:
        start = cleaned[start_cols].bfill(axis=1).iloc[:, 0]
        recovery_col = "Operation Patient in Recovery Datetime"
        if recovery_col in cleaned.columns:
            fallback_end_cols = [col for col in end_cols if col != recovery_col]
            fallback_end = (
                cleaned[fallback_end_cols].max(axis=1)
                if fallback_end_cols
                else pd.Series(pd.NaT, index=cleaned.index)
            )
            end = cleaned[recovery_col].combine_first(fallback_end)
        else:
            end = cleaned[end_cols].max(axis=1)

        derived_touch = (end - start).dt.total_seconds() / 60
        derived_touch = derived_touch.mask(derived_touch < 0)
        cleaned["Model_Hospital_Start_Datetime"] = start
        cleaned["Model_Hospital_End_Datetime"] = end
        cleaned["Derived_Touch_Time_Minutes"] = derived_touch
        cleaned["Model_Hospital_Touch_Minutes"] = derived_touch.combine_first(
            cleaned["Case Touch time (minutes)"]
        )
    else:
        cleaned["Model_Hospital_Start_Datetime"] = pd.NaT
        cleaned["Model_Hospital_End_Datetime"] = pd.NaT
        cleaned["Derived_Touch_Time_Minutes"] = pd.NA
        cleaned["Model_Hospital_Touch_Minutes"] = cleaned[
            "Case Touch time (minutes)"
        ]

    return cleaned


def summarise_vanguard_capacity_impact(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = "2025-04-01",
    end_date: str | pd.Timestamp = "2026-03-31",
) -> pd.Series:
    output = {
        "Rows": 0,
        "Completed_Cases": 0.0,
        "Sessions": 0.0,
        "Cancelled_Cases": 0.0,
        "Primary_Specialty": "Not available",
        "Primary_Specialty_Cases": 0.0,
        "Start_Date": pd.NaT,
        "End_Date": pd.NaT,
    }

    required = [
        "Booked Operation Date",
        "Site / Theatre location",
        SESSION_TYPE_COLUMN,
        "Number of cases completed",
        "Theatre session ID",
    ]

    if df.empty or any(col not in df.columns for col in required):
        return pd.Series(output)

    work = df.copy()
    work = work[
        (work["Booked Operation Date"] >= pd.to_datetime(start_date))
        & (work["Booked Operation Date"] <= pd.to_datetime(end_date))
        & work["Site / Theatre location"]
        .fillna("")
        .astype(str)
        .str.contains("vanguard", case=False, na=False)
    ].copy()

    output["Rows"] = len(work)

    if work.empty:
        return pd.Series(output)

    completed = work[
        work[SESSION_TYPE_COLUMN]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("elective")
        & (pd.to_numeric(work["Number of cases completed"], errors="coerce") > 0)
    ].copy()

    if completed.empty:
        return pd.Series(output)

    completed_cases = pd.to_numeric(
        completed["Number of cases completed"],
        errors="coerce",
    ).fillna(0)

    output["Completed_Cases"] = float(completed_cases.sum())
    output["Sessions"] = float(completed["Theatre session ID"].nunique())
    output["Start_Date"] = completed["Booked Operation Date"].min()
    output["End_Date"] = completed["Booked Operation Date"].max()

    if "Cancelled cases" in work.columns:
        output["Cancelled_Cases"] = float(
            pd.to_numeric(work["Cancelled cases"], errors="coerce").fillna(0).sum()
        )

    if "Specialty (standardised)" in completed.columns:
        specialty_cases = (
            completed.assign(Completed_Cases=completed_cases)
            .groupby("Specialty (standardised)")["Completed_Cases"]
            .sum()
            .sort_values(ascending=False)
        )

        if not specialty_cases.empty:
            output["Primary_Specialty"] = str(specialty_cases.index[0])
            output["Primary_Specialty_Cases"] = float(specialty_cases.iloc[0])

    return pd.Series(output)


def _normalise_session_type(value: str) -> str:
    text = str(value).strip().lower()
    if "elective" in text:
        return "Elective"
    if "emergency" in text or "trauma" in text:
        return "Emergency"
    return "Unknown"


def _classify_session_type(values: pd.Series) -> str:
    types = {
        _normalise_session_type(value)
        for value in values.dropna().astype(str)
        if str(value).strip()
    }
    types.discard("Unknown")

    if types == {"Elective"}:
        return "Elective"
    if types == {"Emergency"}:
        return "Emergency"
    if "Elective" in types and "Emergency" in types:
        return "Mixed elective/emergency"
    return "Unknown"


def _session_type_label(work: pd.DataFrame) -> pd.Series:
    if SESSION_TYPE_COLUMN not in work.columns:
        return pd.Series("Unknown", index=work.index)

    return work.groupby(["Booked Operation Date", "Theatre session ID"])[
        SESSION_TYPE_COLUMN
    ].transform(_classify_session_type)


def _session_has_obstetrics(work: pd.DataFrame) -> pd.Series:
    if "Specialty (standardised)" not in work.columns:
        return pd.Series(False, index=work.index)

    obstetrics = work["Specialty (standardised)"].fillna("").astype(str).str.contains(
        "obstetric|maternity",
        case=False,
        regex=True,
    )
    return obstetrics.groupby(
        [work["Booked Operation Date"], work["Theatre session ID"]]
    ).transform("max")


def summarise_theatre_capacity(
    df: pd.DataFrame,
    recent_months: int = 12,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    session_type_scope: str = "all",
    exclude_obstetrics: bool = False,
    actual_sessions_only_for_utilisation: bool = False,
    touch_time_column: str = "Model_Hospital_Touch_Minutes",
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    work = df.copy()
    work = work.dropna(
        subset=[
            "Theatre session ID",
            "Booked Operation Date",
            "Scheduled start time(Session)",
            "Scheduled finish time(Session)",
        ]
    )

    if work.empty:
        return pd.Series(dtype="float64")

    latest_date = work["Booked Operation Date"].max()

    if start_date is not None or end_date is not None:
        period_start = (
            pd.to_datetime(start_date)
            if start_date is not None
            else work["Booked Operation Date"].min()
        )
        period_end = (
            pd.to_datetime(end_date)
            if end_date is not None
            else work["Booked Operation Date"].max()
        )
        recent = work[
            (work["Booked Operation Date"] >= period_start)
            & (work["Booked Operation Date"] <= period_end)
        ].copy()
    else:
        period_start = latest_date - pd.DateOffset(months=recent_months)
        recent = work[work["Booked Operation Date"] >= period_start].copy()

    if recent.empty:
        recent = work.copy()

    recent["Session_Type"] = _session_type_label(recent)
    recent["Session_Has_Obstetrics"] = _session_has_obstetrics(recent)

    scope = session_type_scope.strip().lower()
    if scope == "elective":
        recent = recent[recent["Session_Type"] == "Elective"].copy()
    elif scope == "emergency":
        recent = recent[recent["Session_Type"] == "Emergency"].copy()
    elif scope in {"mixed", "mixed elective/emergency"}:
        recent = recent[recent["Session_Type"] == "Mixed elective/emergency"].copy()

    if exclude_obstetrics:
        recent = recent[~recent["Session_Has_Obstetrics"]].copy()

    if recent.empty:
        return pd.Series(dtype="float64")

    recent["Scheduled_Minutes"] = (
        recent["Scheduled finish time(Session)"]
        - recent["Scheduled start time(Session)"]
    ).dt.total_seconds() / 60

    recent.loc[recent["Scheduled_Minutes"] < 0, "Scheduled_Minutes"] += 24 * 60
    invalid_scheduled_sessions = recent.loc[
        ~recent["Scheduled_Minutes"].between(30, 720),
        ["Booked Operation Date", "Theatre session ID"],
    ].drop_duplicates()
    recent = recent[recent["Scheduled_Minutes"].between(30, 720)].copy()

    if recent.empty:
        return pd.Series(dtype="float64")

    if touch_time_column not in recent.columns:
        touch_time_column = "Case Touch time (minutes)"

    recent[touch_time_column] = pd.to_numeric(
        recent[touch_time_column],
        errors="coerce",
    ).fillna(0)

    valid_touch_mask = recent[touch_time_column].between(0, 720)
    recent["Valid_Touch_Minutes"] = recent[touch_time_column].where(
        valid_touch_mask,
        0,
    )
    recent["Cases_With_Valid_Touch"] = recent["Number of cases completed"].where(
        valid_touch_mask,
        0,
    )

    if "Cancelled cases" not in recent.columns:
        recent["Cancelled cases"] = 0

    for col in ["Actual start time(Session)", "Actual finish time(Session)"]:
        if col not in recent.columns:
            recent[col] = pd.NaT

    session_summary = (
        recent.groupby(["Booked Operation Date", "Theatre session ID"], as_index=False)
        .agg(
            Session_Date=("Booked Operation Date", "min"),
            Session_Type=("Session_Type", "first"),
            Session_Has_Obstetrics=("Session_Has_Obstetrics", "max"),
            Scheduled_Minutes=("Scheduled_Minutes", "first"),
            Touch_Minutes=("Valid_Touch_Minutes", "sum"),
            Completed_Cases=("Number of cases completed", "sum"),
            Cases_With_Valid_Touch=("Cases_With_Valid_Touch", "sum"),
            Cancelled_Cases=("Cancelled cases", "sum"),
            Actual_Start=("Actual start time(Session)", "min"),
            Actual_Finish=("Actual finish time(Session)", "max"),
        )
        .dropna(subset=["Session_Date"])
    )

    if session_summary.empty:
        return pd.Series(dtype="float64")

    observed_days = (
        session_summary["Session_Date"].max()
        - session_summary["Session_Date"].min()
    ).days + 1

    observed_weeks = max(observed_days / 7, 1)
    completed_cases = session_summary["Completed_Cases"].sum()
    touch_minutes = session_summary["Touch_Minutes"].sum()
    cases_with_valid_touch = session_summary["Cases_With_Valid_Touch"].sum()
    scheduled_minutes = session_summary["Scheduled_Minutes"].sum()
    invalid_touch_rows = int((~valid_touch_mask).sum())
    planned_sessions = len(session_summary)
    session_summary["Actual_Session_Flag"] = (
        (session_summary["Touch_Minutes"] > 0)
        | (session_summary["Completed_Cases"] > 0)
        | session_summary["Actual_Start"].notna()
        | session_summary["Actual_Finish"].notna()
    )
    actual_session_summary = session_summary[
        session_summary["Actual_Session_Flag"]
    ].copy()
    actual_sessions = len(actual_session_summary)
    cancelled_or_not_run_sessions = max(planned_sessions - actual_sessions, 0)
    actual_scheduled_minutes = actual_session_summary["Scheduled_Minutes"].sum()
    cancelled_cases = session_summary["Cancelled_Cases"].sum()

    utilisation_session_summary = (
        actual_session_summary
        if actual_sessions_only_for_utilisation
        else session_summary
    )
    utilisation_sessions = len(utilisation_session_summary)
    utilisation_scheduled_minutes = utilisation_session_summary[
        "Scheduled_Minutes"
    ].sum()
    completed_cases = utilisation_session_summary["Completed_Cases"].sum()
    touch_minutes = utilisation_session_summary["Touch_Minutes"].sum()
    cases_with_valid_touch = utilisation_session_summary[
        "Cases_With_Valid_Touch"
    ].sum()

    avg_case_duration = (
        touch_minutes / cases_with_valid_touch
        if cases_with_valid_touch > 0
        else 45.0
    )

    utilisation = (
        touch_minutes / utilisation_scheduled_minutes
        if utilisation_scheduled_minutes > 0
        else 0.0
    )
    actual_session_utilisation = (
        touch_minutes / actual_scheduled_minutes
        if actual_scheduled_minutes > 0
        else 0.0
    )

    return pd.Series(
        {
            "Latest_Date": latest_date,
            "Recent_Start_Date": session_summary["Session_Date"].min(),
            "Recent_End_Date": session_summary["Session_Date"].max(),
            "Observed_Weeks": observed_weeks,
            "Session_Type_Scope": session_type_scope,
            "Exclude_Obstetrics": exclude_obstetrics,
            "Actual_Sessions_Only_For_Utilisation": (
                actual_sessions_only_for_utilisation
            ),
            "Sessions": utilisation_sessions,
            "Sessions_Per_Week": utilisation_sessions / observed_weeks,
            "Utilisation_Sessions": utilisation_sessions,
            "Utilisation_Sessions_Per_Week": (
                utilisation_sessions / observed_weeks
            ),
            "Planned_Sessions": planned_sessions,
            "Actual_Sessions": actual_sessions,
            "Cancelled_Or_Not_Run_Sessions": cancelled_or_not_run_sessions,
            "Planned_Sessions_Per_Week": planned_sessions / observed_weeks,
            "Actual_Sessions_Per_Week": actual_sessions / observed_weeks,
            "Cancelled_Or_Not_Run_Sessions_Per_Week": (
                cancelled_or_not_run_sessions / observed_weeks
            ),
            "Planned_240_Session_Equivalents": (
                scheduled_minutes / SESSION_STANDARD_MINUTES
            ),
            "Actual_240_Session_Equivalents": (
                actual_scheduled_minutes / SESSION_STANDARD_MINUTES
            ),
            "Utilisation_240_Session_Equivalents": (
                utilisation_scheduled_minutes / SESSION_STANDARD_MINUTES
            ),
            "Planned_240_Session_Equivalents_Per_Week": (
                scheduled_minutes / SESSION_STANDARD_MINUTES / observed_weeks
            ),
            "Actual_240_Session_Equivalents_Per_Week": (
                actual_scheduled_minutes / SESSION_STANDARD_MINUTES / observed_weeks
            ),
            "Utilisation_240_Session_Equivalents_Per_Week": (
                utilisation_scheduled_minutes
                / SESSION_STANDARD_MINUTES
                / observed_weeks
            ),
            "Average_Session_Minutes": utilisation_session_summary[
                "Scheduled_Minutes"
            ].mean(),
            "Median_Session_Minutes": utilisation_session_summary[
                "Scheduled_Minutes"
            ].median(),
            "Average_Case_Duration_Minutes": avg_case_duration,
            "Completed_Cases": completed_cases,
            "Cases_With_Valid_Touch": cases_with_valid_touch,
            "Cancelled_Cases": cancelled_cases,
            "Scheduled_Minutes": utilisation_scheduled_minutes,
            "Planned_Scheduled_Minutes": scheduled_minutes,
            "Actual_Session_Scheduled_Minutes": actual_scheduled_minutes,
            "Utilisation_Scheduled_Minutes": utilisation_scheduled_minutes,
            "Touch_Minutes": touch_minutes,
            "Touch_Minutes_Per_Week": touch_minutes / observed_weeks,
            "Invalid_Touch_Time_Rows": invalid_touch_rows,
            "Invalid_Scheduled_Sessions": len(invalid_scheduled_sessions),
            "Utilisation": utilisation,
            "Actual_Session_Utilisation": actual_session_utilisation,
        }
    )


def summarise_theatre_session_type_split(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    touch_time_column: str = "Model_Hospital_Touch_Minutes",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required = [
        "Theatre session ID",
        "Booked Operation Date",
        "Scheduled start time(Session)",
        "Scheduled finish time(Session)",
    ]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    work = df.dropna(subset=required).copy()

    if start_date is not None:
        work = work[work["Booked Operation Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        work = work[work["Booked Operation Date"] <= pd.to_datetime(end_date)]

    if work.empty:
        return pd.DataFrame()

    work["Session_Type"] = _session_type_label(work)
    work["Session_Has_Obstetrics"] = _session_has_obstetrics(work)
    work["Scheduled_Minutes"] = (
        work["Scheduled finish time(Session)"]
        - work["Scheduled start time(Session)"]
    ).dt.total_seconds() / 60
    work.loc[work["Scheduled_Minutes"] < 0, "Scheduled_Minutes"] += 24 * 60

    if touch_time_column not in work.columns:
        touch_time_column = "Case Touch time (minutes)"
    work[touch_time_column] = pd.to_numeric(
        work[touch_time_column],
        errors="coerce",
    ).fillna(0)
    work["Valid_Touch_Minutes"] = work[touch_time_column].where(
        work[touch_time_column].between(0, 720),
        0,
    )

    for col in ["Actual start time(Session)", "Actual finish time(Session)"]:
        if col not in work.columns:
            work[col] = pd.NaT

    work["Completed_Cases"] = pd.to_numeric(
        work["Number of cases completed"],
        errors="coerce",
    ).fillna(0)

    session_summary = (
        work.groupby(["Booked Operation Date", "Theatre session ID"], as_index=False)
        .agg(
            Session_Type=("Session_Type", "first"),
            Session_Has_Obstetrics=("Session_Has_Obstetrics", "max"),
            Scheduled_Minutes=("Scheduled_Minutes", "first"),
            Touch_Minutes=("Valid_Touch_Minutes", "sum"),
            Completed_Cases=("Completed_Cases", "sum"),
            Actual_Start=("Actual start time(Session)", "min"),
            Actual_Finish=("Actual finish time(Session)", "max"),
        )
    )
    session_summary["Actual_Session_Flag"] = (
        (session_summary["Touch_Minutes"] > 0)
        | (session_summary["Completed_Cases"] > 0)
        | session_summary["Actual_Start"].notna()
        | session_summary["Actual_Finish"].notna()
    )
    session_summary["Valid_Scheduled_Session"] = session_summary[
        "Scheduled_Minutes"
    ].between(30, 720)

    rows = []
    split_order = [
        ("Whole set (all session types)", session_summary),
        (
            "Elective",
            session_summary[session_summary["Session_Type"] == "Elective"],
        ),
        (
            "Non-elective",
            session_summary[session_summary["Session_Type"] != "Elective"],
        ),
        (
            "Elective excl obstetrics (model baseline)",
            session_summary[
                (session_summary["Session_Type"] == "Elective")
                & (~session_summary["Session_Has_Obstetrics"])
            ],
        ),
        (
            "Emergency",
            session_summary[session_summary["Session_Type"] == "Emergency"],
        ),
        (
            "Mixed elective/emergency",
            session_summary[
                session_summary["Session_Type"] == "Mixed elective/emergency"
            ],
        ),
        (
            "Unknown session type",
            session_summary[session_summary["Session_Type"] == "Unknown"],
        ),
    ]

    for label, subset in split_order:
        if subset.empty and label != "Unknown session type":
            continue

        valid_actual = subset[
            subset["Valid_Scheduled_Session"] & subset["Actual_Session_Flag"]
        ]
        scheduled_minutes = valid_actual["Scheduled_Minutes"].sum()
        touch_minutes = valid_actual["Touch_Minutes"].sum()
        utilisation = (
            touch_minutes / scheduled_minutes if scheduled_minutes > 0 else 0.0
        )
        rows.append(
            {
                "Session type": label,
                "Planned sessions": len(subset),
                "Actual sessions delivered": int(
                    subset["Actual_Session_Flag"].sum()
                ),
                "Valid actual sessions used for utilisation": len(valid_actual),
                "Actual 240-min session equivalents": (
                    scheduled_minutes / SESSION_STANDARD_MINUTES
                ),
                "Completed cases": subset["Completed_Cases"].sum(),
                "Touch minutes used": touch_minutes,
                "Scheduled minutes used": scheduled_minutes,
                "Utilisation": utilisation,
                "Obstetric sessions": int(subset["Session_Has_Obstetrics"].sum()),
                "Invalid / 24hr sessions": int(
                    (~subset["Valid_Scheduled_Session"]).sum()
                ),
            }
        )

    return pd.DataFrame(rows)
