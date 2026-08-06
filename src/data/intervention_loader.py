from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_INTERVENTION_PATH = "data/raw/Interventions/Opportunity model.csv"


@st.cache_data(show_spinner=False)
def load_intervention_data(
    path: str = DEFAULT_INTERVENTION_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Intervention model not found: {path}")

    if path.suffix.lower() == ".csv":
        raw_df = _read_csv_with_fallback_encoding(path)
    elif path.suffix.lower() == ".xlsx":
        raw_df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".xls":
        raw_df = pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError(
            f"Unsupported intervention file type: {path.suffix}. Use CSV or Excel."
        )

    raw_df.columns = [_normalise_column_name(col) for col in raw_df.columns]

    rename_map = {
        "group theme": "Group_Theme",
        "intervention": "Intervention",
        "validation 1 5 does it reflect pah reality": "Validation_Score",
        "evidence 1 5 national data girft support": "Evidence_Score",
        "impact 1 5 elective recovery benefit": "Impact_Score",
        "deliverability 1 5 feasible at pace": "Deliverability_Score",
        "total score": "Total_Score",
        "current baseline pah position": "Current_Baseline",
        "girft nhse best practice benchmark": "Benchmark",
        "pah target to validate": "PAH_Target",
        "assumption detail lever narrative": "Assumption_Detail",
        "additional cases slots per week": "Additional_Cases_Per_Week",
        "weeks to recover position": "Weeks_To_Recover",
        "investment required": "Investment_Required",
        "wte required peak": "WTE_Required_Peak",
        "annual financial benefit": "Annual_Financial_Benefit",
        "wte released steady state": "WTE_Released_Steady_State",
        "net position benefit minus cost": "Net_Position",
        "recommendation auto": "Recommendation",
        "validation status": "Validation_Status",
    }

    df = raw_df.rename(columns=rename_map).copy()
    df = df.dropna(how="all")

    required_cols = [
        "Intervention",
        "PAH_Target",
        "Additional_Cases_Per_Week",
        "Weeks_To_Recover",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing expected intervention columns after cleaning: {missing}. "
            f"Columns found after normalisation: {list(df.columns)}"
        )

    if "Group_Theme" not in df.columns:
        df["Group_Theme"] = "Uncategorised"

    df = df.dropna(subset=["Intervention"])

    text_cols = [
        "Group_Theme",
        "Intervention",
        "Current_Baseline",
        "Benchmark",
        "PAH_Target",
        "Assumption_Detail",
        "Recommendation",
        "Validation_Status",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_cols = [
        "Validation_Score",
        "Evidence_Score",
        "Impact_Score",
        "Deliverability_Score",
        "Total_Score",
        "Investment_Required",
        "WTE_Required_Peak",
        "Annual_Financial_Benefit",
        "WTE_Released_Steady_State",
        "Net_Position",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric_series(df[col])

    df["Additional_Cases_Per_Week"] = _clean_numeric_series(
        df["Additional_Cases_Per_Week"]
    ).fillna(0)

    df["Weeks_To_Recover"] = _clean_numeric_series(
        df["Weeks_To_Recover"]
    ).fillna(0)

    df["Total_Additional_Cases"] = (
        df["Additional_Cases_Per_Week"] * df["Weeks_To_Recover"]
    )

    df["Annualised_Additional_Activity"] = (
        df["Additional_Cases_Per_Week"] * 52
    )

    df["Monthly_Additional_Activity"] = (
        df["Additional_Cases_Per_Week"] * 4.33
    )

    df["Is_Modelable"] = (
        (df["Additional_Cases_Per_Week"] > 0)
        & (df["Weeks_To_Recover"] > 0)
    )

    return df.reset_index(drop=True)


def _read_csv_with_fallback_encoding(path: Path) -> pd.DataFrame:
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "ISO-8859-1",
    ]

    last_error = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode CSV using fallback encodings. Last error: {last_error}",
    )


def _normalise_column_name(col) -> str:
    """
    Make messy Excel/CSV headers stable.

    Handles:
    - line breaks
    - £ symbols
    - brackets
    - en-dashes
    - slashes
    - repeated spaces
    """

    text = str(col)

    replacements = {
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "£": "",
        "–": " ",
        "-": " ",
        "/": " ",
        "(": " ",
        ")": " ",
        "’": "",
        "'": "",
        "?": "",
        ":": " ",
        ";": " ",
        ",": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = " ".join(text.split())
    text = text.lower().strip()

    return text


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("N/A", "", regex=False)
        .str.replace("n/a", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def validate_intervention_data(df: pd.DataFrame) -> list[str]:
    warnings = []

    if df.empty:
        warnings.append("Intervention dataset is empty.")

    required_cols = [
        "Group_Theme",
        "Intervention",
        "PAH_Target",
        "Additional_Cases_Per_Week",
        "Weeks_To_Recover",
        "Is_Modelable",
    ]

    for col in required_cols:
        if col not in df.columns:
            warnings.append(f"Missing intervention column: {col}")

    if "Is_Modelable" in df.columns:
        non_modelable = (~df["Is_Modelable"]).sum()
        if non_modelable > 0:
            warnings.append(
                f"{non_modelable:,} interventions do not have numeric cases/week and recovery weeks."
            )

    return warnings