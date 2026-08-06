from pathlib import Path
import pandas as pd
import streamlit as st

from src.utils.specialty_standardisation import standardise_specialty_series


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


@st.cache_data(show_spinner=False)
def load_inpatient_data(path: str = "data/raw/Inpatient/Inpatients.csv") -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Inpatient path not found: {path}")

    if path.is_file():
        return _clean_inpatient_data(_load_single_file(path))

    files = [f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        raise FileNotFoundError(f"No inpatient files found in: {path}")

    dfs = []

    for file in files:
        df = _clean_inpatient_data(_load_single_file(file))
        df["Source_File"] = file.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Spell ID", "Admission datetime"])

    return combined.reset_index(drop=True)


def _load_single_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, encoding="cp1252")
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported inpatient file type: {file_path}")

    df.columns = df.columns.str.strip()
    return df


def _clean_inpatient_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Spell ID", "Admission datetime", "Specialty"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required inpatient columns: {missing}")

    df["Spell ID"] = df["Spell ID"].astype(str).str.strip()

    df["Admission datetime"] = pd.to_datetime(
        df["Admission datetime"],
        errors="coerce",
        dayfirst=True,
    )

    if "Discharge datetime" in df.columns:
        df["Discharge datetime"] = pd.to_datetime(
            df["Discharge datetime"],
            errors="coerce",
            dayfirst=True,
        )

    df = df.dropna(subset=["Admission datetime"])

    text_cols = [
        "Status",
        "Division",
        "Specialty",
        "Admitting ward",
        "Current ward",
        "Patient classification",
        "Elective/emergency",
        "Patient age category",
        "Admission method",
        "Source of admission",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["Standardised_Specialty"] = standardise_specialty_series(df["Specialty"])

    df["Admission_Month"] = (
        df["Admission datetime"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if "LoS" in df.columns:
        df["LoS"] = pd.to_numeric(df["LoS"], errors="coerce")

    return df


def validate_inpatient_data(df: pd.DataFrame) -> list[str]:
    warnings = []

    if df.empty:
        warnings.append("Inpatient dataset is empty.")

    duplicate_count = df["Spell ID"].duplicated().sum()

    if duplicate_count > 0:
        warnings.append(f"{duplicate_count:,} duplicate Spell ID values found.")

    return warnings
