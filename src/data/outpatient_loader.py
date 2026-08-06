from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.specialty_standardisation import standardise_specialty_series


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


CONTACT_VISIT_TYPE_MAPPING = {
    "fu telephone consultation": "Follow Up",
    "fu attendance face to face": "Follow Up",
    "fu telemedicine consultation": "Follow Up",
    "follow up": "Follow Up",
    "first attendance face to face": "First attendance",
    "first telephone consultation": "First attendance",
    "first telephone consulation": "First attendance",
    "new": "First attendance",
}


@st.cache_data(show_spinner=False)
def load_outpatient_data(path: str = "data/raw/Outpatients") -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Outpatient path not found: {path}")

    if path.is_file():
        df = _load_single_outpatient_file(path)
        return _finalise_outpatient_data(df)

    if path.is_dir():
        files = [
            file
            for file in path.iterdir()
            if file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            raise FileNotFoundError(
                f"No CSV or Excel outpatient files found in folder: {path}"
            )

        dfs = []
        skipped_files = []

        for file in files:
            try:
                df = _load_single_outpatient_file(file)
            except ValueError as exc:
                if "Missing required outpatient columns" in str(exc):
                    skipped_files.append(file.name)
                    continue
                raise
            df["Source_File"] = file.name
            dfs.append(df)

        if not dfs:
            skipped = ", ".join(skipped_files)
            raise FileNotFoundError(
                "No appointment-level outpatient files found in folder: "
                f"{path}. Skipped non-appointment files: {skipped}"
            )

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.attrs["Skipped_Files"] = skipped_files

        return _finalise_outpatient_data(combined_df)

    raise ValueError(f"Invalid outpatient path: {path}")


def _load_single_outpatient_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported outpatient file type: {file_path}")

    return _clean_outpatient_data(df)


def _clean_outpatient_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    required_columns = [
        "Contact_ID",
        "Contact_Start",
        "TreatmentFunctionDesc",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required outpatient columns: {missing}")

    df["Contact_Start"] = pd.to_datetime(
        df["Contact_Start"],
        errors="coerce",
        dayfirst=True,
    )

    if "Contact_End" in df.columns:
        df["Contact_End"] = pd.to_datetime(
            df["Contact_End"],
            errors="coerce",
            dayfirst=True,
        )

    df = df.dropna(subset=["Contact_Start"])

    df["Contact_ID"] = df["Contact_ID"].astype(str).str.strip()

    text_columns = [
        "ContactClinicPerfUnit",
        "ContactClinicPerfUnit_Type",
        "Type",
        "TreatmentFunctionDesc",
        "Status",
        "ContactVisitType",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    if "ContactVisitType" in df.columns:
        df["ContactVisitType_Group"] = (
            df["ContactVisitType"]
            .str.lower()
            .map(CONTACT_VISIT_TYPE_MAPPING)
            .fillna(df["ContactVisitType"])
        )

    df["Standardised_Specialty"] = standardise_specialty_series(
        df["TreatmentFunctionDesc"]
    )

    df["Contact_Month"] = (
        df["Contact_Start"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df["Contact_Year"] = df["Contact_Start"].dt.year

    if "Contact_End" in df.columns:
        df["Contact_Duration_Minutes"] = (
            df["Contact_End"] - df["Contact_Start"]
        ).dt.total_seconds() / 60

    return df


def _finalise_outpatient_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final dataset-level cleaning.

    This is applied AFTER files are combined, so it removes duplicates
    across multiple extracts as well as within individual files.
    """

    df = df.copy()

    # Remove exact duplicate rows first
    df = df.drop_duplicates()

    # Keep one record per Contact_ID across all loaded files
    df = df.sort_values(
        by=["Contact_ID", "Contact_Start"],
        ascending=[True, True],
    )

    df = df.drop_duplicates(
        subset=["Contact_ID"],
        keep="last",
    )

    return df.reset_index(drop=True)


def validate_outpatient_data(df: pd.DataFrame) -> list[str]:
    warnings = []

    optional_columns = [
        "Contact_End",
        "ContactClinicPerfUnit",
        "ContactClinicPerfUnit_Type",
        "Type",
        "Status",
        "ContactVisitType",
    ]

    for col in optional_columns:
        if col not in df.columns:
            warnings.append(f"Optional column missing: {col}")

    if df.empty:
        warnings.append("Outpatient dataset is empty after date cleaning.")

    duplicate_count = df["Contact_ID"].duplicated().sum()

    if duplicate_count > 0:
        warnings.append(f"{duplicate_count:,} duplicate Contact_ID values found.")

    return warnings
