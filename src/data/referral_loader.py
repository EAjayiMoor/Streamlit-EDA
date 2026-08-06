from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.specialty_standardisation import standardise_specialty_series


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


@st.cache_data(show_spinner=False)
def load_referral_data(path: str = "data/raw/Refferals") -> pd.DataFrame:
    """
    Load referral data from a file or folder.

    Default folder:
    data/raw/Refferals

    If a folder is provided, all CSV / Excel files inside it are loaded
    and combined into one dataframe.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Referral path not found: {path}")

    if path.is_file():
        return _load_single_referral_file(path)

    if path.is_dir():
        files = [
            file
            for file in path.iterdir()
            if file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            raise FileNotFoundError(
                f"No CSV or Excel referral files found in folder: {path}"
            )

        dfs = []

        for file in files:
            df = _load_single_referral_file(file)
            df["Source_File"] = file.name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    raise ValueError(f"Invalid referral path: {path}")


def _load_single_referral_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported referral file type: {file_path}")

    return _clean_referral_data(df)


def _clean_referral_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    required_columns = [
        "Referral_ID",
        "Referral_Received_Date",
        "TFC_Name",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required referral columns: {missing}")

    df["Referral_Received_Date"] = pd.to_datetime(
        df["Referral_Received_Date"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(subset=["Referral_Received_Date"])
    df["Referral_ID"] = df["Referral_ID"].astype(str)
    df["Standardised_Specialty"] = standardise_specialty_series(df["TFC_Name"])
    text_columns = [
        "TFC_Code",
        "TFC_Name",
        "ReferralSource",
        "ReferralType",
        "Referral_Requesting_Unit",
        "CCGShortCode",
        "CCG",
        "Medical_Priority_Desc",
        "Graded_Med_Priority_Desc",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["Referral_Month"] = (
        df["Referral_Received_Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df["Referral_Year"] = df["Referral_Received_Date"].dt.year

    return df


def validate_referral_data(df: pd.DataFrame) -> list[str]:
    warnings = []

    optional_columns = [
        "ReferralSource",
        "ReferralType",
        "Referral_Requesting_Unit",
        "CCGShortCode",
        "CCG",
        "Medical_Priority_Desc",
        "Graded_Med_Priority_Desc",
    ]

    for col in optional_columns:
        if col not in df.columns:
            warnings.append(f"Optional column missing: {col}")

    if df.empty:
        warnings.append("Referral dataset is empty after date cleaning.")

    duplicate_count = df["Referral_ID"].duplicated().sum()

    if duplicate_count > 0:
        warnings.append(f"{duplicate_count:,} duplicate Referral_ID values found.")

    return warnings
def standardise_specialties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create canonical specialty naming layer.
    """

    df = df.copy()

    df["Standardised_Specialty"] = standardise_specialty_series(
        df["Standardised_Specialty"]
    )

    return df
