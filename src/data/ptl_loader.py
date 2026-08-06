from pathlib import Path

import pandas as pd
import streamlit as st


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


@st.cache_data(show_spinner=False)
def load_ptl_data(path: str = "data/raw/ptl") -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PTL path not found: {path}")

    if path.is_file():
        df = _load_single_ptl_file(path)
        return _clean_ptl_data(df)

    files = [
        file
        for file in path.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"No CSV or Excel PTL files found in: {path}")

    dfs = []

    for file in files:
        df = _load_single_ptl_file(file)
        df["Source_File"] = file.name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    return _clean_ptl_data(combined_df)


def _load_single_ptl_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported PTL file type: {file_path}")

    df.columns = df.columns.str.strip()
    return df


def _clean_ptl_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Month", "Size"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required PTL columns: {missing}")

    df["Month"] = pd.to_datetime(
        df["Month"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(subset=["Month"])

    df["PTL_Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()

    df["Size"] = pd.to_numeric(df["Size"], errors="coerce").fillna(0)

    if "RTT PTL" in df.columns:
        df["RTT PTL"] = df["RTT PTL"].fillna("Unknown").astype(str).str.strip()

    return df


def summarise_ptl_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["PTL_Month", "PTL Size"])

    return (
        df.groupby("PTL_Month")
        .agg(**{"PTL Size": ("Size", "sum")})
        .reset_index()
        .sort_values("PTL_Month")
    )