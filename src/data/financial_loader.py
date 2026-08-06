from pathlib import Path
import re

import pandas as pd
import streamlit as st


FY_2526_MONTHS = [
    "Apr-25",
    "May-25",
    "Jun-25",
    "Jul-25",
    "Aug-25",
    "Sep-25",
    "Oct-25",
    "Nov-25",
    "Dec-25",
    "Jan-26",
    "Feb-26",
    "Mar-26",
]

FY_2425_MONTHS = [
    "Apr-24",
    "May-24",
    "Jun-24",
    "Jul-24",
    "Aug-24",
    "Sep-24",
    "Oct-24",
    "Nov-24",
    "Dec-24",
    "Jan-25",
    "Feb-25",
    "Mar-25",
]


def parse_finance_number(value) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).strip()

    if not text or text in ["-", "-   ", " -   ", "nan"]:
        return 0.0

    is_bracket_negative = text.startswith("(") and text.endswith(")")
    text = (
        text.replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("£", "")
        .replace("�", "")
        .replace("Â", "")
        .replace(" ", "")
    )

    if text in ["", "-"]:
        return 0.0

    try:
        number = float(text)
    except ValueError:
        return 0.0

    if is_bracket_negative:
        number = -abs(number)

    return number


@st.cache_data(show_spinner=False)
def load_trial_balance(
    path: str = "data/raw/trial balance/Copy of TB 2324 - 2526.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    for col in FY_2425_MONTHS + FY_2526_MONTHS:
        if col in df.columns:
            df[col] = df[col].apply(parse_finance_number)

    df["FY_2425_Total"] = df[[c for c in FY_2425_MONTHS if c in df.columns]].sum(
        axis=1
    )
    df["FY_2526_Total"] = df[[c for c in FY_2526_MONTHS if c in df.columns]].sum(
        axis=1
    )

    search_cols = [
        "Cost Centre  Description",
        "Subjective Description",
        "Expenditure Type",
        "PAF category - level 2",
        "Category",
        "Summary Level",
        "Group Level",
        "Staff Type",
        "Staff Category",
        "Specialty",
        "Service Group",
        "Healthgroup",
        "TAC Detail",
    ]
    available_search_cols = [col for col in search_cols if col in df.columns]
    df["Search_Text"] = (
        df[available_search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    )

    return df


@st.cache_data(show_spinner=False)
def load_staff_cost_data(path: str = "data/raw/Staff cost") -> pd.DataFrame:
    folder = Path(path)
    files = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No staff cost files found in: {folder}")

    dfs = []

    for file in files:
        df = _read_staff_cost_file(file)
        df["Source_File"] = file.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined["Year/Month"] = pd.to_datetime(
        combined["Year/Month"],
        errors="coerce",
    )
    combined = combined.dropna(subset=["Year/Month"])

    combined["Financial_Year"] = combined["Year/Month"].apply(financial_year_label)
    combined["Total cost"] = combined["Total cost"].apply(parse_finance_number)
    combined["WTE equivalent"] = pd.to_numeric(
        combined["WTE equivalent"],
        errors="coerce",
    ).fillna(0)

    text_cols = [
        "Service Name",
        "Cost Centre Code Desc",
        "Subjective Code Desc",
        "Specialty (standardised)",
        "Site / Location",
        "Directorate / Division",
        "Staff group",
        "Pay type",
    ]

    for col in text_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna("").astype(str).str.strip()

    return combined.reset_index(drop=True)


def _read_staff_cost_file(file_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        file_path,
        encoding="cp1252",
        header=None,
        skip_blank_lines=False,
        low_memory=False,
    )

    header_index = None

    for idx, row in raw.iterrows():
        values = [str(value).strip() for value in row.tolist()]
        if "Service Type" in values and "Total cost" in values:
            header_index = idx
            break

    if header_index is None:
        raise ValueError(f"Could not find staff cost header in: {file_path}")

    headers = raw.iloc[header_index].tolist()
    df = raw.iloc[header_index + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")
    df.columns = df.columns.astype(str).str.strip()

    return df


def financial_year_label(date_value: pd.Timestamp) -> str:
    year = date_value.year

    if date_value.month >= 4:
        return f"{str(year)[-2:]}/{str(year + 1)[-2:]}"

    return f"{str(year - 1)[-2:]}/{str(year)[-2:]}"


def surgical_theatre_mask(df: pd.DataFrame) -> pd.Series:
    search_cols = [
        "Specialty",
        "Service Group",
        "Healthgroup",
        "Cost Centre  Description",
        "Combined CC & DESC",
        "Specialty (standardised)",
        "Service Name",
        "Cost Centre Code Desc",
        "Directorate / Division",
        "Site / Location",
    ]
    available = [col for col in search_cols if col in df.columns]

    if not available:
        return pd.Series(False, index=df.index)

    text = df[available].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    terms = [
        "theatre",
        "anaesth",
        "surgery",
        "surgical",
        "orthopaedic",
        "trauma",
        "urology",
        "ophthalmology",
        "ent",
        "endoscopy",
        "gastroenterology",
        "vanguard",
        "day stay",
    ]
    pattern = "|".join(re.escape(term) for term in terms)

    return text.str.contains(pattern, regex=True, na=False)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return ""

    return f"£{value:,.0f}"


def format_optional_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Not quantified from raw data"

    return format_currency(value)
