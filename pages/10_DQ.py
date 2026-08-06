import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.data.rtt_loader import load_all_rtt_files


st.set_page_config(
    page_title="Data Quality & Investigation",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Data Quality & Investigation")

st.caption(
    "Assurance and investigation view for checking completeness, duplicates, date coverage and category quality across RTT, referrals and outpatients."
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_CONFIG = {
    "Referrals": {
        "path": "data/raw/Refferals",
        "id_col": "Referral_ID",
        "date_col": "Referral_Received_Date",
        "key_fields": [
            "Referral_ID",
            "Referral_Received_Date",
            "TFC_Name",
            "ReferralSource",
            "ReferralType",
            "CCG",
            "Medical_Priority_Desc",
            "Graded_Med_Priority_Desc",
        ],
        "category_fields": [
            "TFC_Name",
            "ReferralSource",
            "ReferralType",
            "CCG",
            "Medical_Priority_Desc",
            "Graded_Med_Priority_Desc",
        ],
    },
    "Outpatients": {
        "path": "data/raw/Outpatients",
        "id_col": "Contact_ID",
        "date_col": "Contact_Start",
        "key_fields": [
            "Contact_ID",
            "Contact_Start",
            "Contact_End",
            "ContactClinicPerfUnit",
            "ContactClinicPerfUnit_Type",
            "Type",
            "TreatmentFunctionDesc",
            "Status",
            "ContactVisitType",
        ],
        "category_fields": [
            "TreatmentFunctionDesc",
            "ContactClinicPerfUnit",
            "ContactClinicPerfUnit_Type",
            "Type",
            "Status",
            "ContactVisitType",
        ],
    },
}


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_folder(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        return load_single_file(path)

    files = [
        file for file in path.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"No CSV or Excel files found in: {path}")

    dfs = []

    for file in files:
        df = load_single_file(file)
        df["Source_File"] = file.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_single_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df.columns = df.columns.str.strip()
    return df


def clean_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce",
            dayfirst=True,
        )

        df["DQ_Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    return df


def dataset_profile(df: pd.DataFrame, dataset_name: str, id_col: str, date_col: str) -> dict:
    rows = len(df)
    columns = len(df.columns)

    unique_ids = df[id_col].nunique() if id_col in df.columns else None
    duplicate_rows = df[id_col].duplicated().sum() if id_col in df.columns else None
    duplicate_rate = duplicate_rows / rows if rows > 0 and duplicate_rows is not None else None

    if date_col in df.columns:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
    else:
        min_date = None
        max_date = None

    return {
        "Dataset": dataset_name,
        "Rows": rows,
        "Columns": columns,
        "ID Column": id_col if id_col in df.columns else "Not found",
        "Unique IDs": unique_ids,
        "Duplicate ID Rows": duplicate_rows,
        "Duplicate Rate": duplicate_rate,
        "Date Column": date_col if date_col in df.columns else "Not found",
        "Earliest Date": min_date,
        "Latest Date": max_date,
    }


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Values": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
            "Non-Missing Values": df.notna().sum().values,
            "Unique Values": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )

    return result.sort_values("Missing %", ascending=False)


def duplicate_summary(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if id_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    duplicate_df = df[df[id_col].duplicated(keep=False)].copy()

    summary = (
        duplicate_df.groupby(id_col)
        .size()
        .reset_index(name="Duplicate Row Count")
        .sort_values("Duplicate Row Count", ascending=False)
    )

    return summary, duplicate_df


def monthly_volume(df: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work = clean_dates(work, date_col)

    if "DQ_Month" not in work.columns:
        return pd.DataFrame()

    if id_col in work.columns:
        result = (
            work.groupby("DQ_Month")
            .agg(
                Rows=("DQ_Month", "size"),
                Unique_IDs=(id_col, "nunique"),
            )
            .reset_index()
            .sort_values("DQ_Month")
        )
    else:
        result = (
            work.groupby("DQ_Month")
            .agg(Rows=("DQ_Month", "size"))
            .reset_index()
            .sort_values("DQ_Month")
        )

    return result


def category_summary(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in df.columns:
        return pd.DataFrame()

    result = (
        df[field]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    result.columns = [field, "Rows"]
    return result


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

loaded_data = {}
profiles = []

for dataset_name, config in DATASET_CONFIG.items():
    try:
        df = load_raw_folder(config["path"])
        df = clean_dates(df, config["date_col"])
        loaded_data[dataset_name] = df

        profiles.append(
            dataset_profile(
                df=df,
                dataset_name=dataset_name,
                id_col=config["id_col"],
                date_col=config["date_col"],
            )
        )

    except Exception as e:
        loaded_data[dataset_name] = pd.DataFrame()
        profiles.append(
            {
                "Dataset": dataset_name,
                "Rows": 0,
                "Columns": 0,
                "ID Column": config["id_col"],
                "Unique IDs": None,
                "Duplicate ID Rows": None,
                "Duplicate Rate": None,
                "Date Column": config["date_col"],
                "Earliest Date": None,
                "Latest Date": None,
                "Load Error": str(e),
            }
        )


try:
    rtt_df = load_all_rtt_files()
    rtt_df.columns = rtt_df.columns.str.strip()
    loaded_data["RTT"] = rtt_df

    profiles.append(
        {
            "Dataset": "RTT",
            "Rows": len(rtt_df),
            "Columns": len(rtt_df.columns),
            "ID Column": "Not applicable",
            "Unique IDs": None,
            "Duplicate ID Rows": None,
            "Duplicate Rate": None,
            "Date Column": "Month",
            "Earliest Date": rtt_df["Month"].min() if "Month" in rtt_df.columns else None,
            "Latest Date": rtt_df["Month"].max() if "Month" in rtt_df.columns else None,
        }
    )

except Exception as e:
    loaded_data["RTT"] = pd.DataFrame()
    profiles.append(
        {
            "Dataset": "RTT",
            "Rows": 0,
            "Columns": 0,
            "ID Column": "Not applicable",
            "Unique IDs": None,
            "Duplicate ID Rows": None,
            "Duplicate Rate": None,
            "Date Column": "Month",
            "Earliest Date": None,
            "Latest Date": None,
            "Load Error": str(e),
        }
    )


profile_df = pd.DataFrame(profiles)


# ---------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------

st.subheader("Dataset Quality Overview")

c1, c2, c3, c4 = st.columns(4)

total_rows = profile_df["Rows"].sum()
datasets_loaded = (profile_df["Rows"] > 0).sum()
total_duplicate_rows = profile_df["Duplicate ID Rows"].dropna().sum()
datasets_with_errors = profile_df.get("Load Error", pd.Series(dtype=str)).notna().sum()

c1.metric("Datasets loaded", f"{datasets_loaded}")
c2.metric("Total rows loaded", f"{total_rows:,.0f}")
c3.metric("Duplicate ID rows", f"{total_duplicate_rows:,.0f}")
c4.metric("Datasets with load issues", f"{datasets_with_errors}")


with st.expander("How to interpret this page", expanded=False):
    st.markdown(
        """
This page is used to investigate whether the data powering the platform is suitable for analysis and modelling.

It checks:

- row volumes
- duplicate identifiers
- missing values
- date coverage
- incomplete or unusual months
- category values such as specialties, status and clinic type
- source file overlap

This is not a performance page. It is an assurance and investigation page.

A high duplicate count does not always mean bad data. It may mean overlapping extracts, repeated snapshots, lifecycle records or multiple source files covering the same period.
        """
    )


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Dataset Summary",
        "Duplicates",
        "Completeness",
        "Date Coverage",
        "Category Checks",
        "Raw Investigation",
    ]
)


# ---------------------------------------------------------
# Tab 1: Dataset Summary
# ---------------------------------------------------------

with tab1:
    st.subheader("Dataset summary")

    display_profile = profile_df.copy()

    if "Duplicate Rate" in display_profile.columns:
        display_profile["Duplicate Rate"] = (
            display_profile["Duplicate Rate"] * 100
        ).round(2)

    st.dataframe(display_profile, use_container_width=True)

    st.info(
        """
Use this view to check whether each dataset has loaded, how many rows it contains,
whether the expected identifier is present, and whether the latest date looks reasonable.
        """
    )


# ---------------------------------------------------------
# Tab 2: Duplicates
# ---------------------------------------------------------

with tab2:
    st.subheader("Duplicate identifier investigation")

    selected_dataset = st.selectbox(
        "Select dataset for duplicate investigation",
        ["Referrals", "Outpatients"],
        key="duplicate_dataset_select",
    )

    config = DATASET_CONFIG[selected_dataset]
    df = loaded_data[selected_dataset]
    id_col = config["id_col"]

    if df.empty:
        st.warning(f"No data available for {selected_dataset}.")

    elif id_col not in df.columns:
        st.warning(f"ID column `{id_col}` not found in {selected_dataset}.")

    else:
        duplicate_id_summary, duplicate_rows = duplicate_summary(df, id_col)

        duplicate_row_count = len(duplicate_rows)
        duplicate_id_count = duplicate_id_summary[id_col].nunique() if not duplicate_id_summary.empty else 0
        duplicate_rate = duplicate_row_count / len(df) if len(df) > 0 else 0

        c1, c2, c3 = st.columns(3)

        c1.metric("Duplicate rows", f"{duplicate_row_count:,.0f}")
        c2.metric("Duplicate IDs", f"{duplicate_id_count:,.0f}")
        c3.metric("Duplicate row rate", f"{duplicate_rate:.1%}")

        if duplicate_rows.empty:
            st.success("No duplicate IDs detected.")

        else:
            st.subheader("Most repeated IDs")
            st.dataframe(duplicate_id_summary.head(100), use_container_width=True)

            st.subheader("Duplicate row examples")
            st.dataframe(duplicate_rows.head(500), use_container_width=True)

            if "Source_File" in duplicate_rows.columns:
                st.subheader("Duplicate rows by source file")

                source_dup_df = (
                    duplicate_rows.groupby("Source_File")
                    .size()
                    .reset_index(name="Duplicate Rows")
                    .sort_values("Duplicate Rows", ascending=False)
                )

                fig = px.bar(
                    source_dup_df.head(20),
                    x="Duplicate Rows",
                    y="Source_File",
                    orientation="h",
                    title="Duplicate rows by source file",
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Duplicate rows",
                    yaxis_title="Source file",
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(source_dup_df, use_container_width=True)

        st.info(
            """
Interpretation: duplicate IDs may be caused by overlapping extracts, repeated snapshots,
multiple source files covering the same period, or genuine lifecycle records. Check the duplicate examples
before deciding whether to deduplicate.
            """
        )


# ---------------------------------------------------------
# Tab 3: Completeness
# ---------------------------------------------------------

with tab3:
    st.subheader("Missing value and completeness checks")

    selected_dataset = st.selectbox(
        "Select dataset for completeness check",
        list(loaded_data.keys()),
        key="completeness_dataset_select",
    )

    df = loaded_data[selected_dataset]

    if df.empty:
        st.warning(f"No data available for {selected_dataset}.")
    else:
        missing_df = missingness_summary(df)

        st.dataframe(missing_df, use_container_width=True)

        top_missing = missing_df.head(20)

        fig = px.bar(
            top_missing,
            x="Missing %",
            y="Column",
            orientation="h",
            title=f"Top missing fields - {selected_dataset}",
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Missing %",
            yaxis_title="Column",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
Interpretation: fields with high missingness may still be valid if they are not required for analysis.
However, high missingness in key fields such as dates, specialty, status or identifiers may affect model reliability.
            """
        )


# ---------------------------------------------------------
# Tab 4: Date Coverage
# ---------------------------------------------------------

with tab4:
    st.subheader("Date coverage and monthly volume checks")

    selected_dataset = st.selectbox(
        "Select dataset for date coverage",
        ["Referrals", "Outpatients"],
        key="date_dataset_select",
    )

    config = DATASET_CONFIG[selected_dataset]
    df = loaded_data[selected_dataset]

    date_col = config["date_col"]
    id_col = config["id_col"]

    if df.empty:
        st.warning(f"No data available for {selected_dataset}.")

    elif date_col not in df.columns:
        st.warning(f"Date column `{date_col}` not found.")

    else:
        month_df = monthly_volume(df, date_col, id_col)

        if month_df.empty:
            st.warning("No monthly date coverage could be calculated.")

        else:
            fig = px.line(
                month_df,
                x="DQ_Month",
                y="Rows",
                markers=True,
                title=f"Monthly row volume - {selected_dataset}",
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Rows",
            )

            st.plotly_chart(fig, use_container_width=True)

            if "Unique_IDs" in month_df.columns:
                fig2 = px.line(
                    month_df,
                    x="DQ_Month",
                    y="Unique_IDs",
                    markers=True,
                    title=f"Monthly unique IDs - {selected_dataset}",
                )

                fig2.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Unique IDs",
                )

                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Monthly coverage table")
            st.dataframe(month_df, use_container_width=True)

            st.info(
                """
Interpretation: sudden drops in the latest month may indicate incomplete extracts.
Sudden spikes may indicate duplicated files, changed extract logic or genuine operational growth.
                """
            )


# ---------------------------------------------------------
# Tab 5: Category Checks
# ---------------------------------------------------------

with tab5:
    st.subheader("Category and coding checks")

    selected_dataset = st.selectbox(
        "Select dataset for category checks",
        ["Referrals", "Outpatients"],
        key="category_dataset_select",
    )

    config = DATASET_CONFIG[selected_dataset]
    df = loaded_data[selected_dataset]

    available_category_fields = [
        field for field in config["category_fields"]
        if field in df.columns
    ]

    if df.empty:
        st.warning(f"No data available for {selected_dataset}.")

    elif not available_category_fields:
        st.warning("No configured category fields found.")

    else:
        selected_field = st.selectbox(
            "Select category field",
            available_category_fields,
            key="category_field_select",
        )

        cat_df = category_summary(df, selected_field)

        st.subheader(f"Distribution of {selected_field}")

        fig = px.bar(
            cat_df.head(30),
            x="Rows",
            y=selected_field,
            orientation="h",
            title=f"Top values for {selected_field}",
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Rows",
            yaxis_title=selected_field,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cat_df, use_container_width=True)

        st.info(
            """
Interpretation: use this to spot inconsistent specialty naming, unexpected statuses,
blank values, unknown categories or coding changes over time.
            """
        )


# ---------------------------------------------------------
# Tab 6: Raw Investigation
# ---------------------------------------------------------

with tab6:
    st.subheader("Raw data investigation")

    selected_dataset = st.selectbox(
        "Select dataset to inspect",
        list(loaded_data.keys()),
        key="raw_dataset_select",
    )

    df = loaded_data[selected_dataset]

    if df.empty:
        st.warning(f"No data available for {selected_dataset}.")

    else:
        st.write(f"Rows: **{len(df):,.0f}**")
        st.write(f"Columns: **{len(df.columns):,.0f}**")

        st.subheader("Columns")
        st.write(list(df.columns))

        st.subheader("Raw preview")
        st.dataframe(df.head(500), use_container_width=True)

        if selected_dataset in DATASET_CONFIG:
            config = DATASET_CONFIG[selected_dataset]
            id_col = config["id_col"]

            if id_col in df.columns:
                st.subheader("Search by ID")

                search_id = st.text_input(
                    f"Enter {id_col}",
                    key=f"search_{selected_dataset}_{id_col}",
                )

                if search_id:
                    result = df[df[id_col].astype(str) == str(search_id)]
                    st.write(f"Matching rows: **{len(result):,.0f}**")
                    st.dataframe(result, use_container_width=True)