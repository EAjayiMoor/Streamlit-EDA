from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.inpatient_loader import load_inpatient_data
from src.transforms.inpatient_transform import (
    filter_inpatients,
)

from src.data.outpatient_loader import (
    load_outpatient_data,
    validate_outpatient_data,
)

from src.data.ptl_loader import (
    load_ptl_data,
    summarise_ptl_by_month,
)

from src.transforms.outpatient_transform import (
    filter_outpatients,
    summarise_outpatients_by_month,
    summarise_outpatients_by_specialty,
    summarise_outpatients_by_clinic,
    summarise_outpatients_by_clinic_type,
    summarise_outpatients_by_type,
    summarise_outpatients_by_status,
    summarise_outpatients_by_visit_type,
    outpatient_heatmap_matrix,
    add_monthly_contact_growth,
    outpatient_growth_signal,
    summarise_checked_flow_by_month,
)
from src.utils.specialty_standardisation import standardise_specialty_series


st.set_page_config(
    page_title="Outpatient Flow & Capacity Intelligence",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Outpatient Flow & Capacity Intelligence")

st.caption(
    "Outpatient activity intelligence showing how patients move through the middle layer of the elective pathway."
)

DEFAULT_OUTPATIENT_PATH = "data/raw/Outpatients"
DEFAULT_PTL_PATH = "data/raw/ptl"
DEFAULT_RTT_PATH = "data/raw/rtt"


def filter_month_range(
    df: pd.DataFrame,
    month_col: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    if df.empty or month_col not in df.columns:
        return df

    start_month = pd.to_datetime(start_date).to_period("M").to_timestamp()
    end_month = pd.to_datetime(end_date).to_period("M").to_timestamp()

    return df[
        (df[month_col] >= start_month)
        & (df[month_col] <= end_month)
    ].copy()


def get_attended_outpatients(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Status" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    status_clean = (
        df["Status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    attendance_mask = (
        status_clean.str.contains("checked in")
        | status_clean.str.contains("check in")
        | status_clean.str.contains("checked out")
        | status_clean.str.contains("check out")
    )

    return df[attendance_mask].copy()


def summarise_pathway_outpatients(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    output_cols = (
        group_cols
        + [
            "Outpatient First Attendances",
            "Outpatient Follow-Ups",
            "Follow-Ups per 1 First Attendance",
        ]
    )

    visit_type_col = (
        "ContactVisitType_Group"
        if "ContactVisitType_Group" in df.columns
        else "ContactVisitType"
    )

    required_cols = set(group_cols + ["Contact_ID", visit_type_col])
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=output_cols)

    attended = get_attended_outpatients(df)

    if attended.empty:
        return pd.DataFrame(columns=output_cols)

    attended[visit_type_col] = (
        attended[visit_type_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    base_df = attended.groupby(group_cols).size().reset_index()[group_cols]

    follow_up_df = (
        attended[attended[visit_type_col] == "Follow Up"]
        .groupby(group_cols)
        .agg(**{"Outpatient Follow-Ups": ("Contact_ID", "nunique")})
        .reset_index()
    )

    new_df = (
        attended[attended[visit_type_col] == "First attendance"]
        .groupby(group_cols)
        .agg(**{"Outpatient First Attendances": ("Contact_ID", "nunique")})
        .reset_index()
    )

    summary = base_df.merge(new_df, on=group_cols, how="left").merge(
        follow_up_df,
        on=group_cols,
        how="left",
    )

    for col in ["Outpatient First Attendances", "Outpatient Follow-Ups"]:
        summary[col] = summary[col].fillna(0)

    summary["Follow-Ups per 1 First Attendance"] = summary.apply(
        lambda row: row["Outpatient Follow-Ups"]
        / row["Outpatient First Attendances"]
        if row["Outpatient First Attendances"] > 0
        else pd.NA,
        axis=1,
    )

    return summary.sort_values(group_cols).reset_index(drop=True)


def prepare_new_follow_up_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ratio_df = df[
        [
            "Month",
            "Outpatient First Attendances",
            "Outpatient Follow-Ups",
            "Follow-Ups per 1 First Attendance",
        ]
    ].copy()

    def format_statement(row) -> str:
        first_attendances = row["Outpatient First Attendances"]
        follow_ups = row["Outpatient Follow-Ups"]
        ratio = row["Follow-Ups per 1 First Attendance"]

        if first_attendances <= 0 and follow_ups > 0:
            return "No first attendances coded"

        if first_attendances <= 0:
            return "No first or follow-up attendances coded"

        return (
            "For every 1 first attendance, "
            f"there are {ratio:.1f} follow-ups"
        )

    ratio_df["First to Follow-Up Interpretation"] = ratio_df.apply(
        format_statement,
        axis=1,
    )

    ratio_df["Follow-Ups per 1 First Attendance"] = ratio_df[
        "Follow-Ups per 1 First Attendance"
    ].round(1)

    ratio_df["Month"] = ratio_df["Month"].dt.strftime("%B %Y")

    return ratio_df


def summarise_pathway_inpatients(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    output_cols = group_cols + ["Elective Activity"]

    if df.empty or not set(group_cols + ["Spell ID"]).issubset(df.columns):
        return pd.DataFrame(columns=output_cols)

    return (
        df.groupby(group_cols)
        .agg(**{"Elective Activity": ("Spell ID", "nunique")})
        .reset_index()
        .sort_values(group_cols)
    )


def get_rtt_file_signature(
    folder_path: str,
) -> tuple[tuple[str, int, int], ...]:
    folder = Path(folder_path)

    if not folder.exists():
        return tuple()

    return tuple(
        sorted(
            (
                file_path.name,
                file_path.stat().st_mtime_ns,
                file_path.stat().st_size,
            )
            for file_path in folder.glob("*.csv")
        )
    )


@st.cache_data(show_spinner=False)
def load_rtt_ptl_by_month_specialty(
    folder_path: str = DEFAULT_RTT_PATH,
    file_signature: tuple[tuple[str, int, int], ...] = tuple(),
) -> pd.DataFrame:
    _ = file_signature
    folder = Path(folder_path)
    within_18_week_cols = [
        f"Gt {week:02d} To {week + 1:02d} Weeks SUM 1"
        for week in range(18)
    ]
    usecols = [
        "Provider Org Code",
        "Provider Org Name",
        "RTT Part Description",
        "Treatment Function Name",
        "Total All",
    ] + within_18_week_cols

    dfs = []

    for file_path in sorted(folder.glob("*.csv")):
        parts = file_path.stem.split("-")
        if len(parts) >= 4:
            month_date = pd.to_datetime(
                f"{parts[2]} {parts[3]}",
                format="%B %Y",
                errors="coerce",
            )
        else:
            month_date = pd.NaT

        df = pd.read_csv(file_path, usecols=usecols)

        df = df[
            (df["Provider Org Code"].astype(str).str.strip().str.upper() == "RQW")
            & (df["RTT Part Description"] == "Incomplete Pathways")
        ].copy()

        df = df[
            ~df["Treatment Function Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["total"])
        ]

        df["Month"] = month_date
        df["Standardised_Specialty"] = standardise_specialty_series(
            df["Treatment Function Name"]
        )
        df["Total All"] = pd.to_numeric(df["Total All"], errors="coerce").fillna(0)
        df[within_18_week_cols] = df[within_18_week_cols].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)
        df["RTT Within 18 Weeks"] = df[within_18_week_cols].sum(axis=1)

        dfs.append(
            df[
                [
                    "Month",
                    "Standardised_Specialty",
                    "Total All",
                    "RTT Within 18 Weeks",
                ]
            ]
        )

    if not dfs:
        return pd.DataFrame(
            columns=[
                "Month",
                "Standardised_Specialty",
                "RTT Incomplete Pathways",
                "RTT Within 18 Weeks",
                "RTT % Within 18 Weeks",
            ]
        )

    specialty_rtt_df = (
        pd.concat(dfs, ignore_index=True)
        .groupby(["Month", "Standardised_Specialty"], as_index=False)
        .agg(
            **{
                "RTT Incomplete Pathways": ("Total All", "sum"),
                "RTT Within 18 Weeks": ("RTT Within 18 Weeks", "sum"),
            }
        )
        .sort_values(["Month", "Standardised_Specialty"])
    )
    specialty_rtt_df["RTT % Within 18 Weeks"] = specialty_rtt_df[
        "RTT Within 18 Weeks"
    ].div(
        specialty_rtt_df["RTT Incomplete Pathways"].where(
            specialty_rtt_df["RTT Incomplete Pathways"] > 0
        )
    )

    return specialty_rtt_df


def plot_pathway_volumes(
    df: pd.DataFrame,
    activity_cols: list[str],
    rtt_col: str,
    title: str,
):
    rtt_styles = {
        "RTT PTL Size": {
            "color": "#4B5563",
            "dash": "dash",
            "symbol": "diamond",
        },
        "RTT Incomplete Pathways": {
            "color": "#6B7280",
            "dash": "dot",
            "symbol": "square",
        },
    }

    fig = go.Figure()

    for col in activity_cols:
        if col not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df[col],
                mode="lines+markers",
                name=col,
                yaxis="y1",
                hovertemplate=(
                    "Month: %{x|%b %Y}"
                    f"<br>{col}: " + "%{y:,.0f}<extra></extra>"
                ),
            )
        )

    if rtt_col in df.columns:
        rtt_style = rtt_styles.get(
            rtt_col,
            {
                "color": "#4B5563",
                "dash": "dash",
                "symbol": "diamond",
            },
        )

        fig.add_trace(
            go.Scatter(
                x=df["Month"],
                y=df[rtt_col],
                mode="lines+markers",
                name=rtt_col,
                yaxis="y2",
                line=dict(
                    color=rtt_style["color"],
                    width=4,
                    dash=rtt_style["dash"],
                ),
                marker=dict(
                    color=rtt_style["color"],
                    size=9,
                    symbol=rtt_style["symbol"],
                    line=dict(color="white", width=1),
                ),
                connectgaps=False,
                hovertemplate=(
                    "Month: %{x|%b %Y}"
                    f"<br>{rtt_col}: " + "%{y:,.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Month",
        yaxis=dict(
            title="Activity volumes",
            side="left",
        ),
        yaxis2=dict(
            title=rtt_col,
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        height=640,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ---------------------------------------------------------
# Load outpatient data
# ---------------------------------------------------------
try:
    outpatient_df = load_outpatient_data(DEFAULT_OUTPATIENT_PATH)

except Exception as e:
    st.error(f"Could not load outpatient data: {e}")
    st.info(
        "Check that your outpatient files are saved in: "
        "`data/raw/Outpatients` and are CSV or Excel files."
    )
    st.stop()


warnings = validate_outpatient_data(outpatient_df)

for warning in warnings:
    st.warning(warning)


# ---------------------------------------------------------
# Load PTL data
# ---------------------------------------------------------
ptl_available = False
ptl_error = None
ptl_monthly_df = pd.DataFrame()

try:
    ptl_df = load_ptl_data(DEFAULT_PTL_PATH)
    ptl_monthly_df = summarise_ptl_by_month(ptl_df)

    ptl_monthly_df = ptl_monthly_df.rename(
        columns={
            "PTL_Month": "Contact_Month",
            "PTL Size": "PTL Waiting List Size",
        }
    )

    ptl_available = True

except Exception as e:
    ptl_available = False
    ptl_error = str(e)


# ---------------------------------------------------------
# Load inpatient data
# ---------------------------------------------------------
inpatient_available = False
inpatient_error = None
inpatient_df = pd.DataFrame()

try:
    inpatient_df = load_inpatient_data("data/raw/Inpatient/Inpatients.csv")
    inpatient_available = True

except Exception as e:
    inpatient_available = False
    inpatient_error = str(e)


# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------
with st.expander("How to interpret this page", expanded=False):
    st.markdown(
        """
This page shows outpatient contacts, which sit between referral demand and downstream PTL / RTT pressure.

Pathway view:

Referral received  
↓  
Outpatient contact  
↓  
Diagnostics / decision / treatment planning  
↓  
PTL / RTT pathway pressure

Each row in the outpatient dataset is treated as one outpatient contact, identified by `Contact_ID`.

This page helps answer:

- How much outpatient activity is being delivered?
- Which specialties are absorbing the most activity?
- Which clinics or performance units are carrying the greatest load?
- What is the status mix of outpatient contacts?
- Are there signs of outpatient flow pressure?
- Is checked-out activity keeping pace with checked-in activity?
- Does the outpatient flow gap align with PTL waiting list movement?
        """
    )


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
with st.sidebar:
    st.header("Outpatient Filters")

    min_date = outpatient_df["Contact_Start"].min().date()
    max_date = outpatient_df["Contact_Start"].max().date()

    selected_dates = st.date_input(
        "Contact start date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    specialties = sorted(
        outpatient_df["Standardised_Specialty"].dropna().unique()
    )

    selected_specialties = st.multiselect(
        "Specialty",
        specialties,
        default=[],
    )

    clinics = (
        sorted(outpatient_df["ContactClinicPerfUnit"].dropna().unique())
        if "ContactClinicPerfUnit" in outpatient_df.columns
        else []
    )

    selected_clinics = st.multiselect(
        "Clinic / performance unit",
        clinics,
        default=[],
    )

    clinic_types = (
        sorted(outpatient_df["ContactClinicPerfUnit_Type"].dropna().unique())
        if "ContactClinicPerfUnit_Type" in outpatient_df.columns
        else []
    )

    selected_clinic_types = st.multiselect(
        "Clinic / unit type",
        clinic_types,
        default=[],
    )

    contact_types = (
        sorted(outpatient_df["Type"].dropna().unique())
        if "Type" in outpatient_df.columns
        else []
    )

    selected_contact_types = st.multiselect(
        "Contact type",
        contact_types,
        default=[],
    )

    statuses = (
        sorted(outpatient_df["Status"].dropna().unique())
        if "Status" in outpatient_df.columns
        else []
    )

    selected_statuses = st.multiselect(
        "Status",
        statuses,
        default=[],
    )

    st.header("Inpatient Filters")

    patient_classifications = (
        sorted(inpatient_df["Patient classification"].dropna().unique())
        if inpatient_available
        and "Patient classification" in inpatient_df.columns
        else []
    )

    selected_patient_classifications = st.multiselect(
        "Patient classification",
        patient_classifications,
        default=[],
    )

    st.markdown("---")
    st.caption(f"Outpatients loaded from: `{DEFAULT_OUTPATIENT_PATH}`")
    st.caption(f"PTL loaded from: `{DEFAULT_PTL_PATH}`")

    if not inpatient_available:
        st.caption("Inpatient data unavailable")

    if "Source_File" in outpatient_df.columns:
        st.caption(f"Outpatient files loaded: {outpatient_df['Source_File'].nunique()}")


# ---------------------------------------------------------
# Apply filters
# ---------------------------------------------------------
filtered_df = filter_outpatients(
    outpatient_df,
    start_date=start_date,
    end_date=end_date,
    specialties=selected_specialties,
    clinics=selected_clinics,
    clinic_types=selected_clinic_types,
    contact_types=selected_contact_types,
    statuses=selected_statuses,
)


# ---------------------------------------------------------
# Core summaries
# ---------------------------------------------------------
monthly_df = summarise_outpatients_by_month(filtered_df)
monthly_growth_df = add_monthly_contact_growth(monthly_df)
growth_signal = outpatient_growth_signal(monthly_df, baseline_months=6)

total_contacts = filtered_df["Contact_ID"].nunique()
specialty_count = filtered_df["Standardised_Specialty"].nunique()

clinic_count = (
    filtered_df["ContactClinicPerfUnit"].nunique()
    if "ContactClinicPerfUnit" in filtered_df.columns
    else 0
)

months_count = filtered_df["Contact_Month"].nunique()


# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total outpatient contacts", f"{total_contacts:,.0f}")
kpi2.metric("Specialties", f"{specialty_count:,.0f}")
kpi3.metric("Clinics / units", f"{clinic_count:,.0f}")
kpi4.metric(
    "Latest month contacts",
    f"{growth_signal['latest']:,.0f}",
    delta=f"{growth_signal['change_pct']}% vs baseline",
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab_pathway_flow, tab7 = st.tabs(
    [
        "Outpatient Overview",
        "Specialty Flow",
        "Clinic Capacity",
        "Contact Status",
        "Flow Heatmap",
        "Flow Balance vs PTL",
        "Pathway Flow",
        "Pathway Interpretation",
    ]
)


# ---------------------------------------------------------
# Tab 1: Outpatient Overview
# ---------------------------------------------------------
with tab1:
    st.subheader("Monthly outpatient activity")

    if monthly_df.empty:
        st.info("No outpatient data available for the selected filters.")
    else:
        fig = px.line(
            monthly_df,
            x="Contact_Month",
            y="Contacts",
            markers=True,
            title="Monthly outpatient contacts",
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Contacts",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
Interpretation: this chart shows outpatient activity delivered over time.
Rising outpatient contacts may reflect increased pathway activity, additional capacity,
follow-up pressure, or increased demand flowing through from referrals.
            """
        )

        st.subheader("Monthly outpatient table")
        st.dataframe(monthly_growth_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 2: Specialty Flow
# ---------------------------------------------------------
with tab2:
    st.subheader("Outpatient activity by specialty")

    specialty_df = summarise_outpatients_by_specialty(filtered_df)

    if specialty_df.empty:
        st.info("No specialty outpatient data available.")
    else:
        top_n = st.slider(
            "Number of specialties to show",
            min_value=5,
            max_value=30,
            value=15,
            key="outpatient_specialty_top_n",
        )

        fig = px.bar(
            specialty_df.head(top_n),
            x="Contacts",
            y="Standardised_Specialty",
            orientation="h",
            title="Top specialties by outpatient contacts",
        )

        fig.update_layout(
            xaxis_title="Contacts",
            yaxis_title="Specialty",
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This view shows which specialties are absorbing the greatest outpatient activity."
        )

        st.subheader("Specialty outpatient table")
        st.dataframe(specialty_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 3: Clinic Capacity
# ---------------------------------------------------------
with tab3:
    st.subheader("Clinic / performance unit activity")

    clinic_df = summarise_outpatients_by_clinic(filtered_df)
    clinic_type_df = summarise_outpatients_by_clinic_type(filtered_df)

    clinic_col, type_col = st.columns(2)

    with clinic_col:
        st.markdown("### Busiest clinics / units")

        if clinic_df.empty:
            st.info("No clinic data available.")
        else:
            fig = px.bar(
                clinic_df.head(20),
                x="Contacts",
                y="ContactClinicPerfUnit",
                orientation="h",
                title="Top clinics / performance units",
            )

            fig.update_layout(
                xaxis_title="Contacts",
                yaxis_title="Clinic / Unit",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(clinic_df, use_container_width=True)

    with type_col:
        st.markdown("### Clinic / unit type")

        if clinic_type_df.empty:
            st.info("No clinic type data available.")
        else:
            fig = px.pie(
                clinic_type_df,
                names="ContactClinicPerfUnit_Type",
                values="Contacts",
                title="Clinic / unit type mix",
                hole=0.35,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(clinic_type_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 4: Contact Status
# ---------------------------------------------------------
with tab4:
    st.subheader("Outpatient contact status profile")

    status_df = summarise_outpatients_by_status(filtered_df)
    type_df = summarise_outpatients_by_type(filtered_df)
    visit_type_df = summarise_outpatients_by_visit_type(filtered_df)

    status_col, type_col, visit_type_col = st.columns(3)

    with status_col:
        st.markdown("### Status mix")

        if status_df.empty:
            st.info("No status data available.")
        else:
            fig = px.bar(
                status_df,
                x="Status",
                y="Contacts",
                title="Outpatient contact status",
            )

            fig.update_layout(
                xaxis_title="Status",
                yaxis_title="Contacts",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(status_df, use_container_width=True)

    with type_col:
        st.markdown("### Contact type")

        if type_df.empty:
            st.info("No contact type data available.")
        else:
            fig = px.pie(
                type_df,
                names="Type",
                values="Contacts",
                title="Outpatient contact type mix",
                hole=0.35,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(type_df, use_container_width=True)

    with visit_type_col:
        st.markdown("### Contact visit type")

        if visit_type_df.empty:
            st.info("No contact visit type data available.")
        else:
            fig = px.pie(
                visit_type_df,
                names="ContactVisitType",
                values="Contacts",
                title="Contact visit type mix",
                hole=0.35,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(visit_type_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 5: Flow Heatmap
# ---------------------------------------------------------
with tab5:
    st.subheader("Specialty outpatient flow heatmap")

    st.markdown(
        """
This heatmap shows outpatient contact intensity by specialty over time.

Each row represents a specialty and each column represents a month.
Red indicates higher outpatient contact volume and blue indicates lower contact volume.
        """
    )

    heatmap_top_n = st.slider(
        "Number of specialties to include in heatmap",
        min_value=5,
        max_value=30,
        value=20,
        key="outpatient_heatmap_top_n",
    )

    heatmap_matrix = outpatient_heatmap_matrix(
        filtered_df,
        top_n=heatmap_top_n,
    )

    if heatmap_matrix.empty:
        st.info("Not enough data to create the outpatient heatmap.")
    else:
        fig = px.imshow(
            heatmap_matrix,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(
                x="Month",
                y="Specialty",
                color="Contacts",
            ),
            title="Specialty outpatient activity heatmap",
        )

        fig.update_layout(
            xaxis_title="Contact month",
            yaxis_title="Specialty",
            height=700,
            coloraxis_colorbar=dict(title="Contact Volume"),
        )

        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
Interpretation: persistent red cells suggest sustained outpatient flow pressure.
A specialty becoming redder over recent months may indicate increasing outpatient activity,
which may be caused by rising referrals, additional capacity, follow-up demand, or operational change.
            """
        )

        st.subheader("Heatmap data table")
        st.dataframe(heatmap_matrix, use_container_width=True)


# ---------------------------------------------------------
# Tab 6: Flow Balance vs PTL
# ---------------------------------------------------------
with tab6:
    st.subheader("Checked-in vs checked-out activity with PTL waiting list size")

    st.markdown(
        """
This view compares outpatient flow activity with PTL waiting list size over time.

It helps show whether the gap between checked-in and checked-out outpatient activity aligns with changes in the PTL waiting list size.
        """
    )

    flow_df = summarise_checked_flow_by_month(filtered_df)

    if flow_df.empty:
        st.info(
            "No checked-in or checked-out outpatient activity found for the selected filters."
        )

    elif not ptl_available:
        st.warning(f"PTL data could not be loaded: {ptl_error}")

    else:
        combined_df = flow_df.merge(
            ptl_monthly_df,
            on="Contact_Month",
            how="left",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=combined_df["Contact_Month"],
                y=combined_df["Checked In"],
                mode="lines+markers",
                name="Checked In Activity",
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=combined_df["Contact_Month"],
                y=combined_df["Checked Out"],
                mode="lines+markers",
                name="Checked Out Activity",
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=combined_df["Contact_Month"],
                y=combined_df["PTL Waiting List Size"],
                mode="lines+markers",
                name="PTL Waiting List Size",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Outpatient Flow Balance vs PTL Waiting List Size",
            xaxis=dict(title="Month"),
            yaxis=dict(
                title="Outpatient Contacts",
                side="left",
            ),
            yaxis2=dict(
                title="PTL Waiting List Size",
                overlaying="y",
                side="right",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            height=650,
            margin=dict(l=20, r=20, t=80, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
Interpretation:

- Checked-in activity represents outpatient contacts entering / arriving into clinic flow.
- Checked-out activity represents outpatient contacts leaving / completing clinic flow.
- If checked-in activity is consistently higher than checked-out activity, outpatient pressure may be accumulating.
- If checked-out activity is higher than checked-in activity, outpatient flow may be recovering.
- If PTL size rises while checked-out activity is flat or falling, this may suggest outpatient flow is not keeping pace with pathway demand.
- This does not prove causation, but it gives a useful pathway flow signal.
            """
        )

        st.subheader("Flow balance table")

        display_df = combined_df.copy()
        display_df["Contact_Month"] = display_df["Contact_Month"].dt.strftime("%B %Y")

        st.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------
# Pathway Flow tab
# ---------------------------------------------------------
with tab_pathway_flow:
    st.subheader("Pathway Flow: Attendances, Follow-Ups, Inpatients and RTT PTL")

    if inpatient_available:
        inpatient_filtered_df = filter_inpatients(
            inpatient_df,
            start_date=start_date,
            end_date=end_date,
            specialties=selected_specialties,
            patient_classifications=selected_patient_classifications,
        )

    else:
        inpatient_filtered_df = pd.DataFrame()

    outpatient_pathway_df = filtered_df.rename(
        columns={"Contact_Month": "Month"}
    ).copy()

    inpatient_pathway_df = inpatient_filtered_df.rename(
        columns={"Admission_Month": "Month"}
    ).copy()

    ptl_for_merge = ptl_monthly_df.rename(
        columns={
            "Contact_Month": "Month",
            "PTL Waiting List Size": "RTT PTL Size",
        }
    )

    ptl_for_merge = filter_month_range(
        ptl_for_merge,
        "Month",
        start_date,
        end_date,
    )

    if not ptl_available:
        st.warning(f"PTL data could not be loaded: {ptl_error}")

    elif not inpatient_available:
        st.warning(f"Inpatient data could not be loaded: {inpatient_error}")

    else:
        overall_outpatient_df = summarise_pathway_outpatients(
            outpatient_pathway_df,
            group_cols=["Month"],
        )

        overall_inpatient_df = summarise_pathway_inpatients(
            inpatient_pathway_df,
            group_cols=["Month"],
        )

        overall_flow_df = overall_outpatient_df.merge(
            overall_inpatient_df,
            on="Month",
            how="outer",
        ).merge(
            ptl_for_merge,
            on="Month",
            how="outer",
        )

        activity_cols = [
            "Outpatient First Attendances",
            "Outpatient Follow-Ups",
            "Elective Activity",
        ]

        table_cols = [
            "Month",
            "Outpatient First Attendances",
            "Outpatient Follow-Ups",
            "Follow-Ups per 1 First Attendance",
            "Elective Activity",
            "RTT PTL Size",
        ]

        if overall_flow_df.empty:
            st.info("No pathway flow data available for the selected filters.")
        else:
            overall_flow_df = overall_flow_df.sort_values("Month")

            for col in table_cols:
                if col != "Month" and col not in overall_flow_df.columns:
                    overall_flow_df[col] = 0

            numeric_cols = [
                "Outpatient First Attendances",
                "Outpatient Follow-Ups",
                "Elective Activity",
                "RTT PTL Size",
            ]

            overall_flow_df[numeric_cols] = overall_flow_df[
                numeric_cols
            ].apply(pd.to_numeric, errors="coerce").fillna(0)

            st.markdown("### Overall")

            st.plotly_chart(
                plot_pathway_volumes(
                    overall_flow_df,
                    activity_cols=activity_cols,
                    rtt_col="RTT PTL Size",
                    title="Overall Pathway Flow Volumes",
                ),
                use_container_width=True,
            )

            st.markdown("#### First Attendance to Follow-Up Ratio")
            st.dataframe(
                prepare_new_follow_up_ratio_table(overall_flow_df),
                use_container_width=True,
            )

            overall_display_df = overall_flow_df[
                [
                    "Month",
                    "Outpatient First Attendances",
                    "Outpatient Follow-Ups",
                    "Elective Activity",
                    "RTT PTL Size",
                ]
            ].copy()
            overall_display_df["Month"] = overall_display_df["Month"].dt.strftime(
                "%B %Y"
            )

            st.dataframe(overall_display_df, use_container_width=True)

        st.markdown("### By Specialty")
        st.caption(
            "Specialty RTT line uses PAH incomplete RTT pathways from the monthly RTT extracts. "
            "The aggregate PTL CSV only contains total monthly size and has no specialty field."
        )

        try:
            rtt_specialty_ptl_df = load_rtt_ptl_by_month_specialty(
                DEFAULT_RTT_PATH,
                get_rtt_file_signature(DEFAULT_RTT_PATH),
            )
            rtt_specialty_ptl_df = filter_month_range(
                rtt_specialty_ptl_df,
                "Month",
                start_date,
                end_date,
            )
            rtt_specialty_available = True
            rtt_specialty_error = None
        except Exception as e:
            rtt_specialty_ptl_df = pd.DataFrame()
            rtt_specialty_available = False
            rtt_specialty_error = str(e)

        specialty_outpatient_df = summarise_pathway_outpatients(
            outpatient_pathway_df,
            group_cols=["Month", "Standardised_Specialty"],
        )

        specialty_inpatient_df = summarise_pathway_inpatients(
            inpatient_pathway_df,
            group_cols=["Month", "Standardised_Specialty"],
        )

        specialty_options = sorted(
            set(specialty_outpatient_df.get("Standardised_Specialty", []))
            | set(specialty_inpatient_df.get("Standardised_Specialty", []))
            | set(rtt_specialty_ptl_df.get("Standardised_Specialty", []))
        )

        if selected_specialties:
            specialty_options = [
                specialty
                for specialty in specialty_options
                if specialty in selected_specialties
            ]

        if not specialty_options:
            st.info("No specialty pathway flow data available.")
        else:
            default_specialty_index = 0

            if selected_specialties:
                default_specialty_index = specialty_options.index(
                    selected_specialties[0]
                )

            selected_pathway_specialty = st.selectbox(
                "Specialty",
                specialty_options,
                index=default_specialty_index,
                key="pathway_flow_specialty",
            )

            selected_outpatient_specialty_df = specialty_outpatient_df[
                specialty_outpatient_df["Standardised_Specialty"]
                == selected_pathway_specialty
            ]

            selected_inpatient_specialty_df = specialty_inpatient_df[
                specialty_inpatient_df["Standardised_Specialty"]
                == selected_pathway_specialty
            ]

            selected_rtt_specialty_df = rtt_specialty_ptl_df[
                rtt_specialty_ptl_df["Standardised_Specialty"]
                == selected_pathway_specialty
            ]

            specialty_flow_df = selected_outpatient_specialty_df.merge(
                selected_inpatient_specialty_df,
                on=["Month", "Standardised_Specialty"],
                how="outer",
            ).merge(
                selected_rtt_specialty_df,
                on=["Month", "Standardised_Specialty"],
                how="outer",
            )

            if specialty_flow_df.empty:
                st.info("No data available for the selected specialty.")
            else:
                specialty_flow_df = specialty_flow_df.sort_values("Month")

                for col in table_cols:
                    if col != "Month" and col not in specialty_flow_df.columns:
                        specialty_flow_df[col] = 0

                if "RTT Incomplete Pathways" not in specialty_flow_df.columns:
                    specialty_flow_df["RTT Incomplete Pathways"] = pd.NA
                if "RTT Within 18 Weeks" not in specialty_flow_df.columns:
                    specialty_flow_df["RTT Within 18 Weeks"] = pd.NA
                if "RTT % Within 18 Weeks" not in specialty_flow_df.columns:
                    specialty_flow_df["RTT % Within 18 Weeks"] = pd.NA

                specialty_numeric_cols = [
                    "Outpatient First Attendances",
                    "Outpatient Follow-Ups",
                    "Elective Activity",
                ]

                specialty_flow_df[specialty_numeric_cols] = specialty_flow_df[
                    specialty_numeric_cols
                ].apply(pd.to_numeric, errors="coerce").fillna(0)

                specialty_flow_df["RTT Incomplete Pathways"] = pd.to_numeric(
                    specialty_flow_df["RTT Incomplete Pathways"],
                    errors="coerce",
                )
                specialty_flow_df["RTT Within 18 Weeks"] = pd.to_numeric(
                    specialty_flow_df["RTT Within 18 Weeks"],
                    errors="coerce",
                )
                specialty_flow_df["RTT % Within 18 Weeks"] = pd.to_numeric(
                    specialty_flow_df["RTT % Within 18 Weeks"],
                    errors="coerce",
                )

                st.plotly_chart(
                    plot_pathway_volumes(
                        specialty_flow_df,
                        activity_cols=activity_cols,
                        rtt_col="RTT Incomplete Pathways",
                        title=f"{selected_pathway_specialty} Pathway Flow Volumes",
                    ),
                    use_container_width=True,
                )

                rtt_performance_df = specialty_flow_df.dropna(
                    subset=["RTT % Within 18 Weeks"]
                ).copy()
                if not rtt_performance_df.empty:
                    fig_rtt_performance = go.Figure()
                    fig_rtt_performance.add_trace(
                        go.Scatter(
                            x=rtt_performance_df["Month"],
                            y=rtt_performance_df["RTT % Within 18 Weeks"],
                            mode="lines+markers",
                            name="RTT % Within 18 Weeks",
                            line=dict(color="#2563EB", width=4),
                            marker=dict(size=9),
                            hovertemplate=(
                                "Month: %{x|%b %Y}"
                                "<br>% Within 18 Weeks: %{y:.1%}"
                                "<extra></extra>"
                            ),
                        )
                    )
                    fig_rtt_performance.add_hline(
                        y=0.92,
                        line_dash="dash",
                        line_color="#991B1B",
                        annotation_text="92% standard",
                        annotation_position="top right",
                    )
                    fig_rtt_performance.update_layout(
                        template="plotly_white",
                        title=(
                            f"{selected_pathway_specialty} RTT 18-Week "
                            "Performance"
                        ),
                        xaxis_title="Month",
                        yaxis_title="% within 18 weeks",
                        yaxis_tickformat=".0%",
                        height=440,
                        margin=dict(l=20, r=20, t=70, b=20),
                    )
                    st.plotly_chart(fig_rtt_performance, use_container_width=True)

                st.markdown("#### First Attendance to Follow-Up Ratio")
                st.dataframe(
                    prepare_new_follow_up_ratio_table(specialty_flow_df),
                    use_container_width=True,
                )

                specialty_display_df = specialty_flow_df[
                    [
                        "Standardised_Specialty",
                        "Month",
                        "Outpatient First Attendances",
                        "Outpatient Follow-Ups",
                        "Elective Activity",
                        "RTT Incomplete Pathways",
                        "RTT Within 18 Weeks",
                        "RTT % Within 18 Weeks",
                    ]
                ].copy()
                specialty_display_df["Month"] = specialty_display_df[
                    "Month"
                ].dt.strftime("%B %Y")
                specialty_display_df["RTT % Within 18 Weeks"] = (
                    specialty_display_df["RTT % Within 18 Weeks"] * 100
                ).round(1)

                st.dataframe(specialty_display_df, use_container_width=True)

        if not rtt_specialty_available:
            st.warning(
                "Specialty-level RTT PTL size could not be loaded: "
                f"{rtt_specialty_error}"
            )
