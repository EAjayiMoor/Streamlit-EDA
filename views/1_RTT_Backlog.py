import streamlit as st
import plotly.express as px
import pandas as pd

from src.data.rtt_loader import load_all_rtt_files
from src.transforms.rtt_transform import (
    add_wait_band_metrics,
    filter_pah_admitted_backlog,
    filter_pah_incomplete,
    filter_rtt_admitted_backlog,
    filter_rtt_incomplete,
    summarise_admitted_backlog_by_month,
    summarise_admitted_backlog_by_month_specialty,
    summarise_rtt_by_month,
    summarise_rtt_by_month_specialty,
    summarise_weekly_wait_band_distribution,
)
from src.utils.specialty_standardisation import standardise_specialty_series

SURGICAL_ADMITTED_BACKLOG_SPECIALTIES = [
    "Ophthalmology",
    "Trauma & Orthopaedics",
    "General Surgery",
    "Urology",
    "Gynaecology",
    "ENT",
    "Oral Surgery",
]

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.title("RTT Backlog")
st.write(
    """
    This view shows the active RTT waiting list for the selected organisation,
    based on incomplete pathways. It tracks backlog size, performance against
    the 18-week standard, and backlog severity through 52+ week waits.
    """
)

# ---------------------------------------------------------
# Load and transform RTT data
# ---------------------------------------------------------
# Why:
# - load all monthly RTT files from the raw data folder
# - filter to PAH and incomplete pathways only
# - derive 0-18 and 52+ wait-band metrics
# - summarise to one row per month for trend analysis
try:
    uploaded_frames = st.session_state.get("eda_frames", [])
    uploaded_workflow = st.session_state.get("eda_workflow")
    if uploaded_workflow == "rtt" and uploaded_frames:
        raw_df = pd.concat(uploaded_frames, ignore_index=True)
        st.info("Using the validated RTT batch from the upload workspace.")
    else:
        raw_df = load_all_rtt_files()

    provider_options = []
    if "Provider Org Name" in raw_df.columns:
        provider_options = sorted(
            raw_df["Provider Org Name"].dropna().astype(str).str.strip().unique()
        )
    if not provider_options:
        raise ValueError("No provider organisations were found in the RTT data.")

    selected_provider = st.sidebar.selectbox(
        "Organisation",
        provider_options,
        index=0,
        help="Select the hospital or provider to analyse.",
    )
    pah_df = filter_rtt_incomplete(raw_df, selected_provider)
    admitted_df = filter_rtt_admitted_backlog(raw_df, selected_provider)
    metric_df = add_wait_band_metrics(pah_df)
    admitted_metric_df = add_wait_band_metrics(admitted_df)
    summary_df = summarise_rtt_by_month(metric_df)
    specialty_df = summarise_rtt_by_month_specialty(metric_df)
    admitted_summary_df = summarise_admitted_backlog_by_month(admitted_metric_df)
    admitted_specialty_df = summarise_admitted_backlog_by_month_specialty(
        admitted_metric_df
    )
    wait_band_df = summarise_weekly_wait_band_distribution(pah_df)


except Exception as e:
    st.error(f"Error loading or transforming RTT data: {e}")
    st.stop()

# ---------------------------------------------------------
# Defensive checks
# ---------------------------------------------------------
if summary_df.empty:
    st.warning("No RTT summary data available after transformation.")
    st.stop()

# ---------------------------------------------------------
# Month ordering
# ---------------------------------------------------------
# Why:
# - Month labels are text (e.g. 'January 2026')
# - we create a real datetime field for correct chronological sorting
summary_df = summary_df.copy()
summary_df["Month_Date"] = summary_df["Month"].astype("string")

try:
    summary_df["Month_Date"] = summary_df["Month_Date"].apply(
        lambda x: __import__("datetime").datetime.strptime(x, "%B %Y")
    )
except Exception:
    # Fallback: if parsing fails, keep current order
    summary_df["Month_Date"] = range(len(summary_df))

summary_df = summary_df.sort_values("Month_Date").reset_index(drop=True)
summary_df = summary_df.merge(admitted_summary_df, on="Month", how="left")

for admitted_col in [
    "admitted_backlog",
    "admitted_0_18_total",
    "admitted_52_plus_total",
]:
    summary_df[admitted_col] = (
        summary_df[admitted_col].fillna(0).astype(float)
    )

summary_df["admitted_backlog_pct"] = summary_df["admitted_backlog"].div(
    summary_df["Total"].where(summary_df["Total"] > 0)
)

specialty_df = specialty_df.copy()
specialty_df["Month_Date"] = specialty_df["Month"].astype("string")

try:
    specialty_df["Month_Date"] = specialty_df["Month_Date"].apply(
        lambda x: __import__("datetime").datetime.strptime(x, "%B %Y")
    )
except Exception:
    specialty_df["Month_Date"] = range(len(specialty_df))

specialty_df["Specialty"] = standardise_specialty_series(
    specialty_df["Treatment Function Name"]
)
specialty_df = specialty_df.merge(
    admitted_specialty_df[
        [
            "Month",
            "Treatment Function Code",
            "Treatment Function Name",
            "admitted_backlog",
            "admitted_0_18_total",
            "admitted_52_plus_total",
        ]
    ],
    on=["Month", "Treatment Function Code", "Treatment Function Name"],
    how="left",
)

for admitted_col in [
    "admitted_backlog",
    "admitted_0_18_total",
    "admitted_52_plus_total",
]:
    specialty_df[admitted_col] = (
        specialty_df[admitted_col].fillna(0).astype(float)
    )

specialty_df["admitted_backlog_pct"] = specialty_df["admitted_backlog"].div(
    specialty_df["Total"].where(specialty_df["Total"] > 0)
)

specialty_df = specialty_df.sort_values(
    ["Month_Date", "Specialty"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Sidebar filter
# ---------------------------------------------------------
st.sidebar.header("Filters")

available_months = summary_df["Month"].tolist()

selected_months = st.sidebar.multiselect(
    "Select months",
    options=available_months,
    default=available_months,
)

filtered_df = summary_df[summary_df["Month"].isin(selected_months)].copy()

# Re-sort after filtering
if "Month_Date" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("Month_Date").reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data available for the selected month range.")
    st.stop()

# ---------------------------------------------------------
# Latest KPI values
# ---------------------------------------------------------
# Why:
# - KPIs should reflect the latest visible month after filtering
latest = filtered_df.iloc[-1]

latest_total = int(latest["Total"])
latest_18_total = int(latest["waiting_0_18_total"])
latest_52_total = int(latest["waiting_52_plus_total"])
latest_admitted_total = int(latest["admitted_backlog"])
latest_pct_18 = float(latest["pct_0_18"])
latest_pct_52 = float(latest["pct_52_plus"])
latest_admitted_pct = float(latest["admitted_backlog_pct"])

# Previous month values for delta, if available
if len(filtered_df) > 1:
    previous = filtered_df.iloc[-2]

    delta_total = latest_total - int(previous["Total"])
    delta_admitted_total = latest_admitted_total - int(previous["admitted_backlog"])
    delta_pct_18 = latest_pct_18 - float(previous["pct_0_18"])
    delta_pct_52 = latest_pct_52 - float(previous["pct_52_plus"])
else:
    delta_total = None
    delta_admitted_total = None
    delta_pct_18 = None
    delta_pct_52 = None

# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
st.subheader("Latest Position")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Waiting List Total",
        value=f"{latest_total:,}",
        delta=f"{delta_total:+,}" if delta_total is not None else None,
    )

with col2:
    st.metric(
        label="Admitted Backlog (DTA)",
        value=f"{latest_admitted_total:,}",
        delta=(
            f"{delta_admitted_total:+,}"
            if delta_admitted_total is not None
            else None
        ),
    )

with col3:
    st.metric(
        label="% Within 18 Weeks",
        value=f"{latest_pct_18:.1%}",
        delta=f"{delta_pct_18:+.2%}" if delta_pct_18 is not None else None,
    )

with col4:
    st.metric(
        label="0–18 Weeks Total",
        value=f"{latest_18_total:,}",
    )

with col5:
    st.metric(
        label="52+ Waits",
        value=f"{latest_52_total:,}",
        delta=f"{delta_pct_52:+.2%}" if delta_pct_52 is not None else None,
    )

# ---------------------------------------------------------
# Trend charts
# ---------------------------------------------------------
st.subheader("Trend Over Time")

# Waiting list trend
fig_total = px.line(
    filtered_df,
    x="Month",
    y="Total",
    markers=True,
    title="Waiting List Total Over Time",
)

fig_total.update_layout(
    xaxis_title="Month",
    yaxis_title="Patients",
)

st.plotly_chart(fig_total, use_container_width=True)

# Performance and severity trend
chart_df = filtered_df.copy()
chart_df["% Within 18 Weeks"] = chart_df["pct_0_18"]
chart_df["% 52+ Waits"] = chart_df["pct_52_plus"]

fig_pct = px.line(
    chart_df,
    x="Month",
    y=["% Within 18 Weeks", "% 52+ Waits"],
    markers=True,
    title="RTT Performance and Backlog Severity Over Time",
)

fig_pct.update_layout(
    xaxis_title="Month",
    yaxis_title="Percentage",
    yaxis_tickformat=".0%",
    legend_title_text="Metric",
)

st.plotly_chart(fig_pct, use_container_width=True)

# Optional volume chart for 0-18 vs 52+
volume_df = filtered_df.copy()
volume_df["0–18 Weeks"] = volume_df["waiting_0_18_total"]
volume_df["52+ Weeks"] = volume_df["waiting_52_plus_total"]

fig_volume = px.bar(
    volume_df,
    x="Month",
    y=["0–18 Weeks", "52+ Weeks"],
    barmode="group",
    title="0–18 Weeks and 52+ Weeks Volumes",
)

fig_volume.update_layout(
    xaxis_title="Month",
    yaxis_title="Patients",
    legend_title_text="Wait Band",
)

st.plotly_chart(fig_volume, use_container_width=True)

# ---------------------------------------------------------
# Specialty 18-week RTT performance
# ---------------------------------------------------------
st.subheader("RTT Performance by Specialty")
st.caption(
    "RTT performance is calculated as incomplete pathways waiting 0-18 weeks "
    "divided by total incomplete pathways for the selected specialty and month. "
    "Admitted backlog uses 'Incomplete Pathways with DTA' and is shown as a subset "
    "of the incomplete pathway backlog."
)

specialty_months = summary_df["Month"].tolist()
selected_specialty_month = st.selectbox(
    "Select month for specialty RTT performance",
    options=specialty_months,
    index=len(specialty_months) - 1 if specialty_months else 0,
)

min_specialty_backlog = st.slider(
    "Minimum specialty backlog to show",
    min_value=0,
    max_value=1000,
    value=25,
    step=25,
)

specialty_month_df = specialty_df[
    specialty_df["Month"] == selected_specialty_month
].copy()

selected_month_unfiltered_df = specialty_month_df.copy()
selected_month_total = int(selected_month_unfiltered_df["Total"].sum())
selected_month_admitted = int(
    selected_month_unfiltered_df["admitted_backlog"].sum()
)
selected_month_admitted_pct = (
    selected_month_admitted / selected_month_total
    if selected_month_total
    else 0
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(
    "All-Specialty RTT Incomplete Backlog",
    f"{selected_month_total:,}",
)
metric_col2.metric(
    "All-Specialty Admitted Backlog (DTA)",
    f"{selected_month_admitted:,}",
)
metric_col3.metric(
    "All-Specialty Admitted % of Incomplete",
    f"{selected_month_admitted_pct:.1%}",
)

surgical_cohort_df = selected_month_unfiltered_df[
    selected_month_unfiltered_df["Specialty"].isin(
        SURGICAL_ADMITTED_BACKLOG_SPECIALTIES
    )
].copy()

if not surgical_cohort_df.empty:
    surgical_total = int(surgical_cohort_df["Total"].sum())
    surgical_admitted = int(surgical_cohort_df["admitted_backlog"].sum())
    surgical_admitted_pct = surgical_admitted / surgical_total if surgical_total else 0

    st.caption(
        "Surgical cohort matches the 23k specialty backlog view: Ophthalmology, "
        "Trauma & Orthopaedics, General Surgery, Urology, Gynaecology, ENT, "
        "and Oral Surgery. Admitted backlog is the DTA subset waiting for "
        "admitted treatment."
    )

    surgical_metric_col1, surgical_metric_col2, surgical_metric_col3 = st.columns(3)
    surgical_metric_col1.metric(
        "Surgical RTT Cohort",
        f"{surgical_total:,}",
    )
    surgical_metric_col2.metric(
        "Surgical Admitted Backlog (DTA)",
        f"{surgical_admitted:,}",
    )
    surgical_metric_col3.metric(
        "Surgical Admitted % of Cohort",
        f"{surgical_admitted_pct:.1%}",
    )

    surgical_display_df = surgical_cohort_df[
        [
            "Specialty",
            "Total",
            "admitted_backlog",
            "admitted_backlog_pct",
        ]
    ].copy()
    surgical_display_df = surgical_display_df.sort_values(
        "admitted_backlog",
        ascending=False,
    ).rename(
        columns={
            "Total": "RTT Incomplete Pathways",
            "admitted_backlog": "Admitted Backlog (DTA)",
            "admitted_backlog_pct": "Admitted % of Specialty Backlog",
        }
    )
    surgical_display_df["Admitted % of Specialty Backlog"] = (
        surgical_display_df["Admitted % of Specialty Backlog"] * 100
    ).round(1)

    st.dataframe(surgical_display_df, use_container_width=True)

specialty_month_df = specialty_month_df[
    specialty_month_df["Total"] >= min_specialty_backlog
].copy()

if specialty_month_df.empty:
    st.info("No specialty RTT performance data available for the selected month.")
else:
    specialty_month_df = specialty_month_df.sort_values(
        ["pct_0_18", "Total"],
        ascending=[True, False],
    )

    chart_specialty_df = specialty_month_df.copy()
    chart_specialty_df["% Within 18 Weeks"] = chart_specialty_df["pct_0_18"]

    fig_specialty_perf = px.bar(
        chart_specialty_df,
        x="% Within 18 Weeks",
        y="Specialty",
        orientation="h",
        color="Total",
        color_continuous_scale="Blues",
        title=f"RTT 18-Week Performance by Specialty — {selected_specialty_month}",
        hover_data={
            "% Within 18 Weeks": ":.1%",
            "Total": ":,",
            "admitted_backlog": ":,",
            "admitted_backlog_pct": ":.1%",
            "waiting_0_18_total": ":,",
            "waiting_52_plus_total": ":,",
            "Specialty": True,
        },
    )

    fig_specialty_perf.add_vline(
        x=0.92,
        line_dash="dash",
        line_color="#991B1B",
        annotation_text="92% standard",
        annotation_position="top right",
    )

    fig_specialty_perf.update_layout(
        template="plotly_white",
        xaxis_title="% within 18 weeks",
        yaxis_title="Specialty",
        xaxis_tickformat=".0%",
        height=max(520, min(900, 34 * len(chart_specialty_df))),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_specialty_perf, use_container_width=True)

    admitted_chart_df = specialty_month_df[
        specialty_month_df["admitted_backlog"] > 0
    ].copy()

    if not admitted_chart_df.empty:
        admitted_chart_df = admitted_chart_df.sort_values(
            "admitted_backlog",
            ascending=True,
        )

        fig_admitted_specialty = px.bar(
            admitted_chart_df,
            x="admitted_backlog",
            y="Specialty",
            orientation="h",
            color="admitted_backlog_pct",
            color_continuous_scale="Greys",
            title=f"Admitted Backlog (DTA) by Specialty — {selected_specialty_month}",
            hover_data={
                "admitted_backlog": ":,",
                "admitted_backlog_pct": ":.1%",
                "Total": ":,",
                "Specialty": True,
            },
        )

        fig_admitted_specialty.update_layout(
            template="plotly_white",
            xaxis_title="Admitted backlog (DTA patients)",
            yaxis_title="Specialty",
            height=max(420, min(760, 34 * len(admitted_chart_df))),
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_colorbar_title="% of incomplete",
        )

        st.plotly_chart(fig_admitted_specialty, use_container_width=True)

    display_specialty_df = specialty_month_df[
        [
            "Specialty",
            "Total",
            "admitted_backlog",
            "admitted_backlog_pct",
            "waiting_0_18_total",
            "pct_0_18",
            "waiting_18_52_total",
            "waiting_52_plus_total",
            "pct_52_plus",
        ]
    ].rename(
        columns={
            "Total": "RTT Incomplete Pathways",
            "admitted_backlog": "Admitted Backlog (DTA)",
            "admitted_backlog_pct": "Admitted % of Incomplete",
            "waiting_0_18_total": "Patients Within 18 Weeks",
            "pct_0_18": "% Within 18 Weeks",
            "waiting_18_52_total": "18-52 Weeks",
            "waiting_52_plus_total": "52+ Weeks",
            "pct_52_plus": "% 52+ Weeks",
        }
    )

    display_specialty_df["% Within 18 Weeks"] = (
        display_specialty_df["% Within 18 Weeks"] * 100
    ).round(1)
    display_specialty_df["Admitted % of Incomplete"] = (
        display_specialty_df["Admitted % of Incomplete"] * 100
    ).round(1)
    display_specialty_df["% 52+ Weeks"] = (
        display_specialty_df["% 52+ Weeks"] * 100
    ).round(1)

    st.dataframe(display_specialty_df, use_container_width=True)

    trend_specialties = st.multiselect(
        "Select specialties for RTT performance trend",
        options=sorted(specialty_month_df["Specialty"].unique()),
        default=specialty_month_df.head(5)["Specialty"].tolist(),
    )

    if trend_specialties:
        trend_df = specialty_df[specialty_df["Specialty"].isin(trend_specialties)].copy()
        trend_df["% Within 18 Weeks"] = trend_df["pct_0_18"]

        fig_specialty_trend = px.line(
            trend_df,
            x="Month_Date",
            y="% Within 18 Weeks",
            color="Specialty",
            markers=True,
            title="RTT 18-Week Performance Trend by Specialty",
        )

        fig_specialty_trend.add_hline(
            y=0.92,
            line_dash="dash",
            line_color="#991B1B",
            annotation_text="92% standard",
            annotation_position="top right",
        )

        fig_specialty_trend.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="% within 18 weeks",
            yaxis_tickformat=".0%",
            height=560,
            margin=dict(l=20, r=20, t=70, b=20),
        )

        st.plotly_chart(fig_specialty_trend, use_container_width=True)

# ---------------------------------------------------------
# Waiting list shape and size
# ---------------------------------------------------------
st.subheader("Waiting List Shape and Size")

view_type = st.radio(
    "View type",
    ["Chart", "Table"],
    horizontal=True
)

st.write(
    """
    This view shows the full distribution of the waiting list across weekly wait bands
    for the selected month. It provides a more detailed view of backlog shape than the
    0–18 and 52+ summary metrics alone.
    """
)

shape_months = summary_df["Month"].tolist()

selected_shape_month = st.selectbox(
    "Select month for weekly wait-band view",
    options=shape_months,
    index=len(shape_months) - 1 if shape_months else 0,
)

shape_df = wait_band_df[wait_band_df["Month"] == selected_shape_month].copy()

if shape_df.empty:
    st.warning("No data available for the selected month.")
else:
    if view_type == "Chart":
        fig_shape = px.bar(
            shape_df,
            x="Wait_Band_Label",
            y="Volume",
            title=f"Weekly Wait-Band Distribution — {selected_shape_month}",
        )

        fig_shape.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=shape_df.sort_values("Band_Order")["Wait_Band_Label"].tolist(),
            tickangle=45,
        )

        fig_shape.update_layout(
            template="plotly_white",
            xaxis_title="Weekly Wait Band",
            yaxis_title="Patients",
            margin=dict(l=20, r=20, t=60, b=40),
        )

        fig_shape.update_traces(
            hovertemplate="Wait Band: %{x}<br>Patients: %{y:,}<extra></extra>"
        )

        st.plotly_chart(fig_shape, use_container_width=True)

    elif view_type == "Table":
        table_df = shape_df[["Band_Order", "Wait_Band_Label", "Volume"]].copy()
        table_df = table_df.sort_values("Band_Order")

        table_df = table_df.rename(
            columns={
                "Wait_Band_Label": "Weekly Wait Band",
                "Volume": "Patients",
            }
        )

        table_df = table_df[["Weekly Wait Band", "Patients"]]

        st.dataframe(table_df, use_container_width=True)
