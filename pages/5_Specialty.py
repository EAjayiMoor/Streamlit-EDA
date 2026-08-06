import streamlit as st
import pandas as pd
import plotly.express as px

from src.data.rtt_loader import load_all_rtt_files
from src.transforms.rtt_transform import (
    filter_pah_incomplete,
    add_wait_band_metrics,
    summarise_rtt_by_month_specialty,
    get_latest_specialty_backlog,
)
from src.transforms.rtt_transform import build_specialty_heatmap
from src.transforms.rtt_transform import summarise_specialty_weekly_wait_band_distribution
from src.transforms.rtt_transform import summarise_rtt_completions_by_month_specialty

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Specialty View", layout="wide")

st.title("Specialty View")
st.write(
    """
    This page shows RTT backlog by specialty for Princess Alexandra Hospital (PAH),
    using incomplete pathways only. It helps identify which specialties contribute
    most to backlog, how risk is distributed across wait bands, and how backlog
    is changing over time.
    """
)

# ---------------------------------------------------------
# Load and transform RTT data
# ---------------------------------------------------------
try:
    raw_df = load_all_rtt_files()
    pah_df = filter_pah_incomplete(raw_df)
    metric_df = add_wait_band_metrics(pah_df)

    specialty_df = summarise_rtt_by_month_specialty(metric_df)
    specialty_wait_band_df = summarise_specialty_weekly_wait_band_distribution(pah_df)
    # Create a real month field for correct chronological sorting
    # Why:
    # - the heatmap needs to identify latest and previous month per specialty
    # - text month labels alone are not reliable for time ordering
    # Build specialty throughput proxy from completed RTT pathways
    completion_df = summarise_rtt_completions_by_month_specialty(raw_df)
    specialty_df = specialty_df.copy()
    specialty_df["Month_Date"] = pd.to_datetime(specialty_df["Month"], format="%B %Y")
    
    
    # Build heatmap dataset after Month_Date exists
    heatmap_df = build_specialty_heatmap(specialty_df, completion_df)

except Exception as e:
    st.error(f"Error loading or transforming RTT specialty data: {e}")
    st.stop()

if specialty_df.empty:
    st.warning("No specialty RTT data available.")
    st.stop()

# ---------------------------------------------------------
# Sort month properly
# ---------------------------------------------------------
specialty_df = specialty_df.copy()

try:
    specialty_df["Month_Date"] = pd.to_datetime(specialty_df["Month"], format="%B %Y")
except Exception:
    specialty_df["Month_Date"] = range(len(specialty_df))

specialty_df = specialty_df.sort_values(["Month_Date", "Treatment Function Name"]).reset_index(drop=True)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Specialty Filters")
st.sidebar.subheader("Heatmap Weighting")

w_backlog = st.sidebar.slider("Backlog Size Weight", 0.0, 1.0, 0.25)
w_perf = st.sidebar.slider("Performance Risk Weight", 0.0, 1.0, 0.25)
w_severity = st.sidebar.slider("Long Wait Risk Weight", 0.0, 1.0, 0.25)
w_trend = st.sidebar.slider("Trend Risk Weight", 0.0, 1.0, 0.15)
w_throughput = st.sidebar.slider("Throughput Risk Weight", 0.0, 1.0, 0.10)
total_weight = w_backlog + w_perf + w_severity + w_trend + w_throughput

if total_weight == 0:
    total_weight = 1

w_backlog /= total_weight
w_perf /= total_weight
w_severity /= total_weight
w_trend /= total_weight
w_throughput /= total_weight

available_months = specialty_df["Month"].dropna().unique().tolist()
available_months = sorted(
    available_months,
    key=lambda x: pd.to_datetime(x, format="%B %Y") if isinstance(x, str) else x
)

selected_month = st.sidebar.selectbox(
    "Select month for specialty overview",
    options=available_months,
    index=len(available_months) - 1 if available_months else 0,
)

available_specialties = sorted(specialty_df["Treatment Function Name"].dropna().unique().tolist())

selected_specialty = st.sidebar.selectbox(
    "Select specialty for trend view",
    options=available_specialties,
)

profile_mode = st.sidebar.radio(
    "Waiting time profile view",
    options=["Percentage", "Absolute Volume"],
    index=0,
)

top_n = st.sidebar.slider(
    "Number of specialties to display",
    min_value=5,
    max_value=25,
    value=15,
    step=1,
)

# ---------------------------------------------------------
# Filtered views
# ---------------------------------------------------------
latest_df = specialty_df[specialty_df["Month"] == selected_month].copy()
latest_df = latest_df.sort_values("Total", ascending=False).head(top_n)

trend_df = specialty_df[specialty_df["Treatment Function Name"] == selected_specialty].copy()
trend_df = trend_df.sort_values("Month_Date")

if latest_df.empty:
    st.warning("No specialty data available for the selected month.")
    st.stop()

# ---------------------------------------------------------
# Summary metrics for selected month
# ---------------------------------------------------------
total_backlog_selected_month = int(latest_df["Total"].sum())
top_specialty_name = latest_df.iloc[0]["Treatment Function Name"]
top_specialty_total = int(latest_df.iloc[0]["Total"])
top_specialty_pct_52 = float(latest_df.iloc[0]["pct_52_plus"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Month Backlog (Top N Shown)", f"{total_backlog_selected_month:,}")

with col2:
    st.metric("Largest Specialty Backlog", f"{top_specialty_total:,}", delta=top_specialty_name)

with col3:
    st.metric("Top Specialty % 52+", f"{top_specialty_pct_52:.1%}")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Backlog Overview", "Waiting Time Profile", "Trend View", "Summary Table", "Heatmap","Distribution Curve"]
)

# ---------------------------------------------------------
# Tab 1: Specialty backlog overview
# ---------------------------------------------------------
with tab1:
    st.subheader("Specialty Backlog Overview")

    st.write(
        """
        This ranked chart shows the largest specialty contributors to the RTT backlog
        in the selected month. It is intended as the anchor view for specialty-level discussion.
        """
    )

    fig_backlog = px.bar(
        latest_df.sort_values("Total", ascending=True),
        x="Total",
        y="Treatment Function Name",
        orientation="h",
        title=f"Specialty Backlog Overview — {selected_month}",
    )

    fig_backlog.update_layout(
        template="plotly_white",
        xaxis_title="Incomplete RTT Pathways",
        yaxis_title="Specialty",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig_backlog.update_traces(
        hovertemplate="Specialty: %{y}<br>Backlog: %{x:,}<extra></extra>"
    )

    st.plotly_chart(fig_backlog, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Waiting time distribution by specialty
# ---------------------------------------------------------
with tab2:
    st.subheader("Waiting Time Distribution by Specialty")

    st.write(
        """
        This view shows backlog composition by specialty across three wait segments:
        0–18 weeks, 18–52 weeks, and 52+ weeks. Percentage view is preferred for
        comparing risk profile across specialties of different sizes.
        """
    )

    profile_df = latest_df.copy()

    if profile_mode == "Percentage":
        plot_df = profile_df[
            [
                "Treatment Function Name",
                "pct_0_18",
                "pct_18_52",
                "pct_52_plus",
            ]
        ].rename(
            columns={
                "pct_0_18": "0–18 Weeks",
                "pct_18_52": "18–52 Weeks",
                "pct_52_plus": "52+ Weeks",
            }
        )

        fig_profile = px.bar(
            plot_df,
            x="Treatment Function Name",
            y=["0–18 Weeks", "18–52 Weeks", "52+ Weeks"],
            title=f"Waiting Time Distribution by Specialty — {selected_month}",
        )

        fig_profile.update_layout(
            template="plotly_white",
            xaxis_title="Specialty",
            yaxis_title="Percentage of Backlog",
            yaxis_tickformat=".0%",
            barmode="stack",
            margin=dict(l=20, r=20, t=60, b=20),
        )

    else:
        plot_df = profile_df[
            [
                "Treatment Function Name",
                "waiting_0_18_total",
                "waiting_18_52_total",
                "waiting_52_plus_total",
            ]
        ].rename(
            columns={
                "waiting_0_18_total": "0–18 Weeks",
                "waiting_18_52_total": "18–52 Weeks",
                "waiting_52_plus_total": "52+ Weeks",
            }
        )

        fig_profile = px.bar(
            plot_df,
            x="Treatment Function Name",
            y=["0–18 Weeks", "18–52 Weeks", "52+ Weeks"],
            title=f"Waiting Time Distribution by Specialty — {selected_month}",
        )

        fig_profile.update_layout(
            template="plotly_white",
            xaxis_title="Specialty",
            yaxis_title="Patients",
            barmode="stack",
            margin=dict(l=20, r=20, t=60, b=20),
        )

    fig_profile.update_xaxes(tickangle=45)

    st.plotly_chart(fig_profile, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Specialty trend view
# ---------------------------------------------------------
with tab3:
    st.subheader("Trend View")

    st.write(
        """
        This line chart shows direction of travel over time for the selected specialty.
        It supports discussion of whether backlog pressure is improving, worsening,
        or remaining stable.
        """
    )

    fig_trend = px.line(
        trend_df,
        x="Month",
        y="Total",
        markers=True,
        title=f"Backlog Trend — {selected_specialty}",
    )

    fig_trend.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Incomplete RTT Pathways",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig_trend.update_traces(
        hovertemplate="Month: %{x}<br>Backlog: %{y:,}<extra></extra>"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    trend_pct_df = trend_df.copy()
    trend_pct_df["% 52+"] = trend_pct_df["pct_52_plus"]

    fig_risk = px.line(
        trend_pct_df,
        x="Month",
        y="% 52+",
        markers=True,
        title=f"Long-Wait Risk Trend — {selected_specialty}",
    )

    fig_risk.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="% 52+ Waits",
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig_risk.update_traces(
        hovertemplate="Month: %{x}<br>% 52+: %{y:.1%}<extra></extra>"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

# ---------------------------------------------------------
# Tab 4: Summary table
# ---------------------------------------------------------
with tab4:
    st.subheader("Specialty Summary Table")

    table_df = latest_df[
        [
            "Treatment Function Name",
            "Total",
            "waiting_0_18_total",
            "waiting_18_52_total",
            "waiting_52_plus_total",
            "pct_0_18",
            "pct_18_52",
            "pct_52_plus",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "Treatment Function Name": "Specialty",
            "Total": "Backlog Total",
            "waiting_0_18_total": "0–18 Weeks",
            "waiting_18_52_total": "18–52 Weeks",
            "waiting_52_plus_total": "52+ Weeks",
            "pct_0_18": "% 0–18",
            "pct_18_52": "% 18–52",
            "pct_52_plus": "% 52+",
        }
    )

    table_df["% 0–18"] = table_df["% 0–18"].map(lambda x: f"{x:.1%}")
    table_df["% 18–52"] = table_df["% 18–52"].map(lambda x: f"{x:.1%}")
    table_df["% 52+"] = table_df["% 52+"].map(lambda x: f"{x:.1%}")

    st.dataframe(table_df, use_container_width=True)
# ---------------------------------------------------------
# Tab 5: HeatMap
# ---------------------------------------------------------
with tab5:
    st.subheader("Specialty Heatmap")

    st.write(
        """
        This heatmap compares specialties across backlog size, performance risk,
        long-wait risk, trend risk, and throughput risk. Colours show relative
        position across specialties, not absolute good/bad performance.
        """
    )

    selected_specialties = latest_df["Treatment Function Name"].tolist()

    heatmap_plot_df = heatmap_df[
        heatmap_df["Specialty"].isin(selected_specialties)
    ].copy()
    # Recalculate Overall Score dynamically based on sidebar weightings
    heatmap_plot_df["Overall Score"] = (
        heatmap_plot_df["Backlog Size"] * w_backlog
        + heatmap_plot_df["Performance Risk"] * w_perf
        + heatmap_plot_df["Long Wait Risk"] * w_severity
        + heatmap_plot_df["Trend Risk"] * w_trend
        + heatmap_plot_df["Throughput Risk"] * w_throughput
)

# Sort highest pressure specialties to the top
    heatmap_plot_df = heatmap_plot_df.sort_values("Overall Score", ascending=False)

    # Keep only the columns needed for the heatmap, in display order
    heatmap_plot_df = heatmap_plot_df[
        [
            "Specialty",
            "Overall Score",
            "Backlog Size",
            "Performance Risk",
            "Long Wait Risk",
            "Trend Risk",
            "Throughput Risk",
        ]
    ].copy()

    # Debug check
    with st.expander("Debug: Heatmap columns"):
        st.write(heatmap_plot_df.columns.tolist())
        st.dataframe(heatmap_plot_df.head(), use_container_width=True)

    # Only set index if Specialty is still a column
    if "Specialty" in heatmap_plot_df.columns:
        heatmap_plot_df = heatmap_plot_df.set_index("Specialty")

    fig_heatmap = px.imshow(
        heatmap_plot_df,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        text_auto=".2f",
    )

    fig_heatmap.update_layout(
        title="Specialty Comparative Heatmap",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.caption(
        "Colour intensity reflects relative pressure across specialties. "
        "Darker cells indicate higher relative backlog, poorer relative performance, "
        "higher long-wait risk, worsening trend, or lower throughput."
    )    
# ---------------------------------------------------------
# Tab 6: Distribution curves
# ---------------------------------------------------------
with tab6:
    view_type = st.radio(
        "View type",
        ["Chart", "Table"],
        horizontal=True
    )

    st.subheader("Specialty Distribution Curve")

    st.write(
        """
        Full weekly wait-band distribution for the selected specialty.
        Shows how backlog is distributed across all wait durations.
        """
    )

    distribution_months = sorted(
        specialty_wait_band_df["Month"].unique(),
        key=lambda x: pd.to_datetime(x, format="%B %Y")
    )

    selected_distribution_month = st.selectbox(
        "Select month",
        options=distribution_months,
        index=len(distribution_months) - 1
    )

    dist_df = specialty_wait_band_df[
        (specialty_wait_band_df["Treatment Function Name"] == selected_specialty)
        & (specialty_wait_band_df["Month"] == selected_distribution_month)
    ].copy()

    if dist_df.empty:
        st.warning("No data available for this selection")
    else:
        if view_type == "Chart":
            fig = px.bar(
                dist_df,
                x="Wait_Band_Label",
                y="Volume",
                title=f"{selected_specialty} — Weekly Distribution ({selected_distribution_month})",
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=dist_df.sort_values("Band_Order")["Wait_Band_Label"],
                tickangle=45
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Weekly Wait Band",
                yaxis_title="Patients",
                margin=dict(l=20, r=20, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

        elif view_type == "Table":
            table_df = dist_df[["Band_Order", "Wait_Band_Label", "Volume"]].copy()
            table_df = table_df.sort_values("Band_Order")

            table_df = table_df.rename(
                columns={
                    "Wait_Band_Label": "Weekly Wait Band",
                    "Volume": "Patients"
                }
            )

            table_df = table_df[["Weekly Wait Band", "Patients"]]

            st.dataframe(table_df, use_container_width=True)
# ---------------------------------------------------------
# Methodology / note
# ---------------------------------------------------------
with st.expander("Methodology"):
    st.write(
        """
        - **Scope:** PAH only
        - **Population:** Incomplete RTT pathways
        - **Backlog overview:** Total incomplete pathways by specialty
        - **Waiting time profile:** Split into 0–18 weeks, 18–52 weeks, and 52+ weeks
        - **Trend view:** Monthly backlog trend for selected specialty
        - **Note:** Throughput views require admitted and non-admitted RTT datasets and will be added separately
        """
    )