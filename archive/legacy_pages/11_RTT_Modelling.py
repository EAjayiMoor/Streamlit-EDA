import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.rtt_loader import load_all_rtt_files
from src.data.theatre_loader import (
    load_theatre_activity_data,
    summarise_theatre_capacity,
)
from src.models.rtt_forecast import (
    build_monthly_activity_profile,
    calculate_baseline_inputs,
    calculate_outpatient_additional_appointments,
    calculate_theatre_additional_cases,
    create_actual_history_series,
    add_latest_actual_bridge,
    prepare_rtt_history,
    simulate_rtt_forecast_from_weekly_bands,
)
from src.transforms.rtt_transform import (
    add_wait_band_metrics,
    filter_pah_all_rtt_parts,
    filter_pah_incomplete,
    summarise_rtt_additions_by_month,
    summarise_rtt_by_month,
    summarise_rtt_completions_by_month,
    summarise_weekly_wait_band_distribution,
)


st.set_page_config(
    page_title="RTT Forecast Modelling",
    layout="wide",
)

st.title("RTT Forecast Modelling")
st.caption(
    "Ten-month RTT projection with do-nothing, 50%, 75%, and 100% intervention delivery scenarios."
)


EFFORT_LEVELS = {
    "50% delivery": 0.50,
    "75% delivery": 0.75,
    "100% delivery": 1.00,
}


def pct_value(value: float) -> float:
    return float(value) * 100


def format_month(value) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%B %Y")


def build_intervention_summary(
    horizon_months: int,
    active_weeks: float,
    ramp_months: int,
    theatre_inputs: dict,
    outpatient_inputs: dict,
    theatre_rtt_conversion: float,
    outpatient_rtt_conversion: float,
) -> pd.DataFrame:
    rows = []

    for label, effort in EFFORT_LEVELS.items():
        theatre_result = calculate_theatre_additional_cases(
            effort=effort,
            active_weeks=active_weeks,
            horizon_months=horizon_months,
            **theatre_inputs,
        )

        outpatient_result = calculate_outpatient_additional_appointments(
            effort=effort,
            active_weeks=active_weeks,
            horizon_months=horizon_months,
            **outpatient_inputs,
        )

        steady_monthly_effective_activity = (
            theatre_result["monthly_cases"] * theatre_rtt_conversion
            + outpatient_result["monthly_appointments"] * outpatient_rtt_conversion
        )

        profile = build_monthly_activity_profile(
            steady_monthly_activity=steady_monthly_effective_activity,
            horizon_months=horizon_months,
            ramp_months=ramp_months,
        )

        rows.append(
            {
                "Scenario": label,
                "Effort": effort,
                "Theatre Monthly Cases": theatre_result["monthly_cases"],
                "Theatre Total Cases": theatre_result["total_cases"],
                "Theatre Utilisation Gap": theatre_result["utilisation_gap"],
                "Outpatient Template Fill": outpatient_result["template_fill"],
                "Outpatient DNA Reduction": outpatient_result["dna_reduction"],
                "Outpatient PIFU": outpatient_result["pifu"],
                "Outpatient F:N Improvement": outpatient_result["fn_ratio"],
                "Outpatient Monthly Appointments": outpatient_result[
                    "monthly_appointments"
                ],
                "Outpatient Total Appointments": outpatient_result[
                    "total_appointments"
                ],
                "Monthly Effective RTT Activity": steady_monthly_effective_activity,
                "Delivered Effective RTT Activity": sum(profile),
                "Activity Profile": profile,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

try:
    raw_rtt_df = load_all_rtt_files()
    pah_all_parts_df = filter_pah_all_rtt_parts(raw_rtt_df)
    pah_incomplete_df = filter_pah_incomplete(pah_all_parts_df)
    metric_df = add_wait_band_metrics(pah_incomplete_df)

    backlog_df = summarise_rtt_by_month(metric_df)
    demand_df = summarise_rtt_additions_by_month(pah_all_parts_df)
    completion_df = summarise_rtt_completions_by_month(pah_all_parts_df)
    history_df = prepare_rtt_history(backlog_df, demand_df, completion_df)
    wait_band_df = summarise_weekly_wait_band_distribution(pah_incomplete_df)

except Exception as e:
    st.error(f"Error loading RTT data: {e}")
    st.stop()

try:
    theatre_df = load_theatre_activity_data()
    theatre_capacity = summarise_theatre_capacity(theatre_df)
except Exception:
    theatre_capacity = pd.Series(dtype="float64")

if history_df.empty:
    st.warning("No RTT history is available for modelling.")
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

latest_actual = history_df.sort_values("Month_Date").iloc[-1]
latest_month_label = format_month(latest_actual["Month_Date"])

latest_wait_band_df = wait_band_df[
    wait_band_df["Month"] == latest_actual["Month"]
].copy()

latest_wait_bands = (
    latest_wait_band_df.sort_values("Band_Order")
    .set_index("Band_Order")["Volume"]
    .reindex(range(105), fill_value=0)
    .astype(float)
    .tolist()
)

measured_sessions_per_week = float(
    theatre_capacity.get("Sessions_Per_Week", 18.0)
)
measured_session_minutes = float(
    theatre_capacity.get("Median_Session_Minutes", 240.0)
)
measured_case_duration = float(
    theatre_capacity.get("Average_Case_Duration_Minutes", 45.0)
)
measured_utilisation = float(theatre_capacity.get("Utilisation", 0.72))

st.sidebar.header("Forecast Setup")

horizon_months = st.sidebar.slider(
    "Projection horizon",
    min_value=3,
    max_value=24,
    value=10,
    step=1,
)

baseline_months = st.sidebar.slider(
    "Historical baseline months",
    min_value=3,
    max_value=min(24, max(len(history_df), 3)),
    value=min(12, max(len(history_df), 3)),
    step=1,
)

baseline_mode = st.sidebar.selectbox(
    "Do-nothing baseline",
    [
        "Conservative observed flow",
        "Blended trend and flow",
        "Recent backlog trend",
    ],
    help=(
        "Conservative observed flow uses recent RTT additions and completions "
        "without assuming recent backlog reductions continue indefinitely."
    ),
)

active_weeks = st.sidebar.number_input(
    "Active delivery weeks",
    min_value=1.0,
    max_value=104.0,
    value=43.0,
    step=1.0,
)

use_seasonality = st.sidebar.checkbox(
    "Apply historical seasonality",
    value=True,
)

intervention_targeting = st.sidebar.selectbox(
    "Intervention removes backlog from",
    ["Longest waits first", "18-52 first", "Proportional"],
)

ramp_months = st.sidebar.slider(
    "Ramp-up months",
    min_value=0,
    max_value=6,
    value=0,
    step=1,
)

selected_delivery_view = st.sidebar.selectbox(
    "Detailed delivery scenario",
    ["All scenarios"] + list(EFFORT_LEVELS.keys()),
)

st.sidebar.header("Theatre Lever")

theatre_default_source = st.sidebar.radio(
    "Theatre defaults",
    ["PAH measured defaults", "Planning placeholder"],
)

if theatre_default_source == "Planning placeholder":
    default_sessions_per_week = 18.0
    default_session_minutes = 240.0
    default_utilisation = 0.72
    default_case_duration = 45.0
    theatre_source_key = "planning"
else:
    default_sessions_per_week = measured_sessions_per_week
    default_session_minutes = measured_session_minutes
    default_utilisation = measured_utilisation
    default_case_duration = measured_case_duration
    theatre_source_key = "measured"

sessions_per_week = st.sidebar.number_input(
    "Theatre sessions/week",
    min_value=0.0,
    max_value=500.0,
    value=round(default_sessions_per_week, 1),
    step=1.0,
    key=f"theatre_sessions_{theatre_source_key}",
)

session_minutes = st.sidebar.number_input(
    "Average session length (mins)",
    min_value=30.0,
    max_value=720.0,
    value=round(default_session_minutes, 0),
    step=15.0,
    key=f"theatre_session_minutes_{theatre_source_key}",
)

current_utilisation_pct = st.sidebar.number_input(
    "Current theatre utilisation %",
    min_value=0.0,
    max_value=100.0,
    value=round(pct_value(default_utilisation), 1),
    step=0.5,
    key=f"theatre_current_utilisation_{theatre_source_key}",
)

target_utilisation_pct = st.sidebar.number_input(
    "Target theatre utilisation %",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=0.5,
    key=f"theatre_target_utilisation_{theatre_source_key}",
)

avg_case_duration = st.sidebar.number_input(
    "Average case duration (mins)",
    min_value=1.0,
    max_value=480.0,
    value=round(default_case_duration, 0),
    step=5.0,
    key=f"theatre_avg_case_duration_{theatre_source_key}",
)

theatre_rtt_conversion_pct = st.sidebar.slider(
    "Theatre RTT impact %",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
)

st.sidebar.header("Outpatient Lever")

clinic_sessions_per_week = st.sidebar.number_input(
    "Clinic sessions/week",
    min_value=0.0,
    max_value=1000.0,
    value=40.0,
    step=1.0,
)

patients_per_session = st.sidebar.number_input(
    "Patients/session",
    min_value=0.0,
    max_value=100.0,
    value=9.0,
    step=1.0,
)

template_current_fill_pct = st.sidebar.number_input(
    "Current template fill %",
    min_value=0.0,
    max_value=100.0,
    value=78.0,
    step=0.5,
)

template_target_fill_pct = st.sidebar.number_input(
    "Target template fill %",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=0.5,
)

template_rtt_relevant_share_pct = st.sidebar.slider(
    "RTT-relevant template share %",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
)

eligible_new_per_week = st.sidebar.number_input(
    "Eligible new appts/week",
    min_value=0.0,
    max_value=10000.0,
    value=80.0,
    step=5.0,
)

eligible_follow_up_per_week = st.sidebar.number_input(
    "Eligible FU appts/week",
    min_value=0.0,
    max_value=10000.0,
    value=240.0,
    step=5.0,
)

current_dna_rate_pct = st.sidebar.number_input(
    "Current DNA rate %",
    min_value=0.0,
    max_value=100.0,
    value=7.0,
    step=0.5,
)

target_dna_rate_pct = st.sidebar.number_input(
    "Target DNA rate %",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.5,
)

pifu_conversion_pct = st.sidebar.number_input(
    "FU slots moved via PIFU %",
    min_value=0.0,
    max_value=100.0,
    value=12.0,
    step=0.5,
)

fn_ratio_improvement_pct = st.sidebar.number_input(
    "F:N improvement %",
    min_value=0.0,
    max_value=100.0,
    value=26.0,
    step=0.5,
)

outpatient_rtt_conversion_pct = st.sidebar.slider(
    "Outpatient RTT impact %",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
)


# ---------------------------------------------------------
# Run model
# ---------------------------------------------------------

theatre_inputs = {
    "sessions_per_week": sessions_per_week,
    "session_minutes": session_minutes,
    "current_utilisation": current_utilisation_pct / 100,
    "target_utilisation": target_utilisation_pct / 100,
    "avg_case_duration_minutes": avg_case_duration,
}

outpatient_inputs = {
    "clinic_sessions_per_week": clinic_sessions_per_week,
    "patients_per_session": patients_per_session,
    "template_current_fill": template_current_fill_pct / 100,
    "template_target_fill": template_target_fill_pct / 100,
    "template_rtt_relevant_share": template_rtt_relevant_share_pct / 100,
    "eligible_new_per_week": eligible_new_per_week,
    "eligible_follow_up_per_week": eligible_follow_up_per_week,
    "current_dna_rate": current_dna_rate_pct / 100,
    "target_dna_rate": target_dna_rate_pct / 100,
    "pifu_conversion_rate": pifu_conversion_pct / 100,
    "fn_ratio_improvement_rate": fn_ratio_improvement_pct / 100,
}

intervention_summary_df = build_intervention_summary(
    horizon_months=horizon_months,
    active_weeks=active_weeks,
    ramp_months=ramp_months,
    theatre_inputs=theatre_inputs,
    outpatient_inputs=outpatient_inputs,
    theatre_rtt_conversion=theatre_rtt_conversion_pct / 100,
    outpatient_rtt_conversion=outpatient_rtt_conversion_pct / 100,
)

do_nothing_df = simulate_rtt_forecast_from_weekly_bands(
    history_df=history_df,
    latest_wait_bands=latest_wait_bands,
    horizon_months=horizon_months,
    baseline_months=baseline_months,
    additional_activity_by_month=[0.0] * horizon_months,
    scenario_name="Do Nothing",
    use_seasonality=use_seasonality,
    intervention_targeting=intervention_targeting,
    baseline_mode=baseline_mode,
)

scenario_dfs = [do_nothing_df]

for _, row in intervention_summary_df.iterrows():
    scenario_dfs.append(
        simulate_rtt_forecast_from_weekly_bands(
            history_df=history_df,
            latest_wait_bands=latest_wait_bands,
            horizon_months=horizon_months,
            baseline_months=baseline_months,
            additional_activity_by_month=row["Activity Profile"],
            scenario_name=row["Scenario"],
            use_seasonality=use_seasonality,
            intervention_targeting=intervention_targeting,
            baseline_mode=baseline_mode,
        )
    )

projection_df = pd.concat(scenario_dfs, ignore_index=True)
actual_df = create_actual_history_series(history_df)
plot_df = add_latest_actual_bridge(actual_df, projection_df)
combined_plot_df = pd.concat([actual_df, plot_df], ignore_index=True, sort=False)

baseline_inputs = calculate_baseline_inputs(
    history_df=history_df,
    baseline_months=baseline_months,
    baseline_mode=baseline_mode,
)

end_state_df = projection_df.groupby("Scenario").tail(1).copy()
do_nothing_end = end_state_df[end_state_df["Scenario"] == "Do Nothing"].iloc[0]
intervention_end_state = end_state_df[
    end_state_df["Scenario"].isin(EFFORT_LEVELS.keys())
].copy()
intervention_end_state["Backlog Avoided"] = (
    do_nothing_end["Closing Backlog"]
    - intervention_end_state["Closing Backlog"]
)
intervention_end_state["52+ Avoided"] = (
    do_nothing_end["52+ Weeks"] - intervention_end_state["52+ Weeks"]
)

if selected_delivery_view == "All scenarios":
    selected_intervention_scenarios = list(EFFORT_LEVELS.keys())
else:
    selected_intervention_scenarios = [selected_delivery_view]

backlog_scenario_order = ["Actual", "Do Nothing"] + list(EFFORT_LEVELS.keys())
detail_scenario_order = ["Actual", "Do Nothing"] + selected_intervention_scenarios

do_nothing_compare_df = do_nothing_df[
    ["Month_Date", "Closing Backlog", "52+ Weeks"]
].rename(
    columns={
        "Closing Backlog": "Do Nothing Backlog",
        "52+ Weeks": "Do Nothing 52+",
    }
)

lever_definitions = [
    {
        "Lever": "All operational levers combined",
        "Monthly Activity Column": "Monthly Effective RTT Activity",
        "Conversion": 1.0,
    },
    {
        "Lever": "Theatre utilisation, booking and scheduling",
        "Monthly Activity Column": "Theatre Monthly Cases",
        "Conversion": theatre_rtt_conversion_pct / 100,
    },
    {
        "Lever": "Outpatient template management",
        "Monthly Activity Column": "Outpatient Template Fill",
        "Conversion": outpatient_rtt_conversion_pct / 100,
    },
    {
        "Lever": "Outpatient booking: DNA reduction",
        "Monthly Activity Column": "Outpatient DNA Reduction",
        "Conversion": outpatient_rtt_conversion_pct / 100,
    },
    {
        "Lever": "Outpatient PIFU conversion",
        "Monthly Activity Column": "Outpatient PIFU",
        "Conversion": outpatient_rtt_conversion_pct / 100,
    },
    {
        "Lever": "Outpatient F:N ratio improvement",
        "Monthly Activity Column": "Outpatient F:N Improvement",
        "Conversion": outpatient_rtt_conversion_pct / 100,
    },
]

lever_projection_parts = []

for _, scenario_row in intervention_summary_df.iterrows():
    for lever_definition in lever_definitions:
        monthly_activity = (
            scenario_row[lever_definition["Monthly Activity Column"]]
            * lever_definition["Conversion"]
        )
        activity_profile = build_monthly_activity_profile(
            steady_monthly_activity=monthly_activity,
            horizon_months=horizon_months,
            ramp_months=ramp_months,
        )

        lever_df = simulate_rtt_forecast_from_weekly_bands(
            history_df=history_df,
            latest_wait_bands=latest_wait_bands,
            horizon_months=horizon_months,
            baseline_months=baseline_months,
            additional_activity_by_month=activity_profile,
            scenario_name=scenario_row["Scenario"],
            use_seasonality=use_seasonality,
            intervention_targeting=intervention_targeting,
            baseline_mode=baseline_mode,
        )

        lever_df["Lever"] = lever_definition["Lever"]
        lever_df["Delivery Scenario"] = scenario_row["Scenario"]
        lever_df["Monthly Lever Activity"] = monthly_activity
        lever_df["Cumulative Lever Activity"] = lever_df[
            "Additional Activity"
        ].cumsum()
        lever_projection_parts.append(lever_df)

lever_projection_df = pd.concat(lever_projection_parts, ignore_index=True)

lever_projection_df = lever_projection_df.merge(
    do_nothing_compare_df,
    on="Month_Date",
    how="left",
)

lever_projection_df["Backlog Avoided"] = (
    lever_projection_df["Do Nothing Backlog"]
    - lever_projection_df["Closing Backlog"]
)

lever_projection_df["52+ Avoided"] = (
    lever_projection_df["Do Nothing 52+"]
    - lever_projection_df["52+ Weeks"]
)

lever_end_state_df = (
    lever_projection_df.sort_values("Month_Date")
    .groupby(["Delivery Scenario", "Lever"], as_index=False)
    .tail(1)
)

lever_order = [item["Lever"] for item in lever_definitions]


# ---------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------

st.subheader("Current Position and Forecast Basis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest RTT Month", latest_month_label)
c2.metric("Current RTT Backlog", f"{latest_actual['Total']:,.0f}")
c3.metric("Current 52+ Waits", f"{latest_actual['waiting_52_plus_total']:,.0f}")
c4.metric(
    "Baseline Monthly Movement",
    f"{baseline_inputs['blended_monthly_change']:+,.0f}",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Baseline Demand", f"{baseline_inputs['monthly_demand']:,.0f}")
c6.metric("Baseline Throughput", f"{baseline_inputs['monthly_throughput']:,.0f}")
c7.metric("Theatre 100% Cases", f"{intervention_summary_df.iloc[-1]['Theatre Total Cases']:,.0f}")
c8.metric(
    "OP 100% Appointments",
    f"{intervention_summary_df.iloc[-1]['Outpatient Total Appointments']:,.0f}",
)

st.caption(
    f"Do-nothing baseline: {baseline_mode}. "
    "Use Conservative observed flow when recent backlog improvement is not yet a stable run-rate."
)


# ---------------------------------------------------------
# Main charts
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "RTT Activity Impact",
        "RTT Lever Impact",
        "Waiting-Time Impact",
        "Operational Levers",
        "Scenario Table",
    ]
)

with tab1:
    st.subheader("RTT Backlog Impact of Additional Activity")

    st.markdown(
        """
        This view shows the effect of running additional theatre cases and outpatient
        appointments over the forecast period. The 50%, 75%, and 100% lines represent
        increasing levels of in-year delivery above the do-nothing RTT trajectory.
        """
    )

    chart_df = combined_plot_df[
        combined_plot_df["Scenario"].isin(backlog_scenario_order)
    ].copy()

    fig = px.line(
        chart_df,
        x="Month_Date",
        y="Closing Backlog",
        color="Scenario",
        category_orders={"Scenario": backlog_scenario_order},
        markers=True,
        title="RTT Backlog Forecast",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="RTT backlog",
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_traces(
        hovertemplate="Month: %{x|%b %Y}<br>Backlog: %{y:,.0f}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    impact_display_df = intervention_end_state[
        intervention_end_state["Scenario"].isin(EFFORT_LEVELS.keys())
    ][
        [
            "Scenario",
            "Closing Backlog",
            "Backlog Avoided",
            "52+ Weeks",
            "52+ Avoided",
            "% Within 18 Weeks",
        ]
    ].copy()

    impact_display_df["Closing Backlog"] = impact_display_df[
        "Closing Backlog"
    ].round(0)
    impact_display_df["Backlog Avoided"] = impact_display_df[
        "Backlog Avoided"
    ].round(0)
    impact_display_df["52+ Weeks"] = impact_display_df["52+ Weeks"].round(0)
    impact_display_df["52+ Avoided"] = impact_display_df["52+ Avoided"].round(0)
    impact_display_df["% Within 18 Weeks"] = (
        impact_display_df["% Within 18 Weeks"] * 100
    ).round(1)

    with st.expander("End-state numbers"):
        st.dataframe(impact_display_df, use_container_width=True)

with tab2:
    st.subheader("RTT Impact of Booking, Scheduling and Template Levers")

    st.markdown(
        """
        The default view shows the combined RTT effect of all operational levers.
        Individual lever options show marginal impact, so they will look smaller
        on a full-history backlog chart.
        """
    )

    selected_lever = st.selectbox(
        "Lever trajectory",
        lever_order,
    )

    main_chart_scale = st.radio(
        "Main chart scale",
        [
            "Full backlog history",
            "Forecast-only zoom",
            "Indexed to latest actual = 100",
            "Backlog avoided vs do nothing",
        ],
        horizontal=True,
    )

    selected_lever_df = lever_projection_df[
        lever_projection_df["Lever"] == selected_lever
    ][["Month_Date", "Scenario", "Closing Backlog"]].copy()

    selected_lever_impact_df = lever_projection_df[
        lever_projection_df["Lever"] == selected_lever
    ].copy()

    do_nothing_line_df = do_nothing_df[
        ["Month_Date", "Scenario", "Closing Backlog"]
    ].copy()

    lever_projection_line_df = pd.concat(
        [do_nothing_line_df, selected_lever_df],
        ignore_index=True,
    )

    lever_line_df = pd.concat(
        [
            actual_df[["Month_Date", "Scenario", "Closing Backlog"]],
            add_latest_actual_bridge(actual_df, lever_projection_line_df),
        ],
        ignore_index=True,
        sort=False,
    )

    lever_scenario_order = ["Actual", "Do Nothing"] + list(EFFORT_LEVELS.keys())

    yaxis_range = None

    if main_chart_scale == "Forecast-only zoom":
        chart_df = add_latest_actual_bridge(
            actual_df,
            lever_projection_line_df,
        )
        chart_df = chart_df[chart_df["Scenario"] != "Actual"].copy()
        y_col = "Closing Backlog"
        color_col = "Scenario"
        category_orders = {"Scenario": ["Do Nothing"] + list(EFFORT_LEVELS.keys())}
        yaxis_title = "RTT backlog"
        chart_title = f"Forecast-Only RTT Backlog: {selected_lever}"

        zoom_min = chart_df[y_col].min()
        zoom_max = chart_df[y_col].max()
        zoom_padding = max((zoom_max - zoom_min) * 0.12, 250)
        yaxis_range = [
            max(0, zoom_min - zoom_padding),
            zoom_max + zoom_padding,
        ]

    elif main_chart_scale == "Indexed to latest actual = 100":
        latest_actual_value = float(
            actual_df.sort_values("Month_Date").iloc[-1]["Closing Backlog"]
        )
        chart_df = lever_line_df.copy()
        chart_df["Indexed Backlog"] = (
            chart_df["Closing Backlog"] / latest_actual_value * 100
            if latest_actual_value
            else 0
        )
        y_col = "Indexed Backlog"
        color_col = "Scenario"
        category_orders = {"Scenario": lever_scenario_order}
        yaxis_title = "Indexed backlog, latest actual = 100"
        chart_title = f"Indexed RTT Backlog Trajectory: {selected_lever}"

    elif main_chart_scale == "Backlog avoided vs do nothing":
        do_nothing_zero_df = do_nothing_df[["Month_Date"]].copy()
        do_nothing_zero_df["Scenario"] = "Do Nothing"
        do_nothing_zero_df["Backlog Avoided"] = 0

        scenario_avoided_df = selected_lever_impact_df[
            ["Month_Date", "Delivery Scenario", "Backlog Avoided"]
        ].rename(columns={"Delivery Scenario": "Scenario"})

        chart_df = pd.concat(
            [do_nothing_zero_df, scenario_avoided_df],
            ignore_index=True,
        )
        y_col = "Backlog Avoided"
        color_col = "Scenario"
        category_orders = {"Scenario": ["Do Nothing"] + list(EFFORT_LEVELS.keys())}
        yaxis_title = "Backlog avoided"
        chart_title = f"Backlog Avoided vs Do Nothing: {selected_lever}"

    else:
        chart_df = lever_line_df.copy()
        y_col = "Closing Backlog"
        color_col = "Scenario"
        category_orders = {"Scenario": lever_scenario_order}
        yaxis_title = "RTT backlog"
        chart_title = f"RTT Backlog Trajectory: {selected_lever}"

    fig_lever_line = px.line(
        chart_df,
        x="Month_Date",
        y=y_col,
        color=color_col,
        category_orders=category_orders,
        markers=True,
        title=chart_title,
    )

    fig_lever_line.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    if yaxis_range:
        fig_lever_line.update_yaxes(range=yaxis_range)

    st.plotly_chart(fig_lever_line, use_container_width=True)

    lever_table_df = lever_end_state_df[
        lever_end_state_df["Lever"] == selected_lever
    ][
        [
            "Delivery Scenario",
            "Cumulative Lever Activity",
            "Closing Backlog",
            "Backlog Avoided",
            "52+ Weeks",
            "52+ Avoided",
        ]
    ].copy()

    for col in [
        "Cumulative Lever Activity",
        "Closing Backlog",
        "Backlog Avoided",
        "52+ Weeks",
        "52+ Avoided",
    ]:
        lever_table_df[col] = lever_table_df[col].round(0)

    with st.expander(f"End-state numbers: {selected_lever}"):
        st.dataframe(lever_table_df, use_container_width=True)

with tab3:
    metric_choice = st.radio(
        "Metric",
        ["52+ Weeks", "% Within 18 Weeks", "0-18 Weeks", "18-52 Weeks"],
        horizontal=True,
    )

    wait_df = combined_plot_df[
        combined_plot_df["Scenario"].isin(detail_scenario_order)
    ].copy()

    y_title = metric_choice

    if metric_choice == "% Within 18 Weeks":
        wait_df[metric_choice] = wait_df[metric_choice] * 100
        y_title = "% within 18 weeks"

    fig_wait = px.line(
        wait_df,
        x="Month_Date",
        y=metric_choice,
        color="Scenario",
        category_orders={"Scenario": detail_scenario_order},
        markers=True,
        title=f"{metric_choice} Forecast",
    )

    fig_wait.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title=y_title,
        height=620,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(fig_wait, use_container_width=True)

    movement_df = projection_df[
        ["Month_Date", "Scenario", "Demand", "Adjusted Throughput", "Net Change"]
    ].melt(
        id_vars=["Month_Date", "Scenario"],
        var_name="Metric",
        value_name="Patients",
    )

    movement_df = movement_df[
        movement_df["Scenario"].isin(["Do Nothing"] + selected_intervention_scenarios)
    ]

    fig_flow = px.line(
        movement_df,
        x="Month_Date",
        y="Patients",
        color="Metric",
        line_dash="Scenario",
        markers=True,
        title="Demand, Throughput, and Net Movement",
    )

    fig_flow.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Patients",
        height=520,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_flow, use_container_width=True)

with tab4:
    st.subheader("Operational Levers Driving Additional Activity")

    st.markdown(
        """
        The theatre lever converts improved utilisation into additional cases.
        The outpatient levers convert better booking, scheduling, and template
        management into additional appointments.
        """
    )

    theatre_delivery_df = intervention_summary_df[
        intervention_summary_df["Scenario"].isin(list(EFFORT_LEVELS.keys()))
    ].copy()

    theatre_delivery_df["Current Utilisation"] = current_utilisation_pct
    theatre_delivery_df["Achieved Utilisation"] = (
        (current_utilisation_pct / 100)
        + theatre_delivery_df["Theatre Utilisation Gap"]
    ) * 100
    theatre_delivery_df["Target Utilisation"] = target_utilisation_pct

    utilisation_long_df = theatre_delivery_df.melt(
        id_vars="Scenario",
        value_vars=[
            "Current Utilisation",
            "Achieved Utilisation",
            "Target Utilisation",
        ],
        var_name="Utilisation Measure",
        value_name="Utilisation %",
    )

    theatre_col, op_col = st.columns(2)

    with theatre_col:
        fig_theatre_utilisation = px.line(
            utilisation_long_df,
            x="Scenario",
            y="Utilisation %",
            color="Utilisation Measure",
            markers=True,
            title="Theatre Utilisation Improvement by Delivery Scenario",
        )

        fig_theatre_utilisation.update_layout(
            template="plotly_white",
            xaxis_title="Delivery scenario",
            yaxis_title="Theatre utilisation %",
            height=430,
            margin=dict(l=20, r=20, t=70, b=20),
            yaxis_range=[
                max(0, min(current_utilisation_pct, target_utilisation_pct) - 5),
                min(100, max(current_utilisation_pct, target_utilisation_pct) + 5),
            ],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(fig_theatre_utilisation, use_container_width=True)

    outpatient_lever_df = intervention_summary_df[
        intervention_summary_df["Scenario"].isin(list(EFFORT_LEVELS.keys()))
    ][
        [
            "Scenario",
            "Outpatient Template Fill",
            "Outpatient DNA Reduction",
            "Outpatient PIFU",
            "Outpatient F:N Improvement",
        ]
    ].copy()

    outpatient_lever_df[
        [
            "Outpatient Template Fill",
            "Outpatient DNA Reduction",
            "Outpatient PIFU",
            "Outpatient F:N Improvement",
        ]
    ] = outpatient_lever_df[
        [
            "Outpatient Template Fill",
            "Outpatient DNA Reduction",
            "Outpatient PIFU",
            "Outpatient F:N Improvement",
        ]
    ] * horizon_months

    outpatient_lever_long_df = outpatient_lever_df.melt(
        id_vars="Scenario",
        var_name="Outpatient Lever",
        value_name="Additional Appointments",
    )

    outpatient_lever_long_df["Outpatient Lever"] = outpatient_lever_long_df[
        "Outpatient Lever"
    ].replace(
        {
            "Outpatient Template Fill": "Template fill",
            "Outpatient DNA Reduction": "DNA reduction",
            "Outpatient PIFU": "PIFU conversion",
            "Outpatient F:N Improvement": "F:N improvement",
        }
    )

    with op_col:
        fig_outpatient_levers = px.bar(
            outpatient_lever_long_df,
            x="Scenario",
            y="Additional Appointments",
            color="Outpatient Lever",
            title="Outpatient Additional Appointments by Lever",
        )

        fig_outpatient_levers.update_layout(
            template="plotly_white",
            xaxis_title="Delivery scenario",
            yaxis_title=f"Additional appointments over {horizon_months} months",
            height=430,
            margin=dict(l=20, r=20, t=70, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(fig_outpatient_levers, use_container_width=True)

    theatre_cases_df = theatre_delivery_df[
        ["Scenario", "Theatre Total Cases"]
    ].copy()

    fig_theatre_cases = px.bar(
        theatre_cases_df,
        x="Scenario",
        y="Theatre Total Cases",
        title="Additional Theatre Cases from Improved Utilisation",
    )

    fig_theatre_cases.update_layout(
        template="plotly_white",
        xaxis_title="Delivery scenario",
        yaxis_title=f"Additional cases over {horizon_months} months",
        height=430,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_theatre_cases, use_container_width=True)

    st.subheader("Intervention Lever Summary")

    lever_display_df = intervention_summary_df[
        intervention_summary_df["Scenario"].isin(selected_intervention_scenarios)
    ][
        [
            "Scenario",
            "Theatre Utilisation Gap",
            "Theatre Monthly Cases",
            "Theatre Total Cases",
            "Outpatient Template Fill",
            "Outpatient DNA Reduction",
            "Outpatient PIFU",
            "Outpatient F:N Improvement",
            "Outpatient Monthly Appointments",
            "Outpatient Total Appointments",
            "Monthly Effective RTT Activity",
            "Delivered Effective RTT Activity",
        ]
    ].copy()

    lever_display_df["Theatre Utilisation Gap"] = (
        lever_display_df["Theatre Utilisation Gap"] * 100
    ).round(2)

    numeric_cols = [
        col
        for col in lever_display_df.columns
        if col not in ["Scenario", "Theatre Utilisation Gap"]
    ]

    for col in numeric_cols:
        lever_display_df[col] = lever_display_df[col].round(0)

    st.dataframe(lever_display_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        theatre_basis_df = pd.DataFrame(
            {
                "Input": [
                    "Sessions/week",
                    "Session length",
                    "Current utilisation",
                    "Target utilisation",
                    "Average case duration",
                    "Active weeks",
                    "Theatre RTT impact",
                ],
                "Value": [
                    f"{sessions_per_week:,.1f}",
                    f"{session_minutes:,.0f} mins",
                    f"{current_utilisation_pct:.1f}%",
                    f"{target_utilisation_pct:.1f}%",
                    f"{avg_case_duration:,.0f} mins",
                    f"{active_weeks:,.0f}",
                    f"{theatre_rtt_conversion_pct:.0f}%",
                ],
            }
        )

        st.dataframe(theatre_basis_df, use_container_width=True)

    with col2:
        outpatient_basis_df = pd.DataFrame(
            {
                "Input": [
                    "Clinic sessions/week",
                    "Patients/session",
                    "Template fill change",
                    "DNA change",
                    "FU slots/week",
                    "New appointments/week",
                    "PIFU conversion",
                    "F:N improvement",
                    "Outpatient RTT impact",
                ],
                "Value": [
                    f"{clinic_sessions_per_week:,.1f}",
                    f"{patients_per_session:,.1f}",
                    f"{template_current_fill_pct:.1f}% to {template_target_fill_pct:.1f}%",
                    f"{current_dna_rate_pct:.1f}% to {target_dna_rate_pct:.1f}%",
                    f"{eligible_follow_up_per_week:,.0f}",
                    f"{eligible_new_per_week:,.0f}",
                    f"{pifu_conversion_pct:.1f}%",
                    f"{fn_ratio_improvement_pct:.1f}%",
                    f"{outpatient_rtt_conversion_pct:.0f}%",
                ],
            }
        )

        st.dataframe(outpatient_basis_df, use_container_width=True)

    if not theatre_capacity.empty:
        st.subheader("Measured Theatre Data Defaults")

        measured_df = pd.DataFrame(
            {
                "Metric": [
                    "Observation period",
                    "Sessions/week",
                    "Mean session length",
                    "Median session length",
                    "Average case duration",
                    "Utilisation",
                    "Completed cases",
                ],
                "Value": [
                    (
                        f"{format_month(theatre_capacity['Recent_Start_Date'])} "
                        f"to {format_month(theatre_capacity['Recent_End_Date'])}"
                    ),
                    f"{theatre_capacity['Sessions_Per_Week']:,.1f}",
                    f"{theatre_capacity['Average_Session_Minutes']:,.0f} mins",
                    f"{theatre_capacity['Median_Session_Minutes']:,.0f} mins",
                    (
                        f"{theatre_capacity['Average_Case_Duration_Minutes']:,.0f} "
                        "mins"
                    ),
                    f"{theatre_capacity['Utilisation']:.1%}",
                    f"{theatre_capacity['Completed_Cases']:,.0f}",
                ],
            }
        )

        st.dataframe(measured_df, use_container_width=True)

with tab5:
    st.subheader("Monthly Projection Detail")

    detail_df = projection_df[
        projection_df["Scenario"].isin(["Do Nothing"] + selected_intervention_scenarios)
    ][
        [
            "Scenario",
            "Month_Date",
            "Month Number",
            "Opening Backlog",
            "Demand",
            "Baseline Throughput",
            "Additional Activity",
            "Adjusted Throughput",
            "Net Change",
            "Closing Backlog",
            "0-18 Weeks",
            "18-52 Weeks",
            "52+ Weeks",
            "% Within 18 Weeks",
            "% 52+ Weeks",
        ]
    ].copy()

    detail_df["Month_Date"] = detail_df["Month_Date"].dt.strftime("%B %Y")

    round_cols = [
        "Opening Backlog",
        "Demand",
        "Baseline Throughput",
        "Additional Activity",
        "Adjusted Throughput",
        "Net Change",
        "Closing Backlog",
        "0-18 Weeks",
        "18-52 Weeks",
        "52+ Weeks",
    ]

    for col in round_cols:
        detail_df[col] = detail_df[col].round(0)

    detail_df["% Within 18 Weeks"] = (
        detail_df["% Within 18 Weeks"] * 100
    ).round(1)
    detail_df["% 52+ Weeks"] = (detail_df["% 52+ Weeks"] * 100).round(1)

    st.dataframe(detail_df, use_container_width=True)

    with st.expander("Model notes"):
        st.markdown(
            """
The do-nothing line uses a weighted recent baseline from RTT backlog, additions, and completions.
The conservative do-nothing option uses observed RTT additions and completions, so it does not assume recent backlog reductions continue indefinitely.
Historical seasonality is lightly shrunk and capped before being applied to future demand and throughput.
Intervention delivery is modelled as additional effective RTT activity above baseline.
The 52+ forecast ages the actual weekly RTT wait bands from the latest month, rather than ageing one broad 18-52 week cohort.
            """
        )
