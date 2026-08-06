import math

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.rtt_loader import load_all_rtt_files
from src.data.intervention_loader import load_intervention_data, validate_intervention_data

from src.transforms.rtt_transform import (
    filter_pah_incomplete,
    add_wait_band_metrics,
    summarise_rtt_by_month,
)

from src.transforms.intervention_transform import (
    enrich_intervention_logic,
    prepare_intervention_display,
    prepare_logic_chain_display,
)


st.set_page_config(
    page_title="RTT Intervention Recovery Modelling",
    page_icon="📈",
    layout="wide",
)

st.title("📈 RTT Intervention Recovery Modelling")

st.caption(
    "Model the future RTT waiting list trajectory using observed backlog trends, phased intervention recovery assumptions, and cost/value impact."
)


WEEKS_PER_MONTH = 4.33


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def month_label_to_date(month_series: pd.Series) -> pd.Series:
    return pd.to_datetime(month_series, format="%B %Y", errors="coerce")


def safe_number(value, default=0.0) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        return default
    return float(result)


def calculate_actual_history(backlog_df: pd.DataFrame) -> pd.DataFrame:
    actual = backlog_df.copy()
    actual["Month_Date"] = month_label_to_date(actual["Month"])
    actual = actual.dropna(subset=["Month_Date"]).sort_values("Month_Date")

    if "waiting_18_52_total" not in actual.columns:
        actual["waiting_18_52_total"] = (
            actual["Total"]
            - actual["waiting_0_18_total"]
            - actual["waiting_52_plus_total"]
        )

    actual["% Within 18 Weeks"] = actual["waiting_0_18_total"] / actual["Total"]
    actual["% 52+ Weeks"] = actual["waiting_52_plus_total"] / actual["Total"]

    actual["Scenario"] = "Actual"
    actual["Closing Backlog"] = actual["Total"]
    actual["0–18 Weeks"] = actual["waiting_0_18_total"]
    actual["18–52 Weeks"] = actual["waiting_18_52_total"]
    actual["52+ Weeks"] = actual["waiting_52_plus_total"]
    actual["Monthly Backlog Change"] = actual["Closing Backlog"].diff()

    return actual[
        [
            "Month_Date",
            "Scenario",
            "Closing Backlog",
            "0–18 Weeks",
            "18–52 Weeks",
            "52+ Weeks",
            "% Within 18 Weeks",
            "% 52+ Weeks",
            "Monthly Backlog Change",
        ]
    ]


def weighted_average(values: pd.Series) -> float:
    values = values.dropna()

    if values.empty:
        return 0.0

    weights = list(range(1, len(values) + 1))
    return (values * weights).sum() / sum(weights)


def calculate_smoothed_monthly_change(
    actual_df: pd.DataFrame,
    baseline_months: int,
    smoothing_method: str,
) -> float:
    work = actual_df.sort_values("Month_Date").copy()
    work["Monthly_Change"] = work["Closing Backlog"].diff()

    recent_changes = work["Monthly_Change"].tail(baseline_months)

    if smoothing_method == "Simple average":
        return recent_changes.mean()

    if smoothing_method == "Weighted recent average":
        return weighted_average(recent_changes)

    if smoothing_method == "Median":
        return recent_changes.median()

    return recent_changes.mean()


def get_seasonal_factor(month_number: int, use_seasonality: bool) -> float:
    if not use_seasonality:
        return 1.0

    seasonal_factors = {
        1: 1.10,
        2: 1.05,
        3: 0.95,
        4: 0.95,
        5: 0.95,
        6: 1.00,
        7: 1.05,
        8: 1.05,
        9: 1.00,
        10: 1.00,
        11: 1.05,
        12: 1.15,
    }

    return seasonal_factors.get(month_number, 1.0)


def ramp_factor(month_number: int, ramp_months: int) -> float:
    if ramp_months <= 0:
        return 1.0

    if month_number >= ramp_months:
        return 1.0

    return month_number / ramp_months


def simulate_recovery_trajectory(
    start_backlog: float,
    start_0_18: float,
    start_18_52: float,
    start_52_plus: float,
    latest_actual_date: pd.Timestamp,
    horizon_months: int,
    smoothed_monthly_change: float,
    use_seasonality: bool,
    extra_cases_per_week: float,
    active_weeks: int,
    ramp_months: int,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []

    current_backlog = start_backlog
    current_0_18 = start_0_18
    current_18_52 = start_18_52
    current_52_plus = start_52_plus

    active_months = math.ceil(active_weeks / WEEKS_PER_MONTH)

    for month_number in range(1, horizon_months + 1):
        month_date = latest_actual_date + pd.DateOffset(months=month_number)

        seasonal_factor = get_seasonal_factor(
            month_number=month_date.month,
            use_seasonality=use_seasonality,
        )

        do_nothing_change = smoothed_monthly_change * seasonal_factor

        if month_number <= active_months:
            monthly_intervention_activity = (
                extra_cases_per_week
                * WEEKS_PER_MONTH
                * ramp_factor(month_number, ramp_months)
            )
        else:
            monthly_intervention_activity = 0.0

        net_change = do_nothing_change - monthly_intervention_activity

        opening_backlog = current_backlog
        closing_backlog = max(opening_backlog + net_change, 0)

        if opening_backlog > 0:
            scale_factor = closing_backlog / opening_backlog
        else:
            scale_factor = 1

        current_0_18 = max(current_0_18 * scale_factor, 0)
        current_18_52 = max(current_18_52 * scale_factor, 0)
        current_52_plus = max(current_52_plus * scale_factor, 0)
        current_backlog = closing_backlog

        rows.append(
            {
                "Scenario": scenario_name,
                "Month Number": month_number,
                "Month_Date": month_date,
                "Opening Backlog": opening_backlog,
                "Do Nothing Monthly Change": do_nothing_change,
                "Seasonal Factor": seasonal_factor,
                "Ramp Factor": ramp_factor(month_number, ramp_months)
                if month_number <= active_months
                else 0,
                "Additional Activity": monthly_intervention_activity,
                "Net Change": net_change,
                "Closing Backlog": closing_backlog,
                "0–18 Weeks": current_0_18,
                "18–52 Weeks": current_18_52,
                "52+ Weeks": current_52_plus,
                "% Within 18 Weeks": current_0_18 / closing_backlog
                if closing_backlog > 0
                else 0,
                "% 52+ Weeks": current_52_plus / closing_backlog
                if closing_backlog > 0
                else 0,
            }
        )

    return pd.DataFrame(rows)


def create_indexed_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df.copy()

    indexed_parts = []

    actual_df = work[work["Scenario"] == "Actual"].copy()
    future_df = work[work["Scenario"] != "Actual"].copy()

    if actual_df.empty:
        return pd.DataFrame()

    latest_actual_date = actual_df["Month_Date"].max()
    latest_actual_value = actual_df.loc[
        actual_df["Month_Date"] == latest_actual_date,
        value_col,
    ].iloc[0]

    if latest_actual_value == 0:
        latest_actual_value = 1

    actual_df["Indexed Value"] = (actual_df[value_col] / latest_actual_value) * 100
    indexed_parts.append(actual_df)

    for scenario in future_df["Scenario"].dropna().unique():
        scenario_df = future_df[future_df["Scenario"] == scenario].copy()

        bridge_row = actual_df[actual_df["Month_Date"] == latest_actual_date].copy()
        bridge_row["Scenario"] = scenario
        bridge_row["Indexed Value"] = 100

        scenario_df["Indexed Value"] = (
            scenario_df[value_col] / latest_actual_value
        ) * 100

        indexed_parts.append(bridge_row)
        indexed_parts.append(scenario_df)

    return pd.concat(indexed_parts, ignore_index=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

try:
    raw_df = load_all_rtt_files()
    pah_incomplete_df = filter_pah_incomplete(raw_df)
    metric_df = add_wait_band_metrics(pah_incomplete_df)
    backlog_df = summarise_rtt_by_month(metric_df)

except Exception as e:
    st.error(f"Error loading RTT data: {e}")
    st.stop()

if backlog_df.empty:
    st.warning("No RTT backlog data available.")
    st.stop()

try:
    intervention_library_df = load_intervention_data()
    intervention_library_df = enrich_intervention_logic(intervention_library_df)

except Exception as e:
    st.error(f"Error loading intervention model: {e}")
    st.stop()

for warning in validate_intervention_data(intervention_library_df):
    st.warning(warning)


# ---------------------------------------------------------
# Prepare baseline
# ---------------------------------------------------------

actual_history_df = calculate_actual_history(backlog_df)
latest_actual = actual_history_df.iloc[-1]

latest_actual_date = latest_actual["Month_Date"]
start_backlog = float(latest_actual["Closing Backlog"])
start_0_18 = float(latest_actual["0–18 Weeks"])
start_18_52 = float(latest_actual["18–52 Weeks"])
start_52_plus = float(latest_actual["52+ Weeks"])


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Baseline Trend")

baseline_months = st.sidebar.slider(
    "Months used to calculate recent backlog trend",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
)

smoothing_method = st.sidebar.selectbox(
    "Smoothing method",
    [
        "Weighted recent average",
        "Simple average",
        "Median",
    ],
)

use_seasonality = st.sidebar.checkbox(
    "Apply simple seasonality adjustment",
    value=True,
)

horizon_months = st.sidebar.slider(
    "Scenario horizon",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
)


st.sidebar.header("Intervention Lever")

modelable_df = intervention_library_df[intervention_library_df["Is_Modelable"]].copy()

group_options = sorted(modelable_df["Group_Theme"].dropna().unique())

selected_group = st.sidebar.selectbox(
    "Intervention group / theme",
    ["All"] + group_options,
)

if selected_group != "All":
    lever_df = modelable_df[modelable_df["Group_Theme"] == selected_group].copy()
else:
    lever_df = modelable_df.copy()

selected_intervention = st.sidebar.selectbox(
    "Select intervention",
    lever_df["Intervention"].dropna().tolist(),
)

selected_row = lever_df[lever_df["Intervention"] == selected_intervention].iloc[0]

default_cases_per_week = float(selected_row["Additional_Cases_Per_Week"])
default_weeks_to_recover = int(selected_row["Weeks_To_Recover"])

extra_cases_per_week = st.sidebar.number_input(
    "Additional cases / slots per week",
    min_value=0.0,
    max_value=5000.0,
    value=default_cases_per_week,
    step=1.0,
)

active_weeks = st.sidebar.slider(
    "Weeks intervention is active",
    min_value=1,
    max_value=max(4, int(horizon_months * WEEKS_PER_MONTH)),
    value=min(default_weeks_to_recover, max(4, int(horizon_months * WEEKS_PER_MONTH))),
    step=1,
)

ramp_months = st.sidebar.slider(
    "Ramp-up months",
    min_value=0,
    max_value=6,
    value=3,
    step=1,
    help="Models phased adoption of the intervention rather than full benefit from month one.",
)

smoothed_monthly_change = calculate_smoothed_monthly_change(
    actual_df=actual_history_df,
    baseline_months=baseline_months,
    smoothing_method=smoothing_method,
)

monthly_steady_state_activity = extra_cases_per_week * WEEKS_PER_MONTH
total_extra_cases = extra_cases_per_week * active_weeks


# ---------------------------------------------------------
# Cost fields
# ---------------------------------------------------------

investment_required = safe_number(selected_row.get("Investment_Required", 0))
annual_financial_benefit = safe_number(
    selected_row.get("Annual_Financial_Benefit", 0)
)
net_position = safe_number(
    selected_row.get("Net_Position", annual_financial_benefit - investment_required)
)

if net_position == 0 and (annual_financial_benefit != 0 or investment_required != 0):
    net_position = annual_financial_benefit - investment_required

wte_required_peak = safe_number(selected_row.get("WTE_Required_Peak", 0))
wte_released = safe_number(selected_row.get("WTE_Released_Steady_State", 0))


# ---------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------

do_nothing_df = simulate_recovery_trajectory(
    start_backlog=start_backlog,
    start_0_18=start_0_18,
    start_18_52=start_18_52,
    start_52_plus=start_52_plus,
    latest_actual_date=latest_actual_date,
    horizon_months=horizon_months,
    smoothed_monthly_change=smoothed_monthly_change,
    use_seasonality=use_seasonality,
    extra_cases_per_week=0,
    active_weeks=0,
    ramp_months=0,
    scenario_name="Do Nothing",
)

intervention_df_model = simulate_recovery_trajectory(
    start_backlog=start_backlog,
    start_0_18=start_0_18,
    start_18_52=start_18_52,
    start_52_plus=start_52_plus,
    latest_actual_date=latest_actual_date,
    horizon_months=horizon_months,
    smoothed_monthly_change=smoothed_monthly_change,
    use_seasonality=use_seasonality,
    extra_cases_per_week=extra_cases_per_week,
    active_weeks=active_weeks,
    ramp_months=ramp_months,
    scenario_name="Intervention Applied",
)

projection_df = pd.concat([do_nothing_df, intervention_df_model], ignore_index=True)

plot_backlog_df = pd.concat(
    [
        actual_history_df[["Month_Date", "Scenario", "Closing Backlog"]],
        projection_df[["Month_Date", "Scenario", "Closing Backlog"]],
    ],
    ignore_index=True,
)

plot_52_df = pd.concat(
    [
        actual_history_df[["Month_Date", "Scenario", "52+ Weeks"]],
        projection_df[["Month_Date", "Scenario", "52+ Weeks"]],
    ],
    ignore_index=True,
)

plot_18_df = pd.concat(
    [
        actual_history_df[["Month_Date", "Scenario", "% Within 18 Weeks"]],
        projection_df[["Month_Date", "Scenario", "% Within 18 Weeks"]],
    ],
    ignore_index=True,
)

indexed_backlog_df = create_indexed_series(
    pd.concat(
        [
            actual_history_df[["Month_Date", "Scenario", "Closing Backlog"]],
            projection_df[["Month_Date", "Scenario", "Closing Backlog"]],
        ],
        ignore_index=True,
    ),
    value_col="Closing Backlog",
)

do_nothing_end = do_nothing_df.iloc[-1]
intervention_end = intervention_df_model.iloc[-1]

backlog_impact = intervention_end["Closing Backlog"] - do_nothing_end["Closing Backlog"]
long_wait_impact = intervention_end["52+ Weeks"] - do_nothing_end["52+ Weeks"]
within_18_impact = (
    intervention_end["% Within 18 Weeks"] - do_nothing_end["% Within 18 Weeks"]
)


# ---------------------------------------------------------
# Intervention impact calculations
# ---------------------------------------------------------

impact_df = do_nothing_df[
    [
        "Month_Date",
        "Month Number",
        "Closing Backlog",
        "52+ Weeks",
        "Net Change",
    ]
].rename(
    columns={
        "Closing Backlog": "Do Nothing Backlog",
        "52+ Weeks": "Do Nothing 52+",
        "Net Change": "Do Nothing Monthly Change",
    }
)

intervention_compare_df = intervention_df_model[
    [
        "Month_Date",
        "Closing Backlog",
        "52+ Weeks",
        "Net Change",
        "Additional Activity",
    ]
].rename(
    columns={
        "Closing Backlog": "Intervention Backlog",
        "52+ Weeks": "Intervention 52+",
        "Net Change": "Intervention Monthly Change",
    }
)

impact_df = impact_df.merge(
    intervention_compare_df,
    on="Month_Date",
    how="left",
)

impact_df["Backlog Avoided"] = (
    impact_df["Do Nothing Backlog"] - impact_df["Intervention Backlog"]
)

impact_df["52+ Avoided"] = (
    impact_df["Do Nothing 52+"] - impact_df["Intervention 52+"]
)

velocity_df = impact_df[
    [
        "Month_Date",
        "Do Nothing Monthly Change",
        "Intervention Monthly Change",
    ]
].melt(
    id_vars="Month_Date",
    var_name="Scenario",
    value_name="Monthly Backlog Movement",
)

velocity_df["Scenario"] = velocity_df["Scenario"].replace(
    {
        "Do Nothing Monthly Change": "Do Nothing",
        "Intervention Monthly Change": "Intervention Applied",
    }
)

backlog_avoided = impact_df["Backlog Avoided"].iloc[-1]

cost_per_extra_case = (
    investment_required / total_extra_cases if total_extra_cases > 0 else 0
)

cost_per_backlog_avoided = (
    investment_required / backlog_avoided if backlog_avoided > 0 else 0
)

benefit_per_extra_case = (
    annual_financial_benefit / total_extra_cases if total_extra_cases > 0 else 0
)


# ---------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------

st.subheader("Current State → Smoothed Do-Nothing → Intervention Applied")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Actual Month", latest_actual_date.strftime("%B %Y"))
c2.metric("Current Backlog", f"{start_backlog:,.0f}")
c3.metric("Current 52+ Waits", f"{start_52_plus:,.0f}")
c4.metric("Smoothed Monthly Change", f"{smoothed_monthly_change:+,.0f}")

c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Intervention End Backlog",
    f"{intervention_end['Closing Backlog']:,.0f}",
    delta=f"{backlog_impact:+,.0f} vs do nothing",
    delta_color="inverse",
)

c6.metric(
    "Intervention 52+ Waits",
    f"{intervention_end['52+ Weeks']:,.0f}",
    delta=f"{long_wait_impact:+,.0f} vs do nothing",
    delta_color="inverse",
)

c7.metric(
    "Intervention % Within 18 Weeks",
    f"{intervention_end['% Within 18 Weeks']:.1%}",
    delta=f"{within_18_impact:+.1%} vs do nothing",
)

c8.metric("Backlog Avoided", f"{backlog_avoided:,.0f}")

st.markdown(
    f"""
    **Selected intervention:** {selected_intervention}

    **Lever applied:** {extra_cases_per_week:,.1f} additional cases / slots per week for
    {active_weeks:,} weeks, with a **{ramp_months}-month ramp-up**.

    The do-nothing trajectory is based on a **{smoothing_method.lower()}** of the latest
    **{baseline_months} months** of observed backlog movement.
    """
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Indexed Trajectory",
        "Raw Backlog View",
        "Intervention Impact",
        "52+ and Performance",
        "Target Contribution",
        "Cost & Value",
        "Scenario Table",
    ]
)


# ---------------------------------------------------------
# Tab 1: Indexed trajectory hero view
# ---------------------------------------------------------

with tab1:
    st.subheader("Indexed Waiting List Trajectory")

    st.markdown(
        """
        This is the primary trajectory view. It indexes the latest actual backlog month to **100**
        so that the do-nothing and intervention-applied futures can be compared more clearly.
        """
    )

    fig_indexed = px.line(
        indexed_backlog_df,
        x="Month_Date",
        y="Indexed Value",
        color="Scenario",
        markers=True,
        title="Indexed Waiting List Trajectory: Actual vs Do Nothing vs Intervention",
    )

    fig_indexed.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Indexed backlog, latest actual = 100",
        height=720,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig_indexed.update_traces(
        hovertemplate=(
            "Month: %{x|%b %Y}"
            "<br>Indexed backlog: %{y:.1f}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig_indexed, use_container_width=True)

    st.info(
        """
        Interpretation: values above 100 mean the backlog is higher than the latest actual position.
        Values below 100 mean the backlog is lower. This makes the difference between do-nothing and
        intervention-applied trajectories easier to see than a raw backlog chart.
        """
    )


# ---------------------------------------------------------
# Tab 2: Raw backlog view
# ---------------------------------------------------------

with tab2:
    st.subheader("Raw Waiting List Trajectory")

    fig_raw = px.line(
        plot_backlog_df,
        x="Month_Date",
        y="Closing Backlog",
        color="Scenario",
        markers=True,
        title="Raw Waiting List Trajectory",
    )

    fig_raw.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Closing backlog",
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

    fig_raw.update_traces(
        hovertemplate="Month: %{x|%b %Y}<br>Backlog: %{y:,.0f}<extra></extra>"
    )

    st.plotly_chart(fig_raw, use_container_width=True)

    end_state_df = pd.DataFrame(
        {
            "Metric": [
                "Closing Backlog",
                "52+ Weeks",
                "% Within 18 Weeks",
            ],
            "Do Nothing": [
                do_nothing_end["Closing Backlog"],
                do_nothing_end["52+ Weeks"],
                do_nothing_end["% Within 18 Weeks"],
            ],
            "Intervention Applied": [
                intervention_end["Closing Backlog"],
                intervention_end["52+ Weeks"],
                intervention_end["% Within 18 Weeks"],
            ],
        }
    )

    end_state_df["Difference"] = (
        end_state_df["Intervention Applied"] - end_state_df["Do Nothing"]
    )

    st.subheader("End-State Comparison")
    st.dataframe(end_state_df, use_container_width=True)


# ---------------------------------------------------------
# Tab 3: Intervention impact
# ---------------------------------------------------------

with tab3:
    st.subheader("Cumulative Backlog Avoided vs Do Nothing")

    fig_avoided = px.area(
        impact_df,
        x="Month_Date",
        y="Backlog Avoided",
        title="Modelled Recovery Benefit: Backlog Avoided vs Do Nothing",
    )

    fig_avoided.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Backlog avoided",
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    fig_avoided.update_traces(
        hovertemplate=(
            "Month: %{x|%b %Y}"
            "<br>Backlog avoided: %{y:,.0f}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(fig_avoided, use_container_width=True)

    st.info(
        """
        This graph shows the cumulative difference between the do-nothing backlog and the
        intervention-applied backlog. It makes the intervention impact visible even when the
        raw backlog lines look close together.
        """
    )

    st.subheader("Monthly Backlog Movement")

    fig_velocity = px.line(
        velocity_df,
        x="Month_Date",
        y="Monthly Backlog Movement",
        color="Scenario",
        markers=True,
        title="Recovery Velocity: Monthly Backlog Movement",
    )

    fig_velocity.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Monthly backlog movement",
        height=550,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(fig_velocity, use_container_width=True)

    st.subheader("Intervention Ramp-Up and Monthly Activity")

    activity_df = intervention_df_model[
        [
            "Month_Date",
            "Month Number",
            "Ramp Factor",
            "Additional Activity",
            "Do Nothing Monthly Change",
            "Net Change",
        ]
    ].copy()

    activity_long = activity_df.melt(
        id_vars=["Month_Date", "Month Number"],
        value_vars=[
            "Additional Activity",
            "Do Nothing Monthly Change",
            "Net Change",
        ],
        var_name="Metric",
        value_name="Value",
    )

    fig_activity = px.line(
        activity_long,
        x="Month_Date",
        y="Value",
        color="Metric",
        markers=True,
        title="Monthly Trend and Intervention Activity",
    )

    fig_activity.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Monthly movement",
        height=550,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_activity, use_container_width=True)

    fig_ramp = px.bar(
        activity_df,
        x="Month_Date",
        y="Ramp Factor",
        title="Intervention Ramp-Up Profile",
    )

    fig_ramp.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Ramp factor",
        height=430,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_ramp, use_container_width=True)


# ---------------------------------------------------------
# Tab 4: 52+ and performance
# ---------------------------------------------------------

with tab4:
    st.subheader("52+ Waits")

    fig_52 = px.line(
        plot_52_df,
        x="Month_Date",
        y="52+ Weeks",
        color="Scenario",
        markers=True,
        title="52+ Waits Trajectory",
    )

    fig_52.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="52+ waits",
        height=600,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_52, use_container_width=True)

    st.subheader("18-Week Performance")

    perf_df = plot_18_df.copy()
    perf_df["% Within 18 Weeks"] = perf_df["% Within 18 Weeks"] * 100

    fig_perf = px.line(
        perf_df,
        x="Month_Date",
        y="% Within 18 Weeks",
        color="Scenario",
        markers=True,
        title="% Within 18 Weeks Trajectory",
    )

    fig_perf.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="% within 18 weeks",
        height=600,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_perf, use_container_width=True)


# ---------------------------------------------------------
# Tab 5: Target contribution
# ---------------------------------------------------------

with tab5:
    st.subheader("Target Contribution")

    target_df = pd.DataFrame(
        {
            "Field": [
                "Intervention",
                "Group / Theme",
                "Current Baseline",
                "Benchmark",
                "PAH Target",
                "Assumption Detail",
                "Additional Cases / Week",
                "Weeks Active",
                "Ramp-Up Months",
                "Total Extra Cases",
                "Backlog Avoided vs Do Nothing",
                "52+ Avoided vs Do Nothing",
                "18-Week Performance Difference",
            ],
            "Value": [
                selected_intervention,
                selected_row.get("Group_Theme", ""),
                selected_row.get("Current_Baseline", ""),
                selected_row.get("Benchmark", ""),
                selected_row.get("PAH_Target", ""),
                selected_row.get("Assumption_Detail", ""),
                f"{extra_cases_per_week:,.1f}",
                f"{active_weeks:,.0f}",
                f"{ramp_months:,.0f}",
                f"{total_extra_cases:,.0f}",
                f"{backlog_avoided:,.0f}",
                f"{impact_df['52+ Avoided'].iloc[-1]:,.0f}",
                f"{within_18_impact:+.1%}",
            ],
        }
    )

    st.dataframe(target_df, use_container_width=True)

    selected_intervention_df = pd.DataFrame([selected_row]).copy()
    selected_intervention_df["Additional_Cases_Per_Week"] = extra_cases_per_week
    selected_intervention_df["Weeks_To_Recover"] = active_weeks
    selected_intervention_df["Total_Additional_Cases"] = total_extra_cases
    selected_intervention_df = enrich_intervention_logic(selected_intervention_df)

    st.subheader("Intervention Logic")
    st.dataframe(
        prepare_logic_chain_display(selected_intervention_df),
        use_container_width=True,
    )

    st.subheader("Modelable Intervention Library")
    st.dataframe(
        prepare_intervention_display(modelable_df),
        use_container_width=True,
    )


# ---------------------------------------------------------
# Tab 6: Cost & Value
# ---------------------------------------------------------

with tab6:
    st.subheader("Cost & Value Assessment")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Investment Required", f"£{investment_required:,.0f}")
    c2.metric("Annual Financial Benefit", f"£{annual_financial_benefit:,.0f}")
    c3.metric("Net Position", f"£{net_position:,.0f}")
    c4.metric("Cost / Extra Case", f"£{cost_per_extra_case:,.0f}")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Cost / Backlog Avoided", f"£{cost_per_backlog_avoided:,.0f}")
    c6.metric("Benefit / Extra Case", f"£{benefit_per_extra_case:,.0f}")
    c7.metric("Peak WTE Required", f"{wte_required_peak:,.1f}")
    c8.metric("WTE Released", f"{wte_released:,.1f}")

    value_df = pd.DataFrame(
        {
            "Measure": [
                "Intervention",
                "Investment Required",
                "Annual Financial Benefit",
                "Net Position",
                "Total Extra Cases",
                "Backlog Avoided",
                "Cost per Extra Case",
                "Cost per Backlog Avoided",
                "Benefit per Extra Case",
                "Peak WTE Required",
                "WTE Released Steady State",
            ],
            "Value": [
                selected_intervention,
                f"£{investment_required:,.0f}",
                f"£{annual_financial_benefit:,.0f}",
                f"£{net_position:,.0f}",
                f"{total_extra_cases:,.0f}",
                f"{backlog_avoided:,.0f}",
                f"£{cost_per_extra_case:,.0f}",
                f"£{cost_per_backlog_avoided:,.0f}",
                f"£{benefit_per_extra_case:,.0f}",
                f"{wte_required_peak:,.1f}",
                f"{wte_released:,.1f}",
            ],
        }
    )

    st.dataframe(value_df, use_container_width=True)

    cost_chart_df = pd.DataFrame(
        {
            "Metric": [
                "Investment Required",
                "Annual Financial Benefit",
                "Net Position",
            ],
            "Value": [
                investment_required,
                annual_financial_benefit,
                net_position,
            ],
        }
    )

    fig_cost = px.bar(
        cost_chart_df,
        x="Metric",
        y="Value",
        title="Cost and Benefit Summary",
    )

    fig_cost.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title="£",
        height=450,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(fig_cost, use_container_width=True)

    st.info(
        """
        This view adds a value-for-money lens to the selected intervention.

        It shows the financial cost and benefit alongside modelled recovery impact.
        The key interpretation is not just whether the intervention reduces backlog,
        but whether the scale of impact appears proportionate to the investment required.
        """
    )


# ---------------------------------------------------------
# Tab 7: Scenario table
# ---------------------------------------------------------

with tab7:
    st.subheader("Scenario Detail Table")

    display_df = projection_df[
        [
            "Scenario",
            "Month_Date",
            "Month Number",
            "Opening Backlog",
            "Do Nothing Monthly Change",
            "Seasonal Factor",
            "Ramp Factor",
            "Additional Activity",
            "Net Change",
            "Closing Backlog",
            "0–18 Weeks",
            "18–52 Weeks",
            "52+ Weeks",
            "% Within 18 Weeks",
            "% 52+ Weeks",
        ]
    ].copy()

    display_df["Month_Date"] = display_df["Month_Date"].dt.strftime("%B %Y")

    st.dataframe(display_df, use_container_width=True)