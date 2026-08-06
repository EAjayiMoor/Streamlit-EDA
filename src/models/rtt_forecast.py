import math

import pandas as pd


WEEKS_PER_MONTH = 4.3


def month_label_to_date(month_series: pd.Series) -> pd.Series:
    return pd.to_datetime(month_series, format="%B %Y", errors="coerce")


def weighted_average(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return 0.0

    weights = pd.Series(range(1, len(clean) + 1), index=clean.index)
    return float((clean * weights).sum() / weights.sum())


def prepare_rtt_history(
    backlog_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    completion_df: pd.DataFrame,
) -> pd.DataFrame:
    history = backlog_df.copy()
    history["Month_Date"] = month_label_to_date(history["Month"])
    history = history.dropna(subset=["Month_Date"]).sort_values("Month_Date")

    history["waiting_18_52_total"] = (
        history["Total"]
        - history["waiting_0_18_total"]
        - history["waiting_52_plus_total"]
    ).clip(lower=0)

    history = history.merge(demand_df, on="Month", how="left")
    history = history.merge(completion_df, on="Month", how="left")

    fill_cols = [
        "additions_total",
        "admitted_completed_total",
        "nonadmitted_completed_total",
        "completed_total",
    ]

    for col in fill_cols:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)

    history["Monthly_Backlog_Change"] = history["Total"].diff()
    history["Net_Flow"] = history["additions_total"] - history["completed_total"]
    history["pct_0_18"] = history["waiting_0_18_total"] / history["Total"]
    history["pct_52_plus"] = history["waiting_52_plus_total"] / history["Total"]

    return history.reset_index(drop=True)


def calculate_baseline_inputs(
    history_df: pd.DataFrame,
    baseline_months: int,
    backlog_change_weight: float = 0.65,
    baseline_mode: str = "Conservative observed flow",
) -> dict:
    recent = history_df.sort_values("Month_Date").tail(baseline_months).copy()

    if recent.empty:
        return {
            "monthly_demand": 0.0,
            "monthly_throughput": 0.0,
            "monthly_backlog_change": 0.0,
            "monthly_net_flow": 0.0,
            "blended_monthly_change": 0.0,
        }

    weighted_demand = weighted_average(recent["additions_total"])
    weighted_completion = weighted_average(recent["completed_total"])
    weighted_backlog_change = weighted_average(recent["Monthly_Backlog_Change"])
    weighted_net_flow = weighted_average(recent["Net_Flow"])

    if baseline_mode == "Recent backlog trend":
        baseline_change = weighted_backlog_change
    elif baseline_mode == "Blended trend and flow":
        baseline_change = (
            backlog_change_weight * weighted_backlog_change
            + (1 - backlog_change_weight) * weighted_net_flow
        )
    else:
        baseline_change = weighted_net_flow

    flow_implied_completion = max(weighted_demand - baseline_change, 0)

    if baseline_mode == "Blended trend and flow":
        monthly_throughput = (weighted_completion + flow_implied_completion) / 2
    else:
        monthly_throughput = flow_implied_completion

    return {
        "monthly_demand": weighted_demand,
        "monthly_throughput": monthly_throughput,
        "monthly_backlog_change": weighted_backlog_change,
        "monthly_net_flow": weighted_net_flow,
        "blended_monthly_change": weighted_demand - monthly_throughput,
        "baseline_mode": baseline_mode,
    }


def calculate_seasonal_index(
    history_df: pd.DataFrame,
    value_col: str,
    shrinkage: float = 0.35,
    floor: float = 0.85,
    ceiling: float = 1.15,
) -> dict:
    work = history_df.dropna(subset=["Month_Date"]).copy()

    if value_col not in work.columns or len(work) < 18:
        return {month: 1.0 for month in range(1, 13)}

    overall = pd.to_numeric(work[value_col], errors="coerce").mean()

    if not overall or pd.isna(overall):
        return {month: 1.0 for month in range(1, 13)}

    work["Month_Number"] = work["Month_Date"].dt.month
    raw_index = work.groupby("Month_Number")[value_col].mean() / overall

    seasonal_index = {}

    for month in range(1, 13):
        raw_value = float(raw_index.get(month, 1.0))
        shrunk_value = 1 + ((raw_value - 1) * shrinkage)
        seasonal_index[month] = min(max(shrunk_value, floor), ceiling)

    return seasonal_index


def remove_activity_from_bands(
    bands: dict,
    removals: float,
    strategy: str,
) -> dict:
    updated = {key: max(float(value), 0) for key, value in bands.items()}
    removals = max(float(removals), 0)
    total = sum(updated.values())

    if total <= 0 or removals <= 0:
        return updated

    removals = min(removals, total)

    if strategy == "Proportional":
        for band in updated:
            updated[band] -= removals * (updated[band] / total)

    elif strategy in ["Longest waits first", "52+ first"]:
        for band in ["52_plus", "18_52", "0_18"]:
            take = min(updated[band], removals)
            updated[band] -= take
            removals -= take

            if removals <= 0:
                break

    elif strategy == "18-52 first":
        for band in ["18_52", "52_plus", "0_18"]:
            take = min(updated[band], removals)
            updated[band] -= take
            removals -= take

            if removals <= 0:
                break

    else:
        for band in updated:
            updated[band] -= removals * (updated[band] / total)

    return {key: max(value, 0) for key, value in updated.items()}


def simulate_rtt_forecast(
    history_df: pd.DataFrame,
    horizon_months: int,
    baseline_months: int,
    additional_activity_by_month: list[float] | None = None,
    scenario_name: str = "Do Nothing",
    use_seasonality: bool = True,
    intervention_targeting: str = "Longest waits first",
    baseline_mode: str = "Conservative observed flow",
) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    history = history_df.sort_values("Month_Date").copy()
    latest = history.iloc[-1]
    baseline = calculate_baseline_inputs(
        history,
        baseline_months,
        baseline_mode=baseline_mode,
    )

    demand_index = calculate_seasonal_index(history, "additions_total")
    throughput_index = calculate_seasonal_index(history, "completed_total")

    activity_by_month = additional_activity_by_month or [0.0] * horizon_months
    activity_by_month = list(activity_by_month)[:horizon_months]

    if len(activity_by_month) < horizon_months:
        activity_by_month.extend([0.0] * (horizon_months - len(activity_by_month)))

    bands = {
        "0_18": float(latest["waiting_0_18_total"]),
        "18_52": float(latest["waiting_18_52_total"]),
        "52_plus": float(latest["waiting_52_plus_total"]),
    }

    age_0_18_to_18_52_rate = WEEKS_PER_MONTH / 18
    age_18_52_to_52_plus_rate = WEEKS_PER_MONTH / 34

    records = []

    for month_number in range(1, horizon_months + 1):
        month_date = latest["Month_Date"] + pd.DateOffset(months=month_number)
        month_key = month_date.month

        demand_factor = demand_index[month_key] if use_seasonality else 1.0
        throughput_factor = throughput_index[month_key] if use_seasonality else 1.0

        demand = max(baseline["monthly_demand"] * demand_factor, 0)
        baseline_throughput = max(
            baseline["monthly_throughput"] * throughput_factor,
            0,
        )
        additional_activity = max(float(activity_by_month[month_number - 1]), 0)

        opening_backlog = sum(bands.values())

        ageing_0_18 = min(bands["0_18"], bands["0_18"] * age_0_18_to_18_52_rate)
        bands["0_18"] -= ageing_0_18
        bands["18_52"] += ageing_0_18

        ageing_18_52 = min(
            bands["18_52"],
            bands["18_52"] * age_18_52_to_52_plus_rate,
        )
        bands["18_52"] -= ageing_18_52
        bands["52_plus"] += ageing_18_52

        bands["0_18"] += demand

        bands = remove_activity_from_bands(
            bands=bands,
            removals=baseline_throughput,
            strategy="Proportional",
        )

        bands = remove_activity_from_bands(
            bands=bands,
            removals=additional_activity,
            strategy=intervention_targeting,
        )

        closing_backlog = sum(bands.values())

        records.append(
            {
                "Scenario": scenario_name,
                "Month Number": month_number,
                "Month_Date": month_date,
                "Opening Backlog": opening_backlog,
                "Demand": demand,
                "Baseline Throughput": baseline_throughput,
                "Additional Activity": additional_activity,
                "Adjusted Throughput": baseline_throughput
                + additional_activity,
                "Net Change": closing_backlog - opening_backlog,
                "Closing Backlog": closing_backlog,
                "0-18 Weeks": bands["0_18"],
                "18-52 Weeks": bands["18_52"],
                "52+ Weeks": bands["52_plus"],
                "% Within 18 Weeks": (
                    bands["0_18"] / closing_backlog if closing_backlog else 0
                ),
                "% 52+ Weeks": (
                    bands["52_plus"] / closing_backlog if closing_backlog else 0
                ),
            }
        )

    return pd.DataFrame(records)


def age_weekly_wait_bands(
    buckets: list[float],
    weeks: float = WEEKS_PER_MONTH,
) -> list[float]:
    whole_weeks = math.floor(weeks)
    fractional_week = weeks - whole_weeks
    max_index = len(buckets) - 1
    aged = [0.0] * len(buckets)

    for index, value in enumerate(buckets):
        lower_target = min(index + whole_weeks, max_index)
        upper_target = min(lower_target + 1, max_index)

        aged[lower_target] += value * (1 - fractional_week)
        aged[upper_target] += value * fractional_week

    return aged


def add_monthly_demand_to_weekly_bands(
    buckets: list[float],
    monthly_demand: float,
    weeks: float = WEEKS_PER_MONTH,
) -> list[float]:
    updated = buckets.copy()
    whole_weeks = math.floor(weeks)
    fractional_week = weeks - whole_weeks
    weights = [1.0] * whole_weeks

    if fractional_week > 0:
        weights.append(fractional_week)

    weight_total = sum(weights)

    if monthly_demand <= 0 or weight_total <= 0:
        return updated

    for index, weight in enumerate(weights):
        if index >= len(updated):
            break

        updated[index] += monthly_demand * (weight / weight_total)

    return updated


def remove_from_weekly_wait_bands(
    buckets: list[float],
    removals: float,
    strategy: str,
) -> list[float]:
    updated = [max(float(value), 0) for value in buckets]
    removals = max(float(removals), 0)
    total = sum(updated)

    if total <= 0 or removals <= 0:
        return updated

    removals = min(removals, total)

    if strategy == "Proportional":
        return [
            max(value - removals * (value / total), 0)
            for value in updated
        ]

    if strategy in ["Longest waits first", "52+ first"]:
        removal_order = range(len(updated) - 1, -1, -1)
    elif strategy == "18-52 first":
        removal_order = (
            list(range(51, 17, -1))
            + list(range(len(updated) - 1, 51, -1))
            + list(range(17, -1, -1))
        )
    else:
        removal_order = range(len(updated) - 1, -1, -1)

    for index in removal_order:
        take = min(updated[index], removals)
        updated[index] -= take
        removals -= take

        if removals <= 0:
            break

    return [max(value, 0) for value in updated]


def summarise_weekly_wait_bands(buckets: list[float]) -> dict:
    zero_to_18 = sum(buckets[:18])
    eighteen_to_52 = sum(buckets[18:52])
    fifty_two_plus = sum(buckets[52:])
    total = zero_to_18 + eighteen_to_52 + fifty_two_plus

    return {
        "total": total,
        "0_18": zero_to_18,
        "18_52": eighteen_to_52,
        "52_plus": fifty_two_plus,
    }


def simulate_rtt_forecast_from_weekly_bands(
    history_df: pd.DataFrame,
    latest_wait_bands: list[float],
    horizon_months: int,
    baseline_months: int,
    additional_activity_by_month: list[float] | None = None,
    scenario_name: str = "Do Nothing",
    use_seasonality: bool = True,
    intervention_targeting: str = "Longest waits first",
    baseline_mode: str = "Conservative observed flow",
) -> pd.DataFrame:
    """
    Simulate RTT using the actual weekly wait-band shape from the latest month.

    This avoids the false cliff created by ageing a whole broad 18-52 week cohort
    at once. Patients only move into 52+ when their weekly bucket crosses 52 weeks.
    """

    if history_df.empty or not latest_wait_bands:
        return pd.DataFrame()

    history = history_df.sort_values("Month_Date").copy()
    latest = history.iloc[-1]
    baseline = calculate_baseline_inputs(
        history,
        baseline_months,
        baseline_mode=baseline_mode,
    )

    demand_index = calculate_seasonal_index(history, "additions_total")
    throughput_index = calculate_seasonal_index(history, "completed_total")

    activity_by_month = additional_activity_by_month or [0.0] * horizon_months
    activity_by_month = list(activity_by_month)[:horizon_months]

    if len(activity_by_month) < horizon_months:
        activity_by_month.extend([0.0] * (horizon_months - len(activity_by_month)))

    buckets = [max(float(value), 0) for value in latest_wait_bands]

    records = []

    for month_number in range(1, horizon_months + 1):
        month_date = latest["Month_Date"] + pd.DateOffset(months=month_number)
        month_key = month_date.month

        demand_factor = demand_index[month_key] if use_seasonality else 1.0
        throughput_factor = throughput_index[month_key] if use_seasonality else 1.0

        demand = max(baseline["monthly_demand"] * demand_factor, 0)
        baseline_throughput = max(
            baseline["monthly_throughput"] * throughput_factor,
            0,
        )
        additional_activity = max(float(activity_by_month[month_number - 1]), 0)

        opening_backlog = sum(buckets)
        buckets = age_weekly_wait_bands(buckets)
        buckets = add_monthly_demand_to_weekly_bands(buckets, demand)
        buckets = remove_from_weekly_wait_bands(
            buckets=buckets,
            removals=baseline_throughput,
            strategy="Proportional",
        )
        buckets = remove_from_weekly_wait_bands(
            buckets=buckets,
            removals=additional_activity,
            strategy=intervention_targeting,
        )

        band_summary = summarise_weekly_wait_bands(buckets)
        closing_backlog = band_summary["total"]

        records.append(
            {
                "Scenario": scenario_name,
                "Month Number": month_number,
                "Month_Date": month_date,
                "Opening Backlog": opening_backlog,
                "Demand": demand,
                "Baseline Throughput": baseline_throughput,
                "Additional Activity": additional_activity,
                "Adjusted Throughput": baseline_throughput
                + additional_activity,
                "Net Change": closing_backlog - opening_backlog,
                "Closing Backlog": closing_backlog,
                "0-18 Weeks": band_summary["0_18"],
                "18-52 Weeks": band_summary["18_52"],
                "52+ Weeks": band_summary["52_plus"],
                "% Within 18 Weeks": (
                    band_summary["0_18"] / closing_backlog
                    if closing_backlog
                    else 0
                ),
                "% 52+ Weeks": (
                    band_summary["52_plus"] / closing_backlog
                    if closing_backlog
                    else 0
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_theatre_additional_cases(
    sessions_per_week: float,
    active_weeks: float,
    session_minutes: float,
    current_utilisation: float,
    target_utilisation: float,
    avg_case_duration_minutes: float,
    effort: float,
    horizon_months: int,
) -> dict:
    utilisation_gap = max(target_utilisation - current_utilisation, 0) * effort
    avg_case_duration_minutes = max(avg_case_duration_minutes, 1)

    total_cases = (
        sessions_per_week
        * active_weeks
        * session_minutes
        * utilisation_gap
        / avg_case_duration_minutes
    )

    monthly_cases = total_cases / max(horizon_months, 1)

    return {
        "utilisation_gap": utilisation_gap,
        "monthly_cases": monthly_cases,
        "total_cases": total_cases,
    }


def calculate_outpatient_additional_appointments(
    clinic_sessions_per_week: float,
    patients_per_session: float,
    template_current_fill: float,
    template_target_fill: float,
    template_rtt_relevant_share: float,
    eligible_new_per_week: float,
    eligible_follow_up_per_week: float,
    current_dna_rate: float,
    target_dna_rate: float,
    pifu_conversion_rate: float,
    fn_ratio_improvement_rate: float,
    active_weeks: float,
    effort: float,
    horizon_months: int,
) -> dict:
    weekly_template_slots = clinic_sessions_per_week * patients_per_session
    eligible_appointments = eligible_new_per_week + eligible_follow_up_per_week

    template_fill = (
        weekly_template_slots
        * template_rtt_relevant_share
        * max(template_target_fill - template_current_fill, 0)
        * active_weeks
        * effort
    )

    dna_reduction = (
        eligible_appointments
        * max(current_dna_rate - target_dna_rate, 0)
        * active_weeks
        * effort
    )

    pifu = (
        eligible_follow_up_per_week
        * max(pifu_conversion_rate, 0)
        * active_weeks
        * effort
    )

    fn_ratio = (
        eligible_new_per_week
        * max(fn_ratio_improvement_rate, 0)
        * active_weeks
        * effort
    )

    total = template_fill + dna_reduction + pifu + fn_ratio
    monthly_total = total / max(horizon_months, 1)

    return {
        "template_fill": template_fill / max(horizon_months, 1),
        "dna_reduction": dna_reduction / max(horizon_months, 1),
        "pifu": pifu / max(horizon_months, 1),
        "fn_ratio": fn_ratio / max(horizon_months, 1),
        "monthly_appointments": monthly_total,
        "total_appointments": total,
    }


def build_monthly_activity_profile(
    steady_monthly_activity: float,
    horizon_months: int,
    ramp_months: int = 0,
) -> list[float]:
    profile = []

    for month_number in range(1, horizon_months + 1):
        if ramp_months <= 0:
            ramp_factor = 1.0
        else:
            ramp_factor = min(month_number / ramp_months, 1.0)

        profile.append(steady_monthly_activity * ramp_factor)

    return profile


def create_actual_history_series(history_df: pd.DataFrame) -> pd.DataFrame:
    actual = history_df.copy()

    actual["Scenario"] = "Actual"
    actual["Closing Backlog"] = actual["Total"]
    actual["0-18 Weeks"] = actual["waiting_0_18_total"]
    actual["18-52 Weeks"] = actual["waiting_18_52_total"]
    actual["52+ Weeks"] = actual["waiting_52_plus_total"]
    actual["% Within 18 Weeks"] = actual["pct_0_18"]
    actual["% 52+ Weeks"] = actual["pct_52_plus"]

    return actual[
        [
            "Month_Date",
            "Scenario",
            "Closing Backlog",
            "0-18 Weeks",
            "18-52 Weeks",
            "52+ Weeks",
            "% Within 18 Weeks",
            "% 52+ Weeks",
            "Monthly_Backlog_Change",
            "Net_Flow",
            "additions_total",
            "completed_total",
        ]
    ].copy()


def add_latest_actual_bridge(
    actual_df: pd.DataFrame,
    projection_df: pd.DataFrame,
) -> pd.DataFrame:
    if actual_df.empty or projection_df.empty:
        return projection_df

    latest_actual = actual_df.sort_values("Month_Date").iloc[-1]
    bridge_rows = []

    for scenario in projection_df["Scenario"].dropna().unique():
        row = latest_actual.to_dict()
        row["Scenario"] = scenario
        row["Month Number"] = 0
        row["Opening Backlog"] = latest_actual["Closing Backlog"]
        row["Demand"] = math.nan
        row["Baseline Throughput"] = math.nan
        row["Additional Activity"] = 0.0
        row["Adjusted Throughput"] = math.nan
        row["Net Change"] = 0.0
        bridge_rows.append(row)

    return pd.concat(
        [pd.DataFrame(bridge_rows), projection_df],
        ignore_index=True,
        sort=False,
    )
