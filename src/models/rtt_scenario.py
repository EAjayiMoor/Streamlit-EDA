import pandas as pd


def remove_from_bands(
    bands: dict,
    removals: float,
    targeting_strategy: str,
) -> dict:
    """
    Remove completed pathways from wait-band cohorts.

    Why:
    - Different interventions may target different parts of the backlog.
    - Removing from 52+ first has a different impact from proportional removal.
    - This keeps the modelling logic transparent and explainable.
    """

    bands = bands.copy()
    removals = max(float(removals), 0)

    total_backlog = sum(bands.values())

    if total_backlog <= 0 or removals <= 0:
        return bands

    removals = min(removals, total_backlog)

    if targeting_strategy == "Proportional":
        for band in bands:
            bands[band] -= removals * (bands[band] / total_backlog)

    elif targeting_strategy in ["Longest waits first", "52+ first"]:
        for band in ["52_plus", "18_52", "0_18"]:
            take = min(bands[band], removals)
            bands[band] -= take
            removals -= take

            if removals <= 0:
                break

    elif targeting_strategy == "18–52 first":
        for band in ["18_52", "52_plus", "0_18"]:
            take = min(bands[band], removals)
            bands[band] -= take
            removals -= take

            if removals <= 0:
                break

    elif targeting_strategy == "Early waits first":
        for band in ["0_18", "18_52", "52_plus"]:
            take = min(bands[band], removals)
            bands[band] -= take
            removals -= take

            if removals <= 0:
                break

    return {key: max(value, 0) for key, value in bands.items()}


def simulate_rtt_scenario(
    start_0_18: float,
    start_18_52: float,
    start_52_plus: float,
    monthly_demand: float,
    monthly_throughput: float,
    horizon_months: int,
    demand_change_pct: float = 0,
    throughput_change_pct: float = 0,
    additional_activity_per_year: float = 0,
    targeting_strategy: str = "Proportional",
    scenario_name: str = "Scenario",
) -> pd.DataFrame:
    """
    Simulate RTT backlog movement by broad wait-band cohort.

    Model logic:
    - Opening backlog
    - Plus new RTT demand entering 0–18 weeks
    - Plus ageing into later wait bands
    - Minus completed pathways / intervention removals
    - Equals closing backlog

    This is a deterministic scenario model, not a machine learning forecast.
    """

    records = []

    bands = {
        "0_18": float(start_0_18),
        "18_52": float(start_18_52),
        "52_plus": float(start_52_plus),
    }

    adjusted_demand = monthly_demand * (1 + demand_change_pct / 100)
    baseline_adjusted_throughput = monthly_throughput * (1 + throughput_change_pct / 100)
    monthly_additional_activity = additional_activity_per_year / 12
    adjusted_throughput = baseline_adjusted_throughput + monthly_additional_activity

    # Approximate monthly ageing.
    # 0–18 weeks covers 18 weeks; one month is approximately 4 weeks.
    # 18–52 weeks covers 34 weeks; one month is approximately 4 weeks.
    age_0_18_to_18_52_rate = 4 / 18
    age_18_52_to_52_plus_rate = 4 / 34

    for month_number in range(1, horizon_months + 1):
        opening_total = sum(bands.values())

        # Age unresolved patients forward.
        ageing_0_18_to_18_52 = bands["0_18"] * age_0_18_to_18_52_rate
        ageing_18_52_to_52_plus = bands["18_52"] * age_18_52_to_52_plus_rate

        bands["0_18"] -= ageing_0_18_to_18_52
        bands["18_52"] += ageing_0_18_to_18_52

        bands["18_52"] -= ageing_18_52_to_52_plus
        bands["52_plus"] += ageing_18_52_to_52_plus

        # New RTT demand enters the early wait cohort.
        bands["0_18"] += adjusted_demand

        # Throughput removes patients from the backlog according to the selected strategy.
        bands = remove_from_bands(
            bands=bands,
            removals=adjusted_throughput,
            targeting_strategy=targeting_strategy,
        )

        closing_total = sum(bands.values())

        records.append(
            {
                "Month Number": month_number,
                "Scenario": scenario_name,
                "Demand": adjusted_demand,
                "Baseline Throughput": baseline_adjusted_throughput,
                "Additional Activity": monthly_additional_activity,
                "Adjusted Throughput": adjusted_throughput,
                "Net Flow": adjusted_demand - adjusted_throughput,
                "Opening Backlog": opening_total,
                "0–18 Weeks": bands["0_18"],
                "18–52 Weeks": bands["18_52"],
                "52+ Weeks": bands["52_plus"],
                "Closing Backlog": closing_total,
                "% Within 18 Weeks": bands["0_18"] / closing_total if closing_total else 0,
                "% 52+ Weeks": bands["52_plus"] / closing_total if closing_total else 0,
            }
        )

    result = pd.DataFrame(records)

    numeric_cols = [
        "Demand",
        "Baseline Throughput",
        "Additional Activity",
        "Adjusted Throughput",
        "Net Flow",
        "Opening Backlog",
        "0–18 Weeks",
        "18–52 Weeks",
        "52+ Weeks",
        "Closing Backlog",
    ]

    for col in numeric_cols:
        result[col] = result[col].round(0)

    return result