from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.financial_loader import (
    format_currency,
    load_staff_cost_data,
    load_trial_balance,
    surgical_theatre_mask,
)
from src.data.outpatient_loader import load_outpatient_data
from src.data.ptl_loader import load_ptl_data, summarise_ptl_by_month
from src.data.theatre_loader import (
    load_theatre_activity_data,
    summarise_theatre_capacity,
    summarise_theatre_session_type_split,
    summarise_vanguard_capacity_impact,
)
from src.transforms.outpatient_transform import (
    summarise_outpatient_attendances_by_month,
    summarise_outpatient_attendances_by_month_visit_type,
)
from src.transforms.rtt_transform import add_wait_band_metrics, filter_pah_incomplete
from src.utils.data_cleaning import remove_aggregate_rows


st.set_page_config(page_title="Ad Hoc Analysis", layout="wide")

st.title("Ad Hoc Analysis")
st.caption(
    "Fast-turnaround analytical views. Current view: theatre and outpatient what-if analysis."
)

FULL_BASELINE_MONTHS = 60
THEATRE_BASELINE_START = pd.Timestamp("2025-04-01")
THEATRE_BASELINE_END = pd.Timestamp("2026-03-31")
OUTPATIENT_BASELINE_START = pd.Timestamp("2025-04-01")
OUTPATIENT_BASELINE_END = pd.Timestamp("2026-03-31")
DEFAULT_DELIVERY_WEEKS = 43.0
OUTPATIENT_SLOT_VALUE_DEFAULT = 200.0
THEATRE_CASE_VALUE_DEFAULT = 250.0
SESSION_STANDARD_MINUTES = 240
ESTATE_CAPACITY_WEEKS = 52.0
TOTAL_ESTATE_THEATRES = 13
DEFAULT_ESTATE_THEATRES = 10
JOB_PLAN_PATH = Path("data/raw/job planning splits")
RTT_DATA_PATH = Path("data/raw/rtt")
SCENARIO_TARGETS = {
    "What if 50% delivery": 0.785,
    "What if 75% delivery": 0.818,
    "What if 100% delivery": 0.85,
}
DISPLAY_SCENARIO_LABELS = {
    "What if 50% delivery": "What if utilisation improves to 78.5%",
    "What if 75% delivery": "What if utilisation improves to 81.75%",
    "What if 100% delivery": "What if utilisation improves to 85%",
}


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_decimal(value: float, places: int = 1) -> str:
    return f"{value:,.{places}f}"


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def sum_abs(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).abs().sum())


def format_vanguard_backlog_capacity_impact(
    completed_cases: float,
    sessions: float,
    current_backlog: float,
    primary_specialty: str,
    primary_specialty_cases: float,
) -> str:
    if completed_cases <= 0:
        return "Not quantified: no 25/26 Vanguard elective completed cases found in the theatre extract."

    backlog_text = "latest PTL unavailable"
    if current_backlog > 0:
        backlog_text = (
            f"{format_percent(completed_cases / current_backlog)} of latest PTL "
            f"({format_number(current_backlog)})"
        )

    specialty_text = ""
    if primary_specialty and primary_specialty != "Not available":
        specialty_text = (
            f" Main specialty: {primary_specialty} "
            f"({format_number(primary_specialty_cases)} cases)."
        )

    return (
        f"Vanguard delivered {format_number(completed_cases)} elective completed "
        f"cases in 25/26 across {format_number(sessions)} sessions. If Vanguard "
        "capacity is reduced, this capacity needs to be replaced or absorbed "
        f"elsewhere to avoid backlog deterioration; equivalent to {backlog_text}."
        f"{specialty_text}"
    )


def get_theatre_cost_per_minute() -> tuple[float, float]:
    tb_df = load_trial_balance()
    cost_mask = tb_df["Expenditure Type"].fillna("").astype(str).str.lower() != "income"
    theatre_mask = tb_df["Search_Text"].str.contains(
        "theatre|anaesth",
        regex=True,
        na=False,
    )
    theatre_cost = tb_df[cost_mask & theatre_mask]["FY_2526_Total"].abs().sum()
    return float(theatre_cost), 0.0


def load_latest_job_plan_capacity() -> dict:
    files = sorted(JOB_PLAN_PATH.glob("CompareActivitiesReport April *.csv"))

    if not files:
        return {
            "source": "Not available",
            "theatre_weekly": 0.0,
            "outpatient_weekly": 0.0,
            "rows": 0,
        }

    latest_file = files[-1]
    df = pd.read_csv(latest_file, low_memory=False)
    df.columns = df.columns.str.strip()

    for col in ["Operating sessions", "Out-patient activities"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    classification = (
        df.get("Classification", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    substantive_df = df[classification != "locum"].copy()

    return {
        "source": latest_file.name,
        "theatre_weekly": float(substantive_df["Operating sessions"].sum()),
        "outpatient_weekly": float(substantive_df["Out-patient activities"].sum()),
        "rows": len(substantive_df),
    }


@st.cache_data(show_spinner=False)
def load_latest_rtt_wait_band_position(path: str = "data/raw/rtt") -> dict:
    path = Path(path)
    files = sorted(path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No RTT files found in: {path}")

    latest_file = max(
        files,
        key=lambda file: pd.to_datetime(file.name[:8], errors="coerce"),
    )
    df = pd.read_csv(latest_file, low_memory=False)
    df.columns = df.columns.str.strip()

    parts = latest_file.stem.split("-")
    month_label = f"{parts[2]} {parts[3]}" if len(parts) >= 4 else latest_file.stem
    df["Month"] = month_label

    df = remove_aggregate_rows(df)
    pah_incomplete = filter_pah_incomplete(df)
    metric_df = add_wait_band_metrics(pah_incomplete)

    total = float(pd.to_numeric(metric_df["Total All"], errors="coerce").fillna(0).sum())
    waiting_0_18 = float(metric_df["0_18_total"].sum())
    waiting_52_plus = float(metric_df["52_plus_total"].sum())
    waiting_18_51 = max(total - waiting_0_18 - waiting_52_plus, 0)

    return {
        "month": month_label,
        "source": latest_file.name,
        "total": total,
        "waiting_0_18": waiting_0_18,
        "waiting_18_51": waiting_18_51,
        "waiting_52_plus": waiting_52_plus,
        "pct_0_18": waiting_0_18 / total if total > 0 else 0,
        "pct_52_plus": waiting_52_plus / total if total > 0 else 0,
    }


def allocate_rtt_backlog_reduction(opening: dict, converted_capacity: float) -> dict:
    remaining_capacity = max(converted_capacity, 0)

    reduced_52_plus = min(remaining_capacity, opening["52_plus"])
    remaining_capacity -= reduced_52_plus

    reduced_18_51 = min(remaining_capacity, opening["18_51"])
    remaining_capacity -= reduced_18_51

    reduced_0_18 = min(remaining_capacity, opening["0_18"])
    remaining_capacity -= reduced_0_18

    applied_reduction = reduced_52_plus + reduced_18_51 + reduced_0_18
    closing_52_plus = opening["52_plus"] - reduced_52_plus
    closing_18_51 = opening["18_51"] - reduced_18_51
    closing_0_18 = opening["0_18"] - reduced_0_18
    closing_total = closing_52_plus + closing_18_51 + closing_0_18

    return {
        "reduced_52_plus": reduced_52_plus,
        "reduced_18_51": reduced_18_51,
        "reduced_0_18": reduced_0_18,
        "applied_reduction": applied_reduction,
        "unused_capacity": max(remaining_capacity, 0),
        "closing": {
            "52_plus": closing_52_plus,
            "18_51": closing_18_51,
            "0_18": closing_0_18,
            "total": closing_total,
        },
    }


def build_outpatient_baseline(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "planned_appointments": 0.0,
            "attended_appointments": 0.0,
            "dna_appointments": 0.0,
            "planned_sessions": 0.0,
            "actual_sessions": 0.0,
            "observed_weeks": 1.0,
            "fill_rate": 0.0,
            "dna_rate": 0.0,
            "slots_per_session": 0.0,
            "follow_up": 0.0,
            "first_attendance": 0.0,
        }

    work = df.copy()
    status_clean = (
        work["Status"].fillna("").astype(str).str.strip().str.lower()
        if "Status" in work.columns
        else pd.Series("", index=work.index)
    )
    attended_mask = (
        status_clean.str.contains("checked in")
        | status_clean.str.contains("check in")
        | status_clean.str.contains("checked out")
        | status_clean.str.contains("check out")
    )
    dna_mask = status_clean.str.contains("no show")

    planned_appointments = float(work["Contact_ID"].nunique())
    attended_appointments = float(work.loc[attended_mask, "Contact_ID"].nunique())
    dna_appointments = float(work.loc[dna_mask, "Contact_ID"].nunique())

    clinic_col = (
        "ContactClinicPerfUnit"
        if "ContactClinicPerfUnit" in work.columns
        else "TreatmentFunctionDesc"
    )
    work["Session_Date"] = work["Contact_Start"].dt.date
    work["Session_Part"] = work["Contact_Start"].dt.hour.apply(
        lambda hour: "AM" if hour < 13 else "PM"
    )

    session_cols = ["Session_Date", "Session_Part", clinic_col]
    planned_session_df = work.dropna(subset=["Contact_Start"]).drop_duplicates(
        subset=session_cols
    )
    actual_session_df = work.loc[attended_mask].drop_duplicates(subset=session_cols)

    observed_days = (work["Contact_Start"].max() - work["Contact_Start"].min()).days + 1
    observed_weeks = max(observed_days / 7, 1)

    visit_group = (
        work.get("ContactVisitType_Group", pd.Series("Unknown", index=work.index))
        .fillna("Unknown")
        .astype(str)
    )
    follow_up = float(
        work.loc[attended_mask & (visit_group == "Follow Up"), "Contact_ID"].nunique()
    )
    first_attendance = float(
        work.loc[
            attended_mask & (visit_group == "First attendance"),
            "Contact_ID",
        ].nunique()
    )

    planned_sessions = float(len(planned_session_df))
    actual_sessions = float(len(actual_session_df))

    return {
        "planned_appointments": planned_appointments,
        "attended_appointments": attended_appointments,
        "dna_appointments": dna_appointments,
        "planned_sessions": planned_sessions,
        "actual_sessions": actual_sessions,
        "observed_weeks": observed_weeks,
        "fill_rate": (
            attended_appointments / planned_appointments
            if planned_appointments > 0
            else 0.0
        ),
        "dna_rate": (
            dna_appointments / planned_appointments
            if planned_appointments > 0
            else 0.0
        ),
        "slots_per_session": (
            planned_appointments / planned_sessions
            if planned_sessions > 0
            else 0.0
        ),
        "follow_up": follow_up,
        "first_attendance": first_attendance,
    }


def validation_label(condition: bool) -> str:
    return "Pass" if condition else "Check required"


def build_specialty_utilisation(
    df: pd.DataFrame,
    recent_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [
        "Theatre session ID",
        "Booked Operation Date",
        "Scheduled start time(Session)",
        "Scheduled finish time(Session)",
        "Number of cases completed",
        "Specialty (standardised)",
    ]

    if df.empty or any(col not in df.columns for col in required_cols):
        return pd.DataFrame(), pd.DataFrame()

    work = df.dropna(
        subset=[
            "Theatre session ID",
            "Booked Operation Date",
            "Scheduled start time(Session)",
            "Scheduled finish time(Session)",
        ]
    ).copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    work["Specialty (standardised)"] = (
        work["Specialty (standardised)"].fillna("Unknown").astype(str).str.strip()
    )
    work.loc[work["Specialty (standardised)"] == "", "Specialty (standardised)"] = (
        "Unknown"
    )
    work["Scheduled_Minutes"] = (
        work["Scheduled finish time(Session)"]
        - work["Scheduled start time(Session)"]
    ).dt.total_seconds() / 60
    work.loc[work["Scheduled_Minutes"] < 0, "Scheduled_Minutes"] += 24 * 60

    touch_col = (
        "Model_Hospital_Touch_Minutes"
        if "Model_Hospital_Touch_Minutes" in work.columns
        else "Case Touch time (minutes)"
    )
    work[touch_col] = pd.to_numeric(work[touch_col], errors="coerce").fillna(0)
    valid_touch_mask = work[touch_col].between(0, 720)
    work["Valid_Touch_Minutes"] = work[touch_col].where(
        valid_touch_mask,
        0,
    )

    for col in ["Actual start time(Session)", "Actual finish time(Session)"]:
        if col not in work.columns:
            work[col] = pd.NaT

    latest_date = work["Booked Operation Date"].max()
    start_date = latest_date - pd.DateOffset(months=recent_months)
    work = work[work["Booked Operation Date"] >= start_date].copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "Elective/Emergency" in work.columns:
        row_type = (
            work["Elective/Emergency"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        work["_Row_Session_Type"] = row_type.map(
            lambda value: "Elective"
            if "elective" in value
            else (
                "Emergency"
                if "emergency" in value or "trauma" in value
                else "Unknown"
            )
        )
        work["_Session_Type"] = work.groupby(
            ["Booked Operation Date", "Theatre session ID"]
        )["_Row_Session_Type"].transform(
            lambda values: "Elective"
            if set(values.dropna()) == {"Elective"}
            else (
                "Emergency"
                if set(values.dropna()) == {"Emergency"}
                else "Mixed/unknown"
            )
        )
    else:
        work["_Session_Type"] = "Unknown"

    work["_Session_Has_Obstetrics"] = (
        work["Specialty (standardised)"]
        .fillna("")
        .astype(str)
        .str.contains("obstetric|maternity", case=False, regex=True)
        .groupby([work["Booked Operation Date"], work["Theatre session ID"]])
        .transform("max")
    )

    session_summary = (
        work.groupby(["Booked Operation Date", "Theatre session ID"], as_index=False)
        .agg(
            Session_Type=("_Session_Type", "first"),
            Session_Has_Obstetrics=("_Session_Has_Obstetrics", "max"),
            Scheduled_Minutes=("Scheduled_Minutes", "first"),
            Touch_Minutes=("Valid_Touch_Minutes", "sum"),
            Completed_Cases=("Number of cases completed", "sum"),
            Actual_Start=("Actual start time(Session)", "min"),
            Actual_Finish=("Actual finish time(Session)", "max"),
        )
    )
    session_summary["Actual_Session_Flag"] = (
        (session_summary["Touch_Minutes"] > 0)
        | (session_summary["Completed_Cases"] > 0)
        | session_summary["Actual_Start"].notna()
        | session_summary["Actual_Finish"].notna()
    )
    session_summary["Valid_Scheduled_Session"] = session_summary[
        "Scheduled_Minutes"
    ].between(30, 720)

    eligible_sessions = session_summary[
        (session_summary["Session_Type"] == "Elective")
        & (~session_summary["Session_Has_Obstetrics"])
        & (session_summary["Actual_Session_Flag"])
        & (session_summary["Valid_Scheduled_Session"])
    ][["Booked Operation Date", "Theatre session ID"]]

    if eligible_sessions.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = work.merge(
        eligible_sessions,
        on=["Booked Operation Date", "Theatre session ID"],
        how="inner",
    )
    work["Month"] = work["Booked Operation Date"].dt.to_period("M").dt.to_timestamp()

    specialty_session = (
        work.groupby(
            [
                "Booked Operation Date",
                "Theatre session ID",
                "Specialty (standardised)",
            ],
            as_index=False,
        )
        .agg(
            Month=("Month", "min"),
            Scheduled_Minutes=("Scheduled_Minutes", "first"),
            Touch_Minutes=("Valid_Touch_Minutes", "sum"),
            Completed_Cases=("Number of cases completed", "sum"),
        )
    )

    session_totals = (
        specialty_session.groupby(
            ["Booked Operation Date", "Theatre session ID"],
            as_index=False,
        )
        .agg(
            Session_Touch_Minutes=("Touch_Minutes", "sum"),
            Session_Cases=("Completed_Cases", "sum"),
        )
    )

    specialty_session = specialty_session.merge(
        session_totals,
        on=["Booked Operation Date", "Theatre session ID"],
        how="left",
    )

    specialty_session["Allocation_Basis"] = specialty_session["Touch_Minutes"]
    no_touch = specialty_session["Session_Touch_Minutes"] <= 0
    specialty_session.loc[no_touch, "Allocation_Basis"] = specialty_session.loc[
        no_touch,
        "Completed_Cases",
    ]
    no_touch_or_cases = no_touch & (specialty_session["Session_Cases"] <= 0)
    specialty_session.loc[no_touch_or_cases, "Allocation_Basis"] = 1

    basis_total = specialty_session.groupby(
        ["Booked Operation Date", "Theatre session ID"]
    )["Allocation_Basis"].transform("sum")
    specialty_session["Allocated_Scheduled_Minutes"] = (
        specialty_session["Scheduled_Minutes"]
        * specialty_session["Allocation_Basis"]
        / basis_total
    )

    group_cols = ["Month", "Specialty (standardised)"]
    monthly = (
        specialty_session.groupby(group_cols, as_index=False)
        .agg(
            Specialty_Sessions=("Theatre session ID", "count"),
            Scheduled_Minutes=("Allocated_Scheduled_Minutes", "sum"),
            Touch_Minutes=("Touch_Minutes", "sum"),
            Completed_Cases=("Completed_Cases", "sum"),
        )
    )
    monthly["Utilisation"] = (
        monthly["Touch_Minutes"] / monthly["Scheduled_Minutes"]
    ).fillna(0)

    overall = (
        specialty_session.groupby("Specialty (standardised)", as_index=False)
        .agg(
            Specialty_Sessions=("Theatre session ID", "count"),
            Scheduled_Minutes=("Allocated_Scheduled_Minutes", "sum"),
            Touch_Minutes=("Touch_Minutes", "sum"),
            Completed_Cases=("Completed_Cases", "sum"),
        )
    )
    overall["Utilisation"] = (
        overall["Touch_Minutes"] / overall["Scheduled_Minutes"]
    ).fillna(0)

    return monthly, overall


def build_outpatient_specialty_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Contact_ID" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    specialty_col = (
        "Standardised_Specialty"
        if "Standardised_Specialty" in work.columns
        else "TreatmentFunctionDesc"
    )
    if specialty_col not in work.columns:
        return pd.DataFrame()

    work[specialty_col] = (
        work[specialty_col].fillna("Unknown").astype(str).str.strip()
    )
    work.loc[work[specialty_col] == "", specialty_col] = "Unknown"

    status_clean = (
        work["Status"].fillna("").astype(str).str.strip().str.lower()
        if "Status" in work.columns
        else pd.Series("", index=work.index)
    )
    work["_Attended"] = (
        status_clean.str.contains("checked in")
        | status_clean.str.contains("check in")
        | status_clean.str.contains("checked out")
        | status_clean.str.contains("check out")
    )
    work["_DNA"] = status_clean.str.contains("no show")

    visit_group = (
        work.get("ContactVisitType_Group", pd.Series("Unknown", index=work.index))
        .fillna("Unknown")
        .astype(str)
    )
    work["_Follow_Up"] = work["_Attended"] & (visit_group == "Follow Up")
    work["_First_Attendance"] = work["_Attended"] & (
        visit_group == "First attendance"
    )

    clinic_col = (
        "ContactClinicPerfUnit"
        if "ContactClinicPerfUnit" in work.columns
        else "TreatmentFunctionDesc"
    )
    work["Session_Date"] = work["Contact_Start"].dt.date
    work["Session_Part"] = work["Contact_Start"].dt.hour.apply(
        lambda hour: "AM" if hour < 13 else "PM"
    )
    session_cols = [specialty_col, "Session_Date", "Session_Part", clinic_col]
    planned_sessions = (
        work.dropna(subset=["Contact_Start"])
        .drop_duplicates(subset=session_cols)
        .groupby(specialty_col)
        .size()
    )
    actual_sessions = (
        work.loc[work["_Attended"]]
        .drop_duplicates(subset=session_cols)
        .groupby(specialty_col)
        .size()
    )

    rows = []
    for specialty, group in work.groupby(specialty_col):
        planned = float(group["Contact_ID"].nunique())
        attended = float(group.loc[group["_Attended"], "Contact_ID"].nunique())
        dna = float(group.loc[group["_DNA"], "Contact_ID"].nunique())
        follow_up = float(group.loc[group["_Follow_Up"], "Contact_ID"].nunique())
        first_attendance = float(
            group.loc[group["_First_Attendance"], "Contact_ID"].nunique()
        )
        rows.append(
            {
                "Specialty": specialty,
                "Planned appointment records": planned,
                "Actual attended appointments": attended,
                "DNA / no-show appointments": dna,
                "DNA rate": dna / planned if planned > 0 else 0.0,
                "Attendance / fill proxy": attended / planned if planned > 0 else 0.0,
                "First attendances": first_attendance,
                "Follow-up attendances": follow_up,
                "Planned clinic-session proxies": float(
                    planned_sessions.get(specialty, 0)
                ),
                "Actual clinic-session proxies": float(actual_sessions.get(specialty, 0)),
            }
        )

    breakdown = pd.DataFrame(rows)
    if breakdown.empty:
        return breakdown

    breakdown["Attended appointments / actual session"] = breakdown.apply(
        lambda row: (
            row["Actual attended appointments"] / row["Actual clinic-session proxies"]
            if row["Actual clinic-session proxies"] > 0
            else 0.0
        ),
        axis=1,
    )
    return breakdown.sort_values(
        "Actual attended appointments",
        ascending=False,
    ).reset_index(drop=True)


def calculate_outpatient_scenario_outputs(
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
    horizon_months: int,
    baseline_monthly_attendances: float,
    current_ptl_value: float,
    rtt_conversion: float,
    value_per_appointment: float,
) -> dict:
    weekly_template_slots = clinic_sessions_per_week * patients_per_session
    eligible_appointments = eligible_new_per_week + eligible_follow_up_per_week
    horizon = max(horizon_months, 1)

    outputs = {}

    for scenario, target_share in {
        "What if 50% delivery": 0.50,
        "What if 75% delivery": 0.75,
        "What if 100% delivery": 1.00,
    }.items():
        template_fill_target = template_current_fill + (
            max(template_target_fill - template_current_fill, 0) * target_share
        )
        dna_rate_target = current_dna_rate - (
            max(current_dna_rate - target_dna_rate, 0) * target_share
        )
        pifu_target = max(pifu_conversion_rate, 0) * target_share
        fn_ratio_target = max(fn_ratio_improvement_rate, 0) * target_share

        template_fill_total = (
            weekly_template_slots
            * template_rtt_relevant_share
            * max(template_target_fill - template_current_fill, 0)
            * active_weeks
            * target_share
        )
        dna_reduction_total = (
            eligible_appointments
            * max(current_dna_rate - target_dna_rate, 0)
            * active_weeks
            * target_share
        )
        pifu_total = (
            eligible_follow_up_per_week
            * max(pifu_conversion_rate, 0)
            * active_weeks
            * target_share
        )
        fn_ratio_total = (
            eligible_new_per_week
            * max(fn_ratio_improvement_rate, 0)
            * active_weeks
            * target_share
        )
        total = (
            template_fill_total
            + dna_reduction_total
            + pifu_total
            + fn_ratio_total
        )
        monthly_total = total / horizon
        converted_ptl_impact = total * rtt_conversion
        remaining_ptl = max(current_ptl_value - converted_ptl_impact, 0)
        financial_opportunity = total * value_per_appointment

        outputs[scenario] = {
            "target_share": target_share,
            "template_fill_target": template_fill_target,
            "dna_rate_target": dna_rate_target,
            "pifu_target": pifu_target,
            "fn_ratio_target": fn_ratio_target,
            "template_fill_monthly": template_fill_total / horizon,
            "dna_reduction_monthly": dna_reduction_total / horizon,
            "pifu_monthly": pifu_total / horizon,
            "fn_ratio_monthly": fn_ratio_total / horizon,
            "additional_monthly": monthly_total,
            "template_fill_total": template_fill_total,
            "dna_reduction_total": dna_reduction_total,
            "pifu_total": pifu_total,
            "fn_ratio_total": fn_ratio_total,
            "additional_total": total,
            "activity_after_monthly": baseline_monthly_attendances + monthly_total,
            "activity_after_total": (baseline_monthly_attendances * horizon) + total,
            "converted_ptl_impact": converted_ptl_impact,
            "remaining_ptl": remaining_ptl,
            "ptl_reduction_pct": (
                converted_ptl_impact / current_ptl_value
                if current_ptl_value > 0
                else 0
            ),
            "remaining_ptl_pct": (
                remaining_ptl / current_ptl_value
                if current_ptl_value > 0
                else 0
            ),
            "financial_opportunity": financial_opportunity,
        }

    return outputs


try:
    theatre_df = load_theatre_activity_data()
    ptl_df = summarise_ptl_by_month(load_ptl_data())
except Exception as e:
    st.error(f"Error loading ad hoc analysis data: {e}")
    st.stop()

capacity = summarise_theatre_capacity(
    theatre_df,
    start_date=THEATRE_BASELINE_START,
    end_date=THEATRE_BASELINE_END,
    session_type_scope="elective",
    exclude_obstetrics=True,
    actual_sessions_only_for_utilisation=True,
    touch_time_column="Model_Hospital_Touch_Minutes",
)
theatre_activity_split_df = summarise_theatre_session_type_split(
    theatre_df,
    start_date=THEATRE_BASELINE_START,
    end_date=THEATRE_BASELINE_END,
    touch_time_column="Model_Hospital_Touch_Minutes",
)

if not theatre_activity_split_df.empty:
    elective_activity_context = theatre_activity_split_df[
        theatre_activity_split_df["Session type"] == "Elective"
    ]
else:
    elective_activity_context = pd.DataFrame()

full_year_elective_240_session_equivalents = (
    float(elective_activity_context["Actual 240-min session equivalents"].iloc[0])
    if not elective_activity_context.empty
    and "Actual 240-min session equivalents" in elective_activity_context.columns
    else (
        float(capacity["Actual_240_Session_Equivalents"])
        if not capacity.empty
        else 0.0
    )
)
full_year_elective_scheduled_minutes = (
    float(elective_activity_context["Scheduled minutes used"].iloc[0])
    if not elective_activity_context.empty
    and "Scheduled minutes used" in elective_activity_context.columns
    else (
        float(capacity["Actual_Session_Scheduled_Minutes"])
        if not capacity.empty
        else 0.0
    )
)
full_year_elective_touch_minutes = (
    float(elective_activity_context["Touch minutes used"].iloc[0])
    if not elective_activity_context.empty
    and "Touch minutes used" in elective_activity_context.columns
    else (
        float(capacity["Touch_Minutes"])
        if not capacity.empty
        else 0.0
    )
)
full_year_elective_utilisation = (
    full_year_elective_touch_minutes / full_year_elective_scheduled_minutes
    if full_year_elective_scheduled_minutes > 0
    else 0.0
)
full_year_elective_actual_sessions_used = (
    float(
        elective_activity_context["Valid actual sessions used for utilisation"].iloc[0]
    )
    if not elective_activity_context.empty
    and "Valid actual sessions used for utilisation"
    in elective_activity_context.columns
    else (
        float(capacity["Actual_Sessions"])
        if not capacity.empty
        else 0.0
    )
)
full_year_elective_completed_cases = (
    float(elective_activity_context["Completed cases"].iloc[0])
    if not elective_activity_context.empty
    and "Completed cases" in elective_activity_context.columns
    else (
        float(capacity["Completed_Cases"])
        if not capacity.empty
        else 0.0
    )
)

if capacity.empty:
    st.warning("No theatre utilisation summary is available.")
    st.stop()

latest_ptl = ptl_df.sort_values("PTL_Month").iloc[-1]
current_ptl = float(latest_ptl["PTL Size"])
latest_ptl_month = latest_ptl["PTL_Month"]

theatre_cost_2526, _ = get_theatre_cost_per_minute()
job_plan_capacity = load_latest_job_plan_capacity()

try:
    latest_rtt_wait_bands = load_latest_rtt_wait_band_position(str(RTT_DATA_PATH))
    rtt_wait_band_error = None
except Exception as e:
    latest_rtt_wait_bands = {
        "month": "Not available",
        "source": "Not available",
        "total": current_ptl,
        "waiting_0_18": 0.0,
        "waiting_18_51": current_ptl,
        "waiting_52_plus": 0.0,
        "pct_0_18": 0.0,
        "pct_52_plus": 0.0,
    }
    rtt_wait_band_error = e

current_utilisation = float(capacity["Utilisation"])
actual_session_utilisation = float(capacity["Actual_Session_Utilisation"])
scheduled_minutes = float(capacity["Scheduled_Minutes"])
actual_session_scheduled_minutes = float(capacity["Actual_Session_Scheduled_Minutes"])
touch_minutes = float(capacity["Touch_Minutes"])
total_sessions = float(capacity["Planned_Sessions"])
utilisation_sessions = float(capacity["Utilisation_Sessions"])
actual_sessions = float(capacity["Actual_Sessions"])
cancelled_or_not_run_sessions = float(capacity["Cancelled_Or_Not_Run_Sessions"])
total_cases = float(capacity["Completed_Cases"])
avg_procedure_time = float(capacity["Average_Case_Duration_Minutes"])
observed_weeks = float(capacity["Observed_Weeks"])
sessions_per_week = float(capacity["Sessions_Per_Week"])
planned_sessions_per_week = float(capacity["Planned_Sessions_Per_Week"])
average_session_minutes = float(capacity["Average_Session_Minutes"])
cases_per_week = total_cases / observed_weeks if observed_weeks > 0 else 0
planned_240_sessions_per_week = float(capacity["Planned_240_Session_Equivalents_Per_Week"])
actual_240_sessions_per_week = float(capacity["Actual_240_Session_Equivalents_Per_Week"])
utilisation_240_sessions_per_week = float(
    capacity["Utilisation_240_Session_Equivalents_Per_Week"]
)
touch_minutes_per_week = float(capacity["Touch_Minutes_Per_Week"])
raw_theatre_count = (
    theatre_df["Site / Theatre location"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
    if "Site / Theatre location" in theatre_df.columns
    else 0
)


st.sidebar.header("Theatre What-If")

active_delivery_weeks = st.sidebar.number_input(
    "In-year delivery weeks",
    min_value=1.0,
    max_value=52.0,
    value=DEFAULT_DELIVERY_WEEKS,
    step=1.0,
)
estate_theatres = st.sidebar.number_input(
    "Available elective theatres / rooms",
    min_value=0,
    max_value=100,
    value=DEFAULT_ESTATE_THEATRES,
    step=1,
    help=(
        "Default set to the latest PAH-confirmed view of 10 elective theatres "
        f"within a wider estate of {TOTAL_ESTATE_THEATRES} theatres. "
        f"The raw extract has {raw_theatre_count} distinct location labels, "
        "which includes labour ward, Vanguard, cath lab/endoscopy and other labels."
    ),
)
estate_sessions_per_day = st.sidebar.number_input(
    "Estate sessions/theatre/day",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
)
estate_days_per_week = st.sidebar.number_input(
    "Estate operating days/week",
    min_value=0.0,
    max_value=7.0,
    value=5.0,
    step=0.5,
)
estate_capacity_weeks = st.sidebar.number_input(
    "Estate capacity weeks",
    min_value=1.0,
    max_value=52.0,
    value=ESTATE_CAPACITY_WEEKS,
    step=1.0,
)

with st.sidebar.expander("Finance assumptions"):
    agency_reduction_pct = st.slider(
        "Agency spend reduction",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
    )
    wli_reduction_pct = st.slider(
        "WLI / outsourcing reduction",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
    )
    medinet_reduction_pct = st.slider(
        "Endoscopy outsourcing reduction",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
    )
    vanguard_2526_value = st.number_input(
        "Vanguard 25/26 finance value",
        min_value=0.0,
        max_value=20_000_000.0,
        value=3_000_000.0,
        step=50_000.0,
    )
    vanguard_2627_commitment = st.number_input(
        "Vanguard 26/27 commitment",
        min_value=0.0,
        max_value=10_000_000.0,
        value=1_250_000.0,
        step=50_000.0,
    )

with st.sidebar.expander("Outpatient assumptions"):
    outpatient_horizon_months = st.number_input(
        "Outpatient horizon months",
        min_value=1,
        max_value=24,
        value=10,
        step=1,
    )
    outpatient_clinic_sessions_per_week = st.number_input(
        "Clinic sessions/week",
        min_value=0.0,
        max_value=1000.0,
        value=40.0,
        step=1.0,
    )
    outpatient_patients_per_session = st.number_input(
        "Patients/session",
        min_value=0.0,
        max_value=100.0,
        value=9.0,
        step=1.0,
    )
    outpatient_template_current_fill_pct = st.number_input(
        "Current template fill proxy %",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=0.5,
    )
    outpatient_template_target_fill_pct = st.number_input(
        "Template fill planning value %",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=0.5,
    )
    outpatient_template_rtt_share_pct = st.slider(
        "RTT-relevant template share %",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )
    outpatient_eligible_follow_up_per_week = st.number_input(
        "Eligible FU appts/week",
        min_value=0.0,
        max_value=10000.0,
        value=240.0,
        step=5.0,
    )
    outpatient_eligible_new_per_week = st.number_input(
        "Eligible new appts/week",
        min_value=0.0,
        max_value=10000.0,
        value=80.0,
        step=5.0,
    )
    outpatient_current_dna_rate_pct = st.number_input(
        "Current DNA rate %",
        min_value=0.0,
        max_value=100.0,
        value=7.0,
        step=0.5,
    )
    outpatient_target_dna_rate_pct = st.number_input(
        "DNA planning value %",
        min_value=0.0,
        max_value=100.0,
        value=3.0,
        step=0.5,
    )
    outpatient_pifu_conversion_pct = st.number_input(
        "FU slots moved via PIFU %",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.5,
    )
    outpatient_fn_ratio_improvement_pct = st.number_input(
        "F:N improvement %",
        min_value=0.0,
        max_value=100.0,
        value=26.0,
        step=0.5,
    )
    outpatient_rtt_conversion_pct = st.slider(
        "Outpatient RTT/PTL conversion %",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
    )
    outpatient_value_per_appointment = st.number_input(
        "Outpatient cost / capacity value per extra appointment",
        min_value=0.0,
        max_value=1000.0,
        value=OUTPATIENT_SLOT_VALUE_DEFAULT,
        step=25.0,
    )

with st.sidebar.expander("Specialty heatmap"):
    min_heatmap_sessions = st.number_input(
        "Minimum specialty sessions",
        min_value=1,
        max_value=500,
        value=20,
        step=5,
    )
    exclude_unknown_specialty = st.checkbox(
        "Exclude Unknown specialty",
        value=False,
    )

model_current_utilisation = current_utilisation
baseline_delivery_sessions = sessions_per_week * active_delivery_weeks
baseline_planned_sessions_context = planned_sessions_per_week * active_delivery_weeks
baseline_actual_sessions = (
    float(capacity["Actual_Sessions_Per_Week"]) * active_delivery_weeks
)
baseline_cancelled_or_not_run_sessions = (
    float(capacity["Cancelled_Or_Not_Run_Sessions_Per_Week"]) * active_delivery_weeks
)
baseline_planned_240_sessions_context = (
    planned_240_sessions_per_week * active_delivery_weeks
)
baseline_actual_240_sessions = actual_240_sessions_per_week * active_delivery_weeks
baseline_planned_240_sessions = (
    utilisation_240_sessions_per_week * active_delivery_weeks
)
baseline_delivery_cases = cases_per_week * active_delivery_weeks
baseline_delivery_cases_per_list = (
    baseline_delivery_cases / baseline_delivery_sessions
    if baseline_delivery_sessions > 0
    else 0
)
baseline_delivery_scheduled_minutes = baseline_planned_240_sessions * SESSION_STANDARD_MINUTES
baseline_actual_scheduled_minutes = (
    baseline_actual_240_sessions * SESSION_STANDARD_MINUTES
)
baseline_touch_minutes = touch_minutes_per_week * active_delivery_weeks
baseline_effective_sessions = baseline_delivery_sessions * model_current_utilisation
baseline_unutilised_sessions = baseline_delivery_sessions - baseline_effective_sessions
baseline_effective_240_sessions = baseline_touch_minutes / SESSION_STANDARD_MINUTES
baseline_unutilised_240_sessions = (
    baseline_planned_240_sessions - baseline_effective_240_sessions
)
estate_capacity_240_sessions = (
    estate_theatres
    * estate_sessions_per_day
    * estate_days_per_week
    * estate_capacity_weeks
)
estate_utilisation = (
    float(capacity["Actual_240_Session_Equivalents"]) / estate_capacity_240_sessions
    if estate_capacity_240_sessions > 0
    else 0
)
workforce_theatre_capacity_240_sessions = (
    job_plan_capacity["theatre_weekly"] * active_delivery_weeks
)
workforce_theatre_capacity_label = (
    f"{format_number(job_plan_capacity['theatre_weekly'])} per week; "
    f"{format_number(workforce_theatre_capacity_240_sessions)} over "
    f"{format_decimal(active_delivery_weeks, 1)} weeks"
)
workforce_theatre_utilisation = (
    baseline_actual_240_sessions / workforce_theatre_capacity_240_sessions
    if workforce_theatre_capacity_240_sessions > 0
    else 0
)
cost_per_scheduled_minute = (
    theatre_cost_2526 / baseline_delivery_scheduled_minutes
    if baseline_delivery_scheduled_minutes > 0
    else 0
)
full_year_cost_per_scheduled_minute = (
    theatre_cost_2526 / full_year_elective_scheduled_minutes
    if full_year_elective_scheduled_minutes > 0
    else 0
)

scenario_outputs = {}
scenario_a_outputs = {}
scenario_b_outputs = {}

for scenario, scenario_utilisation in SCENARIO_TARGETS.items():
    utilisation_uplift = max(scenario_utilisation - model_current_utilisation, 0)
    actual_utilisation_uplift = max(
        scenario_utilisation - actual_session_utilisation,
        0,
    )
    required_minutes_same_activity = (
        baseline_touch_minutes / scenario_utilisation
        if scenario_utilisation > 0
        else 0
    )
    required_240_sessions_same_activity = (
        required_minutes_same_activity / SESSION_STANDARD_MINUTES
    )
    freed_240_sessions = max(
        baseline_planned_240_sessions - required_240_sessions_same_activity,
        0,
    )
    scenario_a_outputs[scenario] = {
        "utilisation": scenario_utilisation,
        "cases": baseline_delivery_cases,
        "required_240_sessions": required_240_sessions_same_activity,
        "freed_240_sessions": freed_240_sessions,
        "cost_opportunity": (
            freed_240_sessions * SESSION_STANDARD_MINUTES * cost_per_scheduled_minute
        ),
    }

    additional_minutes_throughput = (
        baseline_actual_scheduled_minutes * actual_utilisation_uplift
    )
    cases_unlocked_throughput = (
        additional_minutes_throughput / avg_procedure_time
        if avg_procedure_time > 0
        else 0
    )
    rtt_backlog_reduction_throughput = min(cases_unlocked_throughput, current_ptl)
    remaining_ptl_throughput = max(
        current_ptl - rtt_backlog_reduction_throughput,
        0,
    )
    scenario_b_outputs[scenario] = {
        "utilisation": scenario_utilisation,
        "fixed_240_sessions": baseline_actual_240_sessions,
        "additional_minutes": additional_minutes_throughput,
        "cases_unlocked": cases_unlocked_throughput,
        "rtt_backlog_reduction": rtt_backlog_reduction_throughput,
        "total_cases": baseline_delivery_cases + cases_unlocked_throughput,
        "remaining_ptl": remaining_ptl_throughput,
        "ptl_reduction_pct": (
            rtt_backlog_reduction_throughput / current_ptl
            if current_ptl > 0
            else 0
        ),
        "remaining_ptl_pct": (
            remaining_ptl_throughput / current_ptl if current_ptl > 0 else 0
        ),
        "cost_avoidance": (
            additional_minutes_throughput * cost_per_scheduled_minute
        ),
    }

    effective_sessions = baseline_delivery_sessions * scenario_utilisation
    equivalent_sessions_unlocked = baseline_delivery_sessions * utilisation_uplift
    unutilised_sessions_remaining = baseline_delivery_sessions - effective_sessions
    additional_minutes = (
        baseline_delivery_sessions
        * average_session_minutes
        * utilisation_uplift
    )
    cases_unlocked = (
        additional_minutes / avg_procedure_time if avg_procedure_time > 0 else 0
    )
    scenario_cases = baseline_delivery_cases + cases_unlocked
    scenario_cases_per_list = (
        scenario_cases / baseline_delivery_sessions
        if baseline_delivery_sessions > 0
        else 0
    )
    remaining_ptl = max(current_ptl - cases_unlocked, 0)
    ptl_reduction_pct = cases_unlocked / current_ptl if current_ptl > 0 else 0
    remaining_ptl_pct = remaining_ptl / current_ptl if current_ptl > 0 else 0
    cost_avoidance = additional_minutes * cost_per_scheduled_minute

    scenario_outputs[scenario] = {
        "utilisation": scenario_utilisation,
        "utilisation_uplift": utilisation_uplift,
        "effective_sessions": effective_sessions,
        "equivalent_sessions_unlocked": equivalent_sessions_unlocked,
        "unutilised_sessions_remaining": unutilised_sessions_remaining,
        "additional_minutes": additional_minutes,
        "cases_unlocked": cases_unlocked,
        "total_cases": scenario_cases,
        "cases_per_list": scenario_cases_per_list,
        "remaining_ptl": remaining_ptl,
        "ptl_reduction_pct": ptl_reduction_pct,
        "remaining_ptl_pct": remaining_ptl_pct,
        "cost_avoidance": cost_avoidance,
    }

full_year_more_throughput_outputs = {}
full_year_same_throughput_outputs = {}
for scenario, scenario_utilisation in SCENARIO_TARGETS.items():
    full_year_utilisation_uplift = max(
        scenario_utilisation - full_year_elective_utilisation,
        0,
    )
    full_year_additional_minutes = (
        full_year_elective_scheduled_minutes * full_year_utilisation_uplift
    )
    full_year_additional_cases = (
        full_year_additional_minutes / avg_procedure_time
        if avg_procedure_time > 0
        else 0
    )
    full_year_backlog_reduction = min(full_year_additional_cases, current_ptl)
    full_year_closing_backlog = max(current_ptl - full_year_backlog_reduction, 0)
    full_year_more_throughput_outputs[scenario] = {
        "additional_minutes": full_year_additional_minutes,
        "cases_unlocked": full_year_additional_cases,
        "total_cases": full_year_elective_completed_cases
        + full_year_additional_cases,
        "rtt_backlog_reduction": full_year_backlog_reduction,
        "remaining_ptl": full_year_closing_backlog,
        "ptl_reduction_pct": (
            full_year_backlog_reduction / current_ptl if current_ptl > 0 else 0
        ),
        "remaining_ptl_pct": (
            full_year_closing_backlog / current_ptl if current_ptl > 0 else 0
        ),
        "cost_avoidance": (
            full_year_additional_minutes * full_year_cost_per_scheduled_minute
        ),
    }

    full_year_required_minutes = (
        full_year_elective_touch_minutes / scenario_utilisation
        if scenario_utilisation > 0
        else 0
    )
    full_year_required_240_sessions = (
        full_year_required_minutes / SESSION_STANDARD_MINUTES
    )
    full_year_freed_240_sessions = max(
        full_year_elective_240_session_equivalents - full_year_required_240_sessions,
        0,
    )
    full_year_same_throughput_outputs[scenario] = {
        "required_240_sessions": full_year_required_240_sessions,
        "freed_240_sessions": full_year_freed_240_sessions,
        "cost_opportunity": (
            full_year_freed_240_sessions
            * SESSION_STANDARD_MINUTES
            * full_year_cost_per_scheduled_minute
        ),
    }


def scenario_value(metric: str, scenario: str) -> str:
    output = scenario_outputs[scenario]

    if metric == "utilisation":
        return format_percent(output["utilisation"])
    if metric == "total_cases":
        return format_number(output["total_cases"])
    if metric == "cases_per_list":
        return format_decimal(output["cases_per_list"], 2)
    if metric == "cases_unlocked":
        return format_number(output["cases_unlocked"])
    if metric == "additional_minutes":
        return format_number(output["additional_minutes"])
    if metric == "effective_sessions":
        return format_number(output["effective_sessions"])
    if metric == "equivalent_sessions_unlocked":
        return format_number(output["equivalent_sessions_unlocked"])
    if metric == "unutilised_sessions_remaining":
        return format_number(output["unutilised_sessions_remaining"])
    if metric == "remaining_ptl":
        return format_number(output["remaining_ptl"])
    if metric == "ptl_reduction_pct":
        return format_percent(output["ptl_reduction_pct"])
    if metric == "remaining_ptl_pct":
        return format_percent(output["remaining_ptl_pct"])
    if metric == "cost_avoidance":
        return format_currency(output["cost_avoidance"])
    if metric == "ptl_reduction":
        return f"-{format_number(output['cases_unlocked'])}"

    return ""


outpatient_error = None
outpatient_baseline_monthly = 0.0
outpatient_baseline_total = 0.0
outpatient_latest_month_label = "Not available"
outpatient_month_count = 0
outpatient_baseline_df = pd.DataFrame()
outpatient_baseline_period_label = (
    f"{OUTPATIENT_BASELINE_START.strftime('%b %Y')} to "
    f"{OUTPATIENT_BASELINE_END.strftime('%b %Y')}"
)

try:
    outpatient_df = load_outpatient_data()
    outpatient_baseline_df = outpatient_df[
        (outpatient_df["Contact_Start"] >= OUTPATIENT_BASELINE_START)
        & (outpatient_df["Contact_Start"] < OUTPATIENT_BASELINE_END + pd.Timedelta(days=1))
    ].copy()
    outpatient_monthly_df = summarise_outpatient_attendances_by_month(
        outpatient_baseline_df
    )
    outpatient_visit_type_df = summarise_outpatient_attendances_by_month_visit_type(
        outpatient_baseline_df
    )

    if not outpatient_monthly_df.empty:
        outpatient_baseline_monthly = float(
            outpatient_monthly_df["Outpatient Attendances"].mean()
        )
        outpatient_month_count = len(outpatient_monthly_df)
        outpatient_latest_month_label = outpatient_monthly_df[
            "Contact_Month"
        ].max().strftime("%B %Y")

    outpatient_baseline_total = (
        outpatient_baseline_monthly * outpatient_horizon_months
    )
except Exception as e:
    outpatient_error = e
    outpatient_df = pd.DataFrame()
    outpatient_baseline_df = pd.DataFrame()
    outpatient_monthly_df = pd.DataFrame()
    outpatient_visit_type_df = pd.DataFrame()

outpatient_baseline = build_outpatient_baseline(outpatient_baseline_df)
outpatient_planned_appointments_per_week = (
    outpatient_baseline["planned_appointments"]
    / outpatient_baseline["observed_weeks"]
)
outpatient_attended_appointments_per_week = (
    outpatient_baseline["attended_appointments"]
    / outpatient_baseline["observed_weeks"]
)
outpatient_planned_sessions_per_week = (
    outpatient_baseline["planned_sessions"] / outpatient_baseline["observed_weeks"]
)
outpatient_actual_sessions_per_week = (
    outpatient_baseline["actual_sessions"] / outpatient_baseline["observed_weeks"]
)
outpatient_follow_up_per_week = (
    outpatient_baseline["follow_up"] / outpatient_baseline["observed_weeks"]
)
outpatient_first_attendance_per_week = (
    outpatient_baseline["first_attendance"] / outpatient_baseline["observed_weeks"]
)
outpatient_model_current_fill_pct = (
    outpatient_baseline["fill_rate"] * 100
    if outpatient_baseline["planned_appointments"] > 0
    else outpatient_template_current_fill_pct
)
outpatient_model_current_dna_rate_pct = (
    outpatient_baseline["dna_rate"] * 100
    if outpatient_baseline["planned_appointments"] > 0
    else outpatient_current_dna_rate_pct
)
outpatient_model_clinic_sessions_per_week = (
    outpatient_planned_appointments_per_week / outpatient_patients_per_session
    if outpatient_patients_per_session > 0
    and outpatient_planned_appointments_per_week > 0
    else outpatient_clinic_sessions_per_week
)
outpatient_model_follow_up_per_week = (
    outpatient_follow_up_per_week
    if outpatient_follow_up_per_week > 0
    else outpatient_eligible_follow_up_per_week
)
outpatient_model_first_attendance_per_week = (
    outpatient_first_attendance_per_week
    if outpatient_first_attendance_per_week > 0
    else outpatient_eligible_new_per_week
)
outpatient_planned_appointments_horizon = (
    outpatient_planned_appointments_per_week * active_delivery_weeks
)
outpatient_attended_appointments_horizon = (
    outpatient_attended_appointments_per_week * active_delivery_weeks
)
outpatient_planned_sessions_horizon = (
    outpatient_planned_sessions_per_week * active_delivery_weeks
)
outpatient_actual_sessions_horizon = (
    outpatient_actual_sessions_per_week * active_delivery_weeks
)
workforce_outpatient_capacity_240_sessions = (
    job_plan_capacity["outpatient_weekly"] * active_delivery_weeks
)
workforce_outpatient_utilisation = (
    outpatient_actual_sessions_horizon / workforce_outpatient_capacity_240_sessions
    if workforce_outpatient_capacity_240_sessions > 0
    else 0
)

outpatient_outputs = calculate_outpatient_scenario_outputs(
    clinic_sessions_per_week=outpatient_model_clinic_sessions_per_week,
    patients_per_session=outpatient_patients_per_session,
    template_current_fill=outpatient_model_current_fill_pct / 100,
    template_target_fill=outpatient_template_target_fill_pct / 100,
    template_rtt_relevant_share=outpatient_template_rtt_share_pct / 100,
    eligible_new_per_week=outpatient_model_first_attendance_per_week,
    eligible_follow_up_per_week=outpatient_model_follow_up_per_week,
    current_dna_rate=outpatient_model_current_dna_rate_pct / 100,
    target_dna_rate=outpatient_target_dna_rate_pct / 100,
    pifu_conversion_rate=outpatient_pifu_conversion_pct / 100,
    fn_ratio_improvement_rate=outpatient_fn_ratio_improvement_pct / 100,
    active_weeks=active_delivery_weeks,
    horizon_months=int(outpatient_horizon_months),
    baseline_monthly_attendances=outpatient_baseline_monthly,
    current_ptl_value=current_ptl,
    rtt_conversion=outpatient_rtt_conversion_pct / 100,
    value_per_appointment=outpatient_value_per_appointment,
)


def outpatient_value(metric: str, scenario: str) -> str:
    output = outpatient_outputs[scenario]

    if metric == "template_fill_target":
        return format_percent(output["template_fill_target"])
    if metric == "dna_rate_target":
        return format_percent(output["dna_rate_target"])
    if metric == "pifu_target":
        return format_percent(output["pifu_target"])
    if metric == "fn_ratio_target":
        return format_percent(output["fn_ratio_target"])
    if metric == "template_fill_monthly":
        return format_number(output["template_fill_monthly"])
    if metric == "dna_reduction_monthly":
        return format_number(output["dna_reduction_monthly"])
    if metric == "pifu_monthly":
        return format_number(output["pifu_monthly"])
    if metric == "fn_ratio_monthly":
        return format_number(output["fn_ratio_monthly"])
    if metric == "additional_monthly":
        return format_number(output["additional_monthly"])
    if metric == "additional_total":
        return format_number(output["additional_total"])
    if metric == "activity_after_monthly":
        return format_number(output["activity_after_monthly"])
    if metric == "activity_after_total":
        return format_number(output["activity_after_total"])
    if metric == "converted_ptl_impact":
        return format_number(output["converted_ptl_impact"])
    if metric == "remaining_ptl":
        return format_number(output["remaining_ptl"])
    if metric == "ptl_reduction_pct":
        return format_percent(output["ptl_reduction_pct"])
    if metric == "remaining_ptl_pct":
        return format_percent(output["remaining_ptl_pct"])
    if metric == "financial_opportunity":
        return format_currency(output["financial_opportunity"])
    if metric == "income_opportunity":
        return format_currency(output["additional_total"] * outpatient_income_per_attendance)

    return ""


outpatient_income_2526 = 0.0
outpatient_cost_2526 = 0.0
outpatient_income_per_attendance = 0.0
outpatient_cost_per_attendance = 0.0
outpatient_finance_evidence = "Not available"

try:
    outpatient_finance_tb_df = load_trial_balance()
    outpatient_finance_search = (
        outpatient_finance_tb_df["Search_Text"].fillna("").astype(str)
    )
    outpatient_finance_mask = outpatient_finance_search.str.contains(
        "outpatients|outpatient|out patient|opd|clinic",
        regex=True,
        na=False,
    )
    outpatient_income_mask = (
        outpatient_finance_tb_df["Expenditure Type"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("income")
    )
    outpatient_income_2526 = sum_abs(
        outpatient_finance_tb_df.loc[
            outpatient_finance_mask & outpatient_income_mask,
            "FY_2526_Total",
        ]
    )
    outpatient_cost_2526 = sum_abs(
        outpatient_finance_tb_df.loc[
            outpatient_finance_mask & ~outpatient_income_mask,
            "FY_2526_Total",
        ]
    )
    outpatient_2526_df = outpatient_baseline_df.copy()
    outpatient_2526_status = (
        outpatient_2526_df["Status"].fillna("").astype(str).str.strip().str.lower()
        if "Status" in outpatient_2526_df.columns
        else pd.Series("", index=outpatient_2526_df.index)
    )
    outpatient_2526_attended_mask = (
        outpatient_2526_status.str.contains("checked in")
        | outpatient_2526_status.str.contains("check in")
        | outpatient_2526_status.str.contains("checked out")
        | outpatient_2526_status.str.contains("check out")
    )
    outpatient_attended_2526 = float(
        outpatient_2526_df.loc[
            outpatient_2526_attended_mask,
            "Contact_ID",
        ].nunique()
    )
    outpatient_income_per_attendance = (
        outpatient_income_2526 / outpatient_attended_2526
        if outpatient_attended_2526 > 0
        else 0.0
    )
    outpatient_cost_per_attendance = (
        outpatient_cost_2526 / outpatient_attended_2526
        if outpatient_attended_2526 > 0
        else 0.0
    )
    outpatient_finance_evidence = (
        f"25/26 outpatient income keyword proxy {format_currency(outpatient_income_2526)} "
        f"and cost keyword proxy {format_currency(outpatient_cost_2526)} divided by "
        f"{format_number(outpatient_attended_2526)} attended contacts in Apr 2025-Mar 2026."
    )
except Exception:
    outpatient_finance_evidence = (
        "Income/cost per attendance could not be derived from trial balance and "
        "outpatient contact data."
    )


financial_error = None
financial_rows = []
other_financial_benefit = 0.0

try:
    tb_df = load_trial_balance()
    staff_df = load_staff_cost_data()

    staff_2526_df = staff_df[staff_df["Financial_Year"] == "25/26"].copy()
    staff_surgical_df = staff_2526_df[surgical_theatre_mask(staff_2526_df)].copy()

    tb_cost_mask = (
        tb_df["Expenditure Type"].fillna("").astype(str).str.lower() != "income"
    )
    tb_surgical_df = tb_df[tb_cost_mask & surgical_theatre_mask(tb_df)].copy()

    agency_df = staff_surgical_df[
        staff_surgical_df["Pay type"].str.lower() == "agency"
    ].copy()
    agency_spend_2526 = sum_abs(agency_df["Total cost"])
    agency_benefit = agency_spend_2526 * agency_reduction_pct / 100

    wli_pattern = (
        "waiting list|wli|insourcing|outsourcing|independent sector|independent"
    )
    wli_df = tb_surgical_df[
        tb_surgical_df["Search_Text"].str.contains(
            wli_pattern,
            regex=True,
            na=False,
        )
    ].copy()
    wli_spend_2526 = sum_abs(wli_df["FY_2526_Total"])
    wli_benefit = wli_spend_2526 * wli_reduction_pct / 100

    medinet_df = tb_df[
        tb_df["Search_Text"].str.contains("medinet", regex=False, na=False)
        & tb_cost_mask
    ].copy()
    medinet_source = "Search_Text contains Medinet"

    if medinet_df.empty:
        medinet_df = tb_df[
            tb_df["Search_Text"].str.contains("endoscopy", regex=False, na=False)
            & tb_df["Search_Text"].str.contains(
                "non-nhs|nonpat care|other bodies|purchase of healthcare",
                regex=True,
                na=False,
            )
            & tb_cost_mask
        ].copy()
        medinet_source = (
            "Medinet not separately identifiable; proxy uses Endoscopy + "
            "Non-NHS / NonPat Care / Purchase of Healthcare rows"
        )

    medinet_spend_2526 = sum_abs(medinet_df["FY_2526_Total"])
    medinet_benefit = medinet_spend_2526 * medinet_reduction_pct / 100

    vanguard_df = tb_df[
        tb_df["Search_Text"].str.contains("vanguard", regex=False, na=False)
        & tb_cost_mask
    ].copy()
    vanguard_raw_spend_2526 = sum_abs(vanguard_df["FY_2526_Total"])
    vanguard_benefit = max(vanguard_2526_value - vanguard_2627_commitment, 0)
    vanguard_capacity = summarise_vanguard_capacity_impact(theatre_df)
    vanguard_cases_2526 = float(vanguard_capacity.get("Completed_Cases", 0))
    vanguard_sessions_2526 = float(vanguard_capacity.get("Sessions", 0))
    vanguard_primary_specialty = str(
        vanguard_capacity.get("Primary_Specialty", "Not available")
    )
    vanguard_primary_specialty_cases = float(
        vanguard_capacity.get("Primary_Specialty_Cases", 0)
    )
    vanguard_backlog_impact = format_vanguard_backlog_capacity_impact(
        vanguard_cases_2526,
        vanguard_sessions_2526,
        current_ptl,
        vanguard_primary_specialty,
        vanguard_primary_specialty_cases,
    )

    workforce_spend_2526 = sum_abs(staff_surgical_df["Total cost"])
    monthly_wte_df = (
        staff_surgical_df.groupby("Year/Month", as_index=False)
        .agg(WTE=("WTE equivalent", lambda values: values.abs().sum()))
        .sort_values("Year/Month")
    )
    average_workforce_wte = (
        float(monthly_wte_df["WTE"].mean()) if not monthly_wte_df.empty else 0
    )

    financial_rows = [
        {
            "Opportunity": "Agency spend - surgical / theatres",
            "25/26 actual spend": format_currency(agency_spend_2526),
            "Indicative 26/27 opportunity": format_currency(agency_benefit),
            "Financial category": "Cost avoidance / cashable benefit",
            "Calculation / evidence": (
                f"{format_currency(agency_spend_2526)} x "
                f"{agency_reduction_pct}% reduction. Evidence: "
                f"{len(agency_df):,} staff-cost rows where Pay type = Agency "
                "and surgical/theatre filter is true."
            ),
            "_actual": agency_spend_2526,
            "_benefit": agency_benefit,
        },
        {
            "Opportunity": "WLI / insourcing / outsourcing / independent sector",
            "25/26 actual spend": format_currency(wli_spend_2526),
            "Indicative 26/27 opportunity": format_currency(wli_benefit),
            "Financial category": "Cost avoidance / cashable if budgeted",
            "Calculation / evidence": (
                f"{format_currency(wli_spend_2526)} x {wli_reduction_pct}% "
                f"reduction. Evidence: {len(wli_df):,} trial-balance rows "
                "matched to surgical/theatre plus WLI / outsourcing keywords."
            ),
            "_actual": wli_spend_2526,
            "_benefit": wli_benefit,
        },
        {
            "Opportunity": "Known: Endoscopy Medinet outsourcing",
            "25/26 actual spend": format_currency(medinet_spend_2526),
            "Indicative 26/27 opportunity": format_currency(medinet_benefit),
            "Financial category": "Cost avoidance / cashable if budgeted",
            "Calculation / evidence": (
                f"{format_currency(medinet_spend_2526)} x "
                f"{medinet_reduction_pct}% reduction / cease use. Evidence: "
                f"{len(medinet_df):,} trial-balance rows. Source rule: "
                f"{medinet_source}."
            ),
            "_actual": medinet_spend_2526,
            "_benefit": medinet_benefit,
        },
        {
            "Opportunity": "Vanguard theatre capacity",
            "25/26 actual spend": format_currency(vanguard_2526_value),
            "Indicative 26/27 opportunity": format_currency(vanguard_benefit),
            "Financial category": "Cost avoidance / cashable if budgeted",
            "Calculation / evidence": (
                f"Finance-provided 25/26 value {format_currency(vanguard_2526_value)} "
                f"less 26/27 commitment {format_currency(vanguard_2627_commitment)}. "
                f"Raw trial-balance Vanguard keyword match identifies "
                f"{format_currency(vanguard_raw_spend_2526)} across "
                f"{len(vanguard_df):,} rows."
            ),
            "Backlog / capacity impact": vanguard_backlog_impact,
            "_actual": vanguard_2526_value,
            "_benefit": vanguard_benefit,
        },
        {
            "Opportunity": "Workforce - surgical / theatre FTE baseline",
            "25/26 actual spend": (
                f"{format_currency(workforce_spend_2526)}; "
                f"avg WTE {average_workforce_wte:,.1f}"
            ),
            "Indicative 26/27 opportunity": "Requires budgeted FTE / establishment",
            "Financial category": "Cashable",
            "Calculation / evidence": (
                "Raw staff-cost data gives actual workforce cost and WTE, but "
                "does not include budgeted establishment needed to calculate a "
                "cashable opportunity."
            ),
            "_actual": 0,
            "_benefit": 0,
        },
    ]

    for row in financial_rows:
        row.setdefault(
            "Backlog / capacity impact",
            "Not directly quantified from this finance row.",
        )

    quantified_rows = [
        row for row in financial_rows if isinstance(row["_benefit"], (int, float))
    ]
    total_actual = sum(row["_actual"] for row in quantified_rows)
    other_financial_benefit = sum(row["_benefit"] for row in quantified_rows)
    financial_rows.append(
        {
            "Opportunity": "Total quantified finance opportunity excluding theatre-utilisation scenario",
            "25/26 actual spend": format_currency(total_actual),
            "Indicative 26/27 opportunity": format_currency(other_financial_benefit),
            "Financial category": "",
            "Calculation / evidence": (
                "Sum of quantified finance rows above. Theatre utilisation "
                "scenario opportunity is shown separately to avoid hidden double counting."
            ),
            "Backlog / capacity impact": (
                "Not additive. Vanguard capacity impact is shown separately because "
                "it is capacity to replace, not a backlog reduction saving."
            ),
            "_actual": total_actual,
            "_benefit": other_financial_benefit,
        }
    )
except Exception as e:
    financial_error = e


table_rows = [
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Current utilisation",
        "Current baseline": format_percent(current_utilisation),
        "What if 50% delivery": scenario_value(
            "utilisation",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "utilisation",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "utilisation",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Current utilisation = valid case touch minutes / scheduled session "
            f"minutes = {format_number(touch_minutes)} / "
            f"{format_number(scheduled_minutes)} = "
            f"{format_percent(current_utilisation)}. What-if columns are fixed "
            "utilisation end states: 78.5%, 81.75%, and 85%."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Scheduled sessions (physical lists)",
        "Current baseline": format_number(baseline_delivery_sessions),
        "What if 50% delivery": format_number(baseline_delivery_sessions),
        "What if 75% delivery": format_number(baseline_delivery_sessions),
        "What if 100% delivery": format_number(baseline_delivery_sessions),
        "Notes / calculation": (
            f"Physical scheduled lists are held constant because this scenario "
            f"models better use of existing sessions, not extra lists. "
            f"Calculation = sessions per week x in-year delivery weeks = "
            f"{format_decimal(sessions_per_week, 2)} x "
            f"{format_decimal(active_delivery_weeks, 1)} = "
            f"{format_number(baseline_delivery_sessions)}."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Effective utilised sessions",
        "Current baseline": format_number(baseline_effective_sessions),
        "What if 50% delivery": scenario_value(
            "effective_sessions",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "effective_sessions",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "effective_sessions",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Effective utilised sessions = scheduled sessions x utilisation. "
            "This is the modelled session movement and converts utilisation "
            "into the equivalent number of fully used sessions."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": scenario_value(
            "cases_unlocked",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "cases_unlocked",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "cases_unlocked",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional case volume = additional utilised minutes / average "
            "procedure time. This expresses the utilisation opportunity as "
            "extra cases rather than session capacity."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Unutilised session equivalent remaining",
        "Current baseline": format_number(baseline_unutilised_sessions),
        "What if 50% delivery": scenario_value(
            "unutilised_sessions_remaining",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "unutilised_sessions_remaining",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "unutilised_sessions_remaining",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Remaining unutilised equivalent sessions = scheduled sessions - "
            "effective utilised sessions."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Total cases",
        "Current baseline": format_number(baseline_delivery_cases),
        "What if 50% delivery": scenario_value(
            "total_cases",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "total_cases",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "total_cases",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Baseline cases = full-period completed case run rate x in-year "
            f"delivery weeks = {format_decimal(cases_per_week, 1)} x "
            f"{format_decimal(active_delivery_weeks, 1)} = "
            f"{format_number(baseline_delivery_cases)}. What-if total cases = "
            "baseline cases + additional case volume."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Cases per list",
        "Current baseline": format_decimal(baseline_delivery_cases_per_list, 2),
        "What if 50% delivery": scenario_value(
            "cases_per_list",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "cases_per_list",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "cases_per_list",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Cases per list = total cases / scheduled physical sessions. This "
            "increases in the what-if columns because cases increase while "
            "scheduled physical sessions are held constant."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Average procedure time",
        "Current baseline": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 50% delivery": "Held constant",
        "What if 75% delivery": "Held constant",
        "What if 100% delivery": "Held constant",
        "Notes / calculation": (
            f"Average procedure time = valid case touch minutes / cases with "
            f"valid touch time = {format_number(touch_minutes)} / "
            f"{format_number(float(capacity['Cases_With_Valid_Touch']))} = "
            f"{format_decimal(avg_procedure_time, 1)} minutes. Touch-time rows "
            "over 720 minutes are excluded as outliers."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "PTL after additional cases",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": scenario_value(
            "remaining_ptl",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "remaining_ptl",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "remaining_ptl",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Latest PTL month: {latest_ptl_month.strftime('%B %Y')}. "
            "What-if PTL = current PTL - additional case volume."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Additional utilised minutes",
        "Current baseline": "0",
        "What if 50% delivery": scenario_value(
            "additional_minutes",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "additional_minutes",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "additional_minutes",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional utilised minutes = scheduled sessions x average session "
            "minutes x utilisation uplift. Utilisation uplift is the difference "
            "between current utilisation and the scenario end state."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": scenario_value(
            "cases_unlocked",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "cases_unlocked",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "cases_unlocked",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional case volume = additional utilised minutes / average procedure "
            "time. This assumes the extra utilised time can be converted into "
            "completed cases at the observed average case duration."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "PTL variance / reduction",
        "Current baseline": "0",
        "What if 50% delivery": scenario_value(
            "ptl_reduction",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "ptl_reduction",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "ptl_reduction",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "PTL variance = -additional case volume. Assumption: each additional case "
            "removes one pathway from the PTL."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "PTL reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": scenario_value(
            "ptl_reduction_pct",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "ptl_reduction_pct",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "ptl_reduction_pct",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "PTL reduction % = additional case volume / current PTL. This shows the "
            "proportional backlog impact of the theatre utilisation improvement."
        ),
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Remaining PTL %",
        "Current baseline": "100.0%",
        "What if 50% delivery": scenario_value(
            "remaining_ptl_pct",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "remaining_ptl_pct",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "remaining_ptl_pct",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Remaining PTL % = PTL after additional cases / current PTL.",
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Indicative financial opportunity",
        "Current baseline": "£0",
        "What if 50% delivery": scenario_value(
            "cost_avoidance",
            "What if 50% delivery",
        ),
        "What if 75% delivery": scenario_value(
            "cost_avoidance",
            "What if 75% delivery",
        ),
        "What if 100% delivery": scenario_value(
            "cost_avoidance",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional utilised minutes x theatre/anaesthetic cost per scheduled "
            "minute. Cost per scheduled minute = 25/26 theatre and anaesthetic "
            "spend / in-year scheduled minutes. This is an opportunity proxy, "
            "not automatically cashable."
        ),
    },
]

table_rows = [
    {
        "Opportunity": "Definitions and baseline",
        "Metric": "Baseline label",
        "Current baseline": (
            f"Average of {capacity['Recent_Start_Date'].strftime('%b %Y')} to "
            f"{capacity['Recent_End_Date'].strftime('%b %Y')}, annualised over "
            f"{format_decimal(active_delivery_weeks, 1)} weeks"
        ),
        "What if 50% delivery": "Same baseline",
        "What if 75% delivery": "Same baseline",
        "What if 100% delivery": "Same baseline",
        "Notes / calculation": (
            "No mixed-period baseline: the model uses the full available theatre "
            "history and annualises the observed weekly run rate."
        ),
        "Actions": "Confirm whether final pack should instead use a single-year baseline, e.g. Apr 2025 to Mar 2026.",
    },
    {
        "Opportunity": "Definitions and baseline",
        "Metric": "Session standard",
        "Current baseline": f"{SESSION_STANDARD_MINUTES} minutes",
        "What if 50% delivery": f"{SESSION_STANDARD_MINUTES} minutes",
        "What if 75% delivery": f"{SESSION_STANDARD_MINUTES} minutes",
        "What if 100% delivery": f"{SESSION_STANDARD_MINUTES} minutes",
        "Notes / calculation": (
            "All session measures are converted to 240-minute equivalent units. "
            "A 480-minute list is therefore counted as 2.0 session equivalents."
        ),
        "Actions": "Keep this standard in the slide footnote so 240 vs 480 minute lists are not mixed.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Planned sessions",
        "Current baseline": format_number(baseline_delivery_sessions),
        "What if 50% delivery": "Held constant",
        "What if 75% delivery": "Held constant",
        "What if 100% delivery": "Held constant",
        "Notes / calculation": (
            "Planned sessions = total sessions scheduled to run. It is the "
            "unique scheduled theatre session count, annualised from the raw "
            "theatre system."
        ),
        "Actions": "Validate that the theatre extract includes all cancelled sessions, not only sessions with at least one booked case.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Actual sessions used for utilisation",
        "Current baseline": format_number(baseline_actual_sessions),
        "What if 50% delivery": "Held constant in Scenario B",
        "What if 75% delivery": "Held constant in Scenario B",
        "What if 100% delivery": "Held constant in Scenario B",
        "Notes / calculation": (
            "Actual sessions used for utilisation = delivered sessions with "
            "touch time, completed cases, actual start, or actual finish recorded."
        ),
        "Actions": "Check with theatres whether this is the agreed definition of a session that actually ran.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Cancelled / not-run sessions",
        "Current baseline": format_number(baseline_cancelled_or_not_run_sessions),
        "What if 50% delivery": "Not modelled",
        "What if 75% delivery": "Not modelled",
        "What if 100% delivery": "Not modelled",
        "Notes / calculation": (
            "Cancelled / not-run sessions = planned sessions - actual sessions. "
            "Cancelled cases are captured separately and do not necessarily mean "
            "the whole session was cancelled."
        ),
        "Actions": "Request a definitive cancelled-session flag if available from the theatre system.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Planned sessions, 240-min equivalents",
        "Current baseline": format_number(baseline_planned_240_sessions),
        "What if 50% delivery": "Held constant in Scenario B",
        "What if 75% delivery": "Held constant in Scenario B",
        "What if 100% delivery": "Held constant in Scenario B",
        "Notes / calculation": (
            f"Scheduled minutes / {SESSION_STANDARD_MINUTES}. This is the core "
            "unit used for comparing lists of different lengths."
        ),
        "Actions": "Use this row rather than raw list count when comparing 240-minute and longer lists.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Actual sessions, 240-min equivalents",
        "Current baseline": format_number(baseline_actual_240_sessions),
        "What if 50% delivery": "Held constant in Scenario B",
        "What if 75% delivery": "Held constant in Scenario B",
        "What if 100% delivery": "Held constant in Scenario B",
        "Notes / calculation": (
            f"Scheduled minutes for sessions that ran / {SESSION_STANDARD_MINUTES}."
        ),
        "Actions": "Use as the fixed session base for the throughput scenario.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Derived theatre utilisation",
        "Current baseline": format_percent(current_utilisation),
        "What if 50% delivery": "78.5%",
        "What if 75% delivery": "81.8%",
        "What if 100% delivery": "85.0%",
        "Notes / calculation": (
            "Theatre utilisation = valid case touch time / scheduled minutes = "
            f"{format_number(baseline_touch_minutes)} / "
            f"{format_number(baseline_delivery_scheduled_minutes)}."
        ),
        "Actions": "Keep this minutes-based definition fixed across all theatre scenarios.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Total cases",
        "Current baseline": format_number(baseline_delivery_cases),
        "What if 50% delivery": "Held constant in Scenario A",
        "What if 75% delivery": "Held constant in Scenario A",
        "What if 100% delivery": "Held constant in Scenario A",
        "Notes / calculation": (
            "Baseline cases = observed completed cases per week x delivery weeks. "
            "Scenario A keeps cases constant; Scenario B increases cases."
        ),
        "Actions": "Confirm whether completed cases should be all procedures or RTT-relevant elective cases only.",
    },
    {
        "Opportunity": "Capacity layer",
        "Metric": "Estate capacity, 240-min equivalents",
        "Current baseline": format_number(estate_capacity_240_sessions),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            f"{estate_theatres} theatres x {format_decimal(estate_sessions_per_day, 1)} "
            f"sessions/day x {format_decimal(estate_days_per_week, 1)} days/week x "
            f"{format_decimal(active_delivery_weeks, 1)} weeks."
        ),
        "Actions": "Validate estate room count, normal operating days and whether weekend/evening sessions should be included.",
    },
    {
        "Opportunity": "Capacity layer",
        "Metric": "Estate utilisation",
        "Current baseline": format_percent(estate_utilisation),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            "Actual delivered 240-min session equivalents / estate capacity."
        ),
        "Actions": "Use to show whether theatre estate is a constraint before proposing extra activity.",
    },
    {
        "Opportunity": "Capacity layer",
        "Metric": "Workforce capacity from job plans",
        "Current baseline": format_number(workforce_theatre_capacity_240_sessions),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            "Latest substantive job-plan Operating sessions x delivery weeks. "
            f"Source: {job_plan_capacity['source']}."
        ),
        "Actions": "Validate that Operating sessions maps to weekly 240-minute DCC sessions and agree whether anaesthetic rows should be included.",
    },
    {
        "Opportunity": "Capacity layer",
        "Metric": "Workforce utilisation",
        "Current baseline": format_percent(workforce_theatre_utilisation),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            "Actual delivered 240-min session equivalents / substantive "
            "job-plan operating-session capacity."
        ),
        "Actions": "Treat as directional until job-plan scope is validated by specialty and role.",
    },
    {
        "Opportunity": "Scenario A - same activity, fewer sessions",
        "Metric": "Sessions required, 240-min equivalents",
        "Current baseline": format_number(baseline_planned_240_sessions),
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["required_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["required_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["required_240_sessions"]
        ),
        "Notes / calculation": (
            "Keeps case touch minutes and cases constant. Required scheduled "
            "minutes = baseline touch minutes / target utilisation."
        ),
        "Actions": "Use for efficiency and cost opportunity only; do not use this row to claim PTL reduction.",
    },
    {
        "Opportunity": "Scenario A - same activity, fewer sessions",
        "Metric": "Sessions freed, 240-min equivalents",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["freed_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["freed_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["freed_240_sessions"]
        ),
        "Notes / calculation": (
            "Baseline planned session equivalents - sessions required at target "
            "utilisation. Cases are unchanged."
        ),
        "Actions": "Only cashable if freed sessions remove budgeted cost, WLI, outsourcing or temporary capacity.",
    },
    {
        "Opportunity": "Scenario A - same activity, fewer sessions",
        "Metric": "Indicative cost opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_a_outputs["What if 50% delivery"]["cost_opportunity"]
        ),
        "What if 75% delivery": format_currency(
            scenario_a_outputs["What if 75% delivery"]["cost_opportunity"]
        ),
        "What if 100% delivery": format_currency(
            scenario_a_outputs["What if 100% delivery"]["cost_opportunity"]
        ),
        "Notes / calculation": (
            "Sessions freed x 240 minutes x theatre/anaesthetic cost per "
            "scheduled minute."
        ),
        "Actions": "Validate finance treatment; this is cost avoidance unless budgeted spend is removed.",
    },
    {
        "Opportunity": "Scenario B - same sessions, more activity",
        "Metric": "Additional cases",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Keeps actual delivered 240-min sessions constant. Additional cases "
            "= additional utilised minutes / observed average procedure time."
        ),
        "Actions": "Use for throughput and RTT/PTL impact; do not describe this as sessions freed.",
    },
    {
        "Opportunity": "Scenario B - same sessions, more activity",
        "Metric": "PTL after additional cases",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl"]
        ),
        "Notes / calculation": (
            f"Latest PTL month: {latest_ptl_month.strftime('%B %Y')}. "
            "PTL impact assumes one additional case removes one pathway."
        ),
        "Actions": "Validate whether all unlocked cases are RTT-relevant before using as a backlog commitment.",
    },
    {
        "Opportunity": "Scenario B - same sessions, more activity",
        "Metric": "Additional activity value",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cost_avoidance"]
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cost_avoidance"]
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cost_avoidance"]
        ),
        "Notes / calculation": (
            "Additional utilised minutes x theatre/anaesthetic cost per scheduled "
            "minute. This is a value proxy for extra internal throughput."
        ),
        "Actions": "Only treat as finance benefit if it replaces external/premium capacity or avoids future spend.",
    },
    {
        "Opportunity": "Validation checks",
        "Metric": "Actual <= planned",
        "Current baseline": validation_label(
            baseline_actual_240_sessions <= baseline_planned_240_sessions_context
        ),
        "What if 50% delivery": "Pass",
        "What if 75% delivery": "Pass",
        "What if 100% delivery": "Pass",
        "Notes / calculation": (
            "Actual delivered session equivalents must not exceed planned "
            "session equivalents."
        ),
        "Actions": "Investigate if this check fails after source data refresh.",
    },
    {
        "Opportunity": "Validation checks",
        "Metric": "Scenario guardrail",
        "Current baseline": "Applied",
        "What if 50% delivery": "A: cases fixed; B: sessions fixed",
        "What if 75% delivery": "A: cases fixed; B: sessions fixed",
        "What if 100% delivery": "A: cases fixed; B: sessions fixed",
        "Notes / calculation": (
            "Scenario A changes sessions, not cases/PTL. Scenario B changes "
            "cases/PTL, not sessions."
        ),
        "Actions": "Keep these scenarios separate in the slide narrative.",
    },
]

estate_capacity_minutes = estate_capacity_240_sessions * SESSION_STANDARD_MINUTES
current_estate_time_utilisation = (
    touch_minutes / estate_capacity_minutes
    if estate_capacity_minutes > 0
    else 0
)
scenario_estate_time_utilisation = {
    scenario: (
        (scheduled_minutes * SCENARIO_TARGETS[scenario])
        / estate_capacity_minutes
        if estate_capacity_minutes > 0
        else 0
    )
    for scenario, output in scenario_b_outputs.items()
}
scenario_effective_240_sessions = {
    scenario: baseline_planned_240_sessions * target
    for scenario, target in SCENARIO_TARGETS.items()
}
scenario_equivalent_sessions_unlocked = {
    scenario: max(effective_sessions - baseline_effective_240_sessions, 0)
    for scenario, effective_sessions in scenario_effective_240_sessions.items()
}
scenario_unutilised_240_sessions = {
    scenario: max(baseline_planned_240_sessions - effective_sessions, 0)
    for scenario, effective_sessions in scenario_effective_240_sessions.items()
}
actual_sessions_ran_pct = (
    baseline_actual_sessions / baseline_planned_sessions_context
    if baseline_planned_sessions_context > 0
    else 0
)
actual_sessions_vs_estate_capacity = (
    float(capacity["Actual_240_Session_Equivalents"]) / estate_capacity_240_sessions
    if estate_capacity_240_sessions > 0
    else 0
)

baseline_theatre_evidence = theatre_df[
    (theatre_df["Booked Operation Date"] >= THEATRE_BASELINE_START)
    & (theatre_df["Booked Operation Date"] <= THEATRE_BASELINE_END)
].copy()
cancellation_date_rows = (
    int(
        pd.to_datetime(
            baseline_theatre_evidence["Cancellation Date"].replace("NULL", pd.NA),
            errors="coerce",
            dayfirst=True,
        )
        .notna()
        .sum()
    )
    if "Cancellation Date" in baseline_theatre_evidence.columns
    else 0
)
cancellation_reason_rows = 0
if "Cancellation reasons" in baseline_theatre_evidence.columns:
    cancellation_reason_rows = int(
        baseline_theatre_evidence["Cancellation reasons"]
        .replace("NULL", pd.NA)
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .notna()
        .sum()
    )
cancelled_case_rows = (
    int((baseline_theatre_evidence["Cancelled cases"] > 0).sum())
    if "Cancelled cases" in baseline_theatre_evidence.columns
    else 0
)
rtt_columns = [
    col
    for col in theatre_df.columns
    if any(token in col.lower() for token in ["rtt", "pathway", "waiting"])
]

capacity_assumption_rows = [
    {
        "Capacity layer": "Estate",
        "Metric": "Estate capacity available, 240-min sessions",
        "Value": format_number(estate_capacity_240_sessions),
        "Calculation / source": (
            f"{estate_theatres} theatres x "
            f"{format_decimal(estate_sessions_per_day, 1)} sessions/day x "
            f"{format_decimal(estate_days_per_week, 1)} days/week x "
            f"{format_decimal(estate_capacity_weeks, 1)} estate weeks. "
            f"Theatre count uses PAH-confirmed elective estate of "
            f"{DEFAULT_ESTATE_THEATRES}, within a total estate of "
            f"{TOTAL_ESTATE_THEATRES}; raw extract has {raw_theatre_count} "
            "distinct location labels."
        ),
        "Why fixed": (
            "This is the physical theatre supply denominator. It changes only "
            "if theatre count, operating days, sessions per day, or estate "
            "weeks change."
        ),
        "Actions": "Validate if the 10 elective theatres exclude Vanguard, emergency, obstetric and non-theatre procedure rooms.",
    },
    {
        "Capacity layer": "Estate",
        "Metric": "Actual sessions run vs available estate theatre capacity",
        "Value": format_percent(actual_sessions_vs_estate_capacity),
        "Calculation / source": (
            "Elective actual delivered 240-minute session equivalents / available "
            "estate capacity."
        ),
        "Why fixed": (
            "This is a baseline diagnostic. Utilisation targets change the "
            "time used within sessions; they do not change the physical estate "
            "capacity denominator."
        ),
        "Actions": "Use Estate time utilised in the main table to see the scenario movement.",
    },
    {
        "Capacity layer": "Workforce",
        "Metric": "Total substantive operating sessions from job plans",
        "Value": workforce_theatre_capacity_label,
        "Calculation / source": (
            "Sum of `Operating sessions` in the latest job-planning split, "
            "excluding Locum rows. The period total then multiplies that weekly "
            f"amount by {format_decimal(active_delivery_weeks, 1)} active delivery weeks. "
            f"Source: {job_plan_capacity['source']}."
        ),
        "Why fixed": (
            "This is the total substantive operating-session amount available "
            "from the job-plan file, not a theatre-utilisation percentage."
        ),
        "Actions": "Validate whether `Operating sessions` maps to weekly theatre DCC sessions and whether anaesthetic rows should be included.",
    },
]

theatre_baseline_rows = [
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Elective utilisation - actual sessions",
        "Current baseline": format_percent(current_utilisation),
        "What if 50% delivery": "78.5%",
        "What if 75% delivery": "81.8%",
        "What if 100% delivery": "85.0%",
        "Notes / calculation": (
            "Baseline = Apr 2025-Mar 2026 elective, non-obstetric actual-session "
            "touch time / elective actual-session scheduled minutes. Emergency, "
            "mixed, obstetrics, cancelled/not-run and invalid/24-hour sessions "
            "are excluded from the calculation."
        ),
        "Actions": "Use this as the Model Hospital-aligned utilisation denominator until PAH confirms any local exclusions.",
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Effective utilised sessions",
        "Current baseline": format_number(baseline_effective_240_sessions),
        "What if 50% delivery": format_number(
            scenario_effective_240_sessions["What if 50% delivery"]
        ),
        "What if 75% delivery": format_number(
            scenario_effective_240_sessions["What if 75% delivery"]
        ),
        "What if 100% delivery": format_number(
            scenario_effective_240_sessions["What if 100% delivery"]
        ),
        "Notes / calculation": (
            "240-minute session equivalent measure. Baseline = valid touch "
            "minutes / 240. What-if = actual delivered 240-minute session equivalents "
            "x target utilisation."
        ),
        "Actions": "Use this as the clearest bridge between utilisation % and session capacity.",
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Additional case volume = extra utilised minutes inside the same "
            "elective actual-session base / observed average procedure time. "
            "This is the clearest operational expression of the utilisation uplift."
        ),
        "Actions": "Use this row for the main utilisation opportunity narrative.",
    },
    {
        "Opportunity": "Theatre utilisation",
        "Metric": "Unutilised session equivalent remaining",
        "Current baseline": format_number(baseline_unutilised_240_sessions),
        "What if 50% delivery": format_number(
            scenario_unutilised_240_sessions["What if 50% delivery"]
        ),
        "What if 75% delivery": format_number(
            scenario_unutilised_240_sessions["What if 75% delivery"]
        ),
        "What if 100% delivery": format_number(
            scenario_unutilised_240_sessions["What if 100% delivery"]
        ),
        "Notes / calculation": (
            "Actual delivered 240-minute session equivalents minus effective utilised "
            "sessions. This reduces as theatre utilisation improves."
        ),
        "Actions": "Keep as an efficiency measure, not a count of cancelled lists.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Planned sessions scheduled",
        "Current baseline": format_number(baseline_planned_sessions_context),
        "What if 50% delivery": format_number(baseline_planned_sessions_context),
        "What if 75% delivery": format_number(baseline_planned_sessions_context),
        "What if 100% delivery": format_number(baseline_planned_sessions_context),
        "Notes / calculation": (
            "Context input. Unique elective, non-obstetric scheduled sessions "
            "from Apr 2025-Mar 2026, annualised. The utilisation model itself "
            "uses actual sessions used for utilisation."
        ),
        "Actions": "Validate that cancelled whole sessions are included in the theatre extract.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Actual sessions used for utilisation",
        "Current baseline": format_number(baseline_actual_sessions),
        "What if 50% delivery": format_number(baseline_actual_sessions),
        "What if 75% delivery": format_number(baseline_actual_sessions),
        "What if 100% delivery": format_number(baseline_actual_sessions),
        "Notes / calculation": (
            "Core model input. Elective, non-obstetric sessions with touch time, "
            "completed cases, actual start, or actual finish recorded."
        ),
        "Actions": "Confirm actual-session definition with theatre operations.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Actual % of planned sessions that ran",
        "Current baseline": format_percent(actual_sessions_ran_pct),
        "What if 50% delivery": format_percent(actual_sessions_ran_pct),
        "What if 75% delivery": format_percent(actual_sessions_ran_pct),
        "What if 100% delivery": format_percent(actual_sessions_ran_pct),
        "Notes / calculation": (
            "Actual elective sessions delivered / planned elective sessions scheduled. "
            "This is a session-delivery measure, not utilisation, so it stays "
            "constant unless cancellations or list delivery change."
        ),
        "Actions": "Confirm whether fully cancelled sessions with no booked cases are present in the extract.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Planned sessions, 240-min equivalents",
        "Current baseline": format_number(baseline_planned_240_sessions_context),
        "What if 50% delivery": format_number(baseline_planned_240_sessions_context),
        "What if 75% delivery": format_number(baseline_planned_240_sessions_context),
        "What if 100% delivery": format_number(baseline_planned_240_sessions_context),
        "Notes / calculation": (
            f"Constant input. Scheduled session minutes / {SESSION_STANDARD_MINUTES}. "
            "Shown for context; the utilisation model denominator uses actual "
            "sessions delivered."
        ),
        "Actions": "Use this standardised unit where sessions are compared.",
    },
    {
        "Opportunity": "Capacity baseline",
        "Metric": "Actual sessions, 240-min equivalents",
        "Current baseline": format_number(baseline_actual_240_sessions),
        "What if 50% delivery": format_number(baseline_actual_240_sessions),
        "What if 75% delivery": format_number(baseline_actual_240_sessions),
        "What if 100% delivery": format_number(baseline_actual_240_sessions),
        "Notes / calculation": (
            "Core model input. Scheduled minutes for elective, non-obstetric "
            "sessions that ran / 240."
        ),
        "Actions": "Use as the fixed session base for Scenario 2.",
    },
    {
        "Opportunity": "Scenario 1 - same activity, fewer sessions",
        "Metric": "Sessions required to deliver same cases",
        "Current baseline": format_number(baseline_planned_240_sessions),
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["required_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["required_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["required_240_sessions"]
        ),
        "Notes / calculation": (
            "Scenario 1: same activity, fewer sessions. Required sessions = "
            "baseline touch minutes / target utilisation / 240."
        ),
        "Actions": "Use for efficiency and capacity-release narrative only.",
    },
    {
        "Opportunity": "Scenario 1 - same activity, fewer sessions",
        "Metric": "Sessions freed",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["freed_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["freed_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["freed_240_sessions"]
        ),
        "Notes / calculation": (
            "Scenario 1: baseline actual-session 240-min equivalents minus "
            "sessions required at target utilisation."
        ),
        "Actions": "Only cashable if the freed sessions remove budgeted or premium capacity.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "Total cases over delivery period",
        "Current baseline": format_number(baseline_delivery_cases),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["total_cases"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["total_cases"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["total_cases"]
        ),
        "Notes / calculation": (
            "Scenario 2: same actual sessions, more activity. Target cases = "
            "baseline cases + additional case volume. Baseline uses the "
            "Apr 2025-Mar 2026 weekly case rate over the selected delivery weeks."
        ),
        "Actions": "Confirm whether cases should be all completed cases or RTT-relevant elective cases only.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "Additional cases",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Scenario 2: extra utilised minutes inside the same sessions / "
            "average procedure time."
        ),
        "Actions": "Use for throughput and RTT/PTL impact narrative.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "Cases per list",
        "Current baseline": (
            format_decimal(
                baseline_delivery_cases / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 50% delivery": (
            format_decimal(
                scenario_b_outputs["What if 50% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 75% delivery": (
            format_decimal(
                scenario_b_outputs["What if 75% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 100% delivery": (
            format_decimal(
                scenario_b_outputs["What if 100% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "Notes / calculation": (
            "Total cases / actual delivered 240-minute session equivalents. "
            "This is recalculated as utilisation improves in the same-sessions "
            "throughput scenario."
        ),
        "Actions": "Validate against specialty mix if used operationally.",
    },
    {
        "Opportunity": "Activity assumption",
        "Metric": "Average procedure time",
        "Current baseline": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 50% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 75% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 100% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "Notes / calculation": (
            "Valid case touch minutes / cases with valid touch time. Touch-time "
            "rows over 720 minutes are excluded as outliers."
        ),
        "Actions": "Validate whether specialty-level case duration should be used for final modelling.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "PTL after additional cases",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl"]
        ),
        "Notes / calculation": (
            "Scenario 2: current PTL less additional cases, assuming one case "
            "removes one pathway."
        ),
        "Actions": "Confirm conversion to RTT/PTL reduction before external use.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "PTL reduction",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": "Same as additional cases under the one-case-one-pathway assumption.",
        "Actions": "Validate RTT relevance of unlocked cases.",
    },
    {
        "Opportunity": "Scenario 1 - same activity, fewer sessions",
        "Metric": "Scenario 1 cost opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_a_outputs["What if 50% delivery"]["cost_opportunity"]
        ),
        "What if 75% delivery": format_currency(
            scenario_a_outputs["What if 75% delivery"]["cost_opportunity"]
        ),
        "What if 100% delivery": format_currency(
            scenario_a_outputs["What if 100% delivery"]["cost_opportunity"]
        ),
        "Notes / calculation": "Sessions freed x 240 minutes x theatre/anaesthetic cost per scheduled minute.",
        "Actions": "Finance validation required before treating as cashable.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "Scenario 2 additional activity value",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cost_avoidance"]
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cost_avoidance"]
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cost_avoidance"]
        ),
        "Notes / calculation": "Additional utilised minutes x theatre/anaesthetic cost per scheduled minute.",
        "Actions": "Treat as avoided-cost/capacity value unless it displaces budgeted spend.",
    },
    {
        "Opportunity": "Scenario 2 - same sessions, more activity",
        "Metric": "Scenario 2 income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "Notes / calculation": (
            f"Additional case volume x agreed average income per theatre case "
            f"({format_currency(THEATRE_CASE_VALUE_DEFAULT)})."
        ),
        "Actions": "Gross income lens only; Finance should confirm tariff/payment treatment and whether the activity is genuinely incremental.",
    },
    {
        "Opportunity": "Estate and workforce capacity",
        "Metric": "Estate time utilised",
        "Current baseline": format_percent(current_estate_time_utilisation),
        "What if 50% delivery": format_percent(
            scenario_estate_time_utilisation["What if 50% delivery"]
        ),
        "What if 75% delivery": format_percent(
            scenario_estate_time_utilisation["What if 75% delivery"]
        ),
        "What if 100% delivery": format_percent(
            scenario_estate_time_utilisation["What if 100% delivery"]
        ),
        "Notes / calculation": (
            "This changes with utilisation. It is case touch time after the "
            "scenario divided by available estate minutes."
        ),
        "Actions": "Use this row, not fixed estate capacity, to show how improved utilisation uses more of the estate.",
    },
    {
        "Opportunity": "Validation checks",
        "Metric": "Actual <= planned validation",
        "Current baseline": validation_label(
            baseline_actual_240_sessions <= baseline_planned_240_sessions_context
        ),
        "What if 50% delivery": validation_label(
            baseline_actual_240_sessions <= baseline_planned_240_sessions_context
        ),
        "What if 75% delivery": validation_label(
            baseline_actual_240_sessions <= baseline_planned_240_sessions_context
        ),
        "What if 100% delivery": validation_label(
            baseline_actual_240_sessions <= baseline_planned_240_sessions_context
        ),
        "Notes / calculation": "Actual session equivalents must not exceed planned session equivalents.",
        "Actions": "Investigate if this fails after source data refresh.",
    },
]

theatre_scenario_rows = [
    {
        "Scenario": "Scenario 1 - do the same with fewer sessions",
        "Metric": "Sessions required, 240-min equivalents",
        "Current baseline": format_number(baseline_planned_240_sessions),
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["required_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["required_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["required_240_sessions"]
        ),
        "Notes / calculation": (
            "Sessions are deliberately recalculated. Cases stay fixed; required "
            "sessions = baseline touch minutes / target utilisation / 240."
        ),
        "Actions": "Use for efficiency/capacity-release narrative only.",
    },
    {
        "Scenario": "Scenario 1 - do the same with fewer sessions",
        "Metric": "Sessions freed, 240-min equivalents",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_a_outputs["What if 50% delivery"]["freed_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            scenario_a_outputs["What if 75% delivery"]["freed_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            scenario_a_outputs["What if 100% delivery"]["freed_240_sessions"]
        ),
        "Notes / calculation": "Baseline planned session equivalents - sessions required.",
        "Actions": "Only cashable if the freed sessions remove budgeted cost or external/premium capacity.",
    },
    {
        "Scenario": "Scenario 1 - do the same with fewer sessions",
        "Metric": "Total cases per year",
        "Current baseline": format_number(baseline_delivery_cases),
        "What if 50% delivery": format_number(baseline_delivery_cases),
        "What if 75% delivery": format_number(baseline_delivery_cases),
        "What if 100% delivery": format_number(baseline_delivery_cases),
        "Notes / calculation": (
            "Cases are fixed by design in Scenario 1. This is the Apr 2025-Mar "
            f"2026 baseline annualised over {format_decimal(active_delivery_weeks, 1)} weeks."
        ),
        "Actions": "Do not use Scenario 1 to claim PTL reduction.",
    },
    {
        "Scenario": "Scenario 1 - do the same with fewer sessions",
        "Metric": "Cases per list",
        "Current baseline": (
            format_decimal(
                baseline_delivery_cases / baseline_planned_240_sessions,
                2,
            )
            if baseline_planned_240_sessions > 0
            else "0.00"
        ),
        "What if 50% delivery": (
            format_decimal(
                baseline_delivery_cases
                / scenario_a_outputs["What if 50% delivery"]["required_240_sessions"],
                2,
            )
            if scenario_a_outputs["What if 50% delivery"]["required_240_sessions"] > 0
            else "0.00"
        ),
        "What if 75% delivery": (
            format_decimal(
                baseline_delivery_cases
                / scenario_a_outputs["What if 75% delivery"]["required_240_sessions"],
                2,
            )
            if scenario_a_outputs["What if 75% delivery"]["required_240_sessions"] > 0
            else "0.00"
        ),
        "What if 100% delivery": (
            format_decimal(
                baseline_delivery_cases
                / scenario_a_outputs["What if 100% delivery"]["required_240_sessions"],
                2,
            )
            if scenario_a_outputs["What if 100% delivery"]["required_240_sessions"] > 0
            else "0.00"
        ),
        "Notes / calculation": (
            "Fixed cases / sessions required. This rises because the same cases "
            "are delivered in fewer 240-minute equivalent sessions."
        ),
        "Actions": "Use only within Scenario 1 efficiency narrative.",
    },
    {
        "Scenario": "Scenario 1 - do the same with fewer sessions",
        "Metric": "Indicative financial opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_a_outputs["What if 50% delivery"]["cost_opportunity"]
        ),
        "What if 75% delivery": format_currency(
            scenario_a_outputs["What if 75% delivery"]["cost_opportunity"]
        ),
        "What if 100% delivery": format_currency(
            scenario_a_outputs["What if 100% delivery"]["cost_opportunity"]
        ),
        "Notes / calculation": "Sessions freed x 240 minutes x theatre/anaesthetic cost per scheduled minute.",
        "Actions": "Finance validation required before treating as cashable.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Sessions used, 240-min equivalents",
        "Current baseline": format_number(baseline_actual_240_sessions),
        "What if 50% delivery": format_number(baseline_actual_240_sessions),
        "What if 75% delivery": format_number(baseline_actual_240_sessions),
        "What if 100% delivery": format_number(baseline_actual_240_sessions),
        "Notes / calculation": "Sessions are deliberately held constant in Scenario 2.",
        "Actions": "Guardrail: session numbers must not change in this scenario.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Additional cases = extra utilised minutes inside the same delivered "
            "sessions / average procedure time."
        ),
        "Actions": "Use for throughput and RTT/PTL impact narrative.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Total cases per year",
        "Current baseline": format_number(baseline_delivery_cases),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["total_cases"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["total_cases"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["total_cases"]
        ),
        "Notes / calculation": "Baseline completed cases + additional cases.",
        "Actions": "Validate all unlocked cases are clinically deliverable and RTT-relevant.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Cases per list",
        "Current baseline": (
            format_decimal(
                baseline_delivery_cases / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 50% delivery": (
            format_decimal(
                scenario_b_outputs["What if 50% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 75% delivery": (
            format_decimal(
                scenario_b_outputs["What if 75% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "What if 100% delivery": (
            format_decimal(
                scenario_b_outputs["What if 100% delivery"]["total_cases"]
                / baseline_actual_240_sessions,
                2,
            )
            if baseline_actual_240_sessions > 0
            else "0.00"
        ),
        "Notes / calculation": (
            "Total cases / actual delivered 240-minute session equivalents. "
            "This rises because sessions are fixed and activity increases."
        ),
        "Actions": "Validate against specialty mix if used operationally.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "PTL after additional cases",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl"]
        ),
        "Notes / calculation": "Current PTL - additional cases; assumes one case removes one pathway.",
        "Actions": "Confirm conversion to RTT/PTL reduction before external use.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "PTL variance / reduction",
        "Current baseline": "0",
        "What if 50% delivery": f"-{format_number(scenario_b_outputs['What if 50% delivery']['cases_unlocked'])}",
        "What if 75% delivery": f"-{format_number(scenario_b_outputs['What if 75% delivery']['cases_unlocked'])}",
        "What if 100% delivery": f"-{format_number(scenario_b_outputs['What if 100% delivery']['cases_unlocked'])}",
        "Notes / calculation": (
            "Negative number shows the reduction from current PTL. Assumes one "
            "additional case removes one pathway."
        ),
        "Actions": "Validate RTT relevance of unlocked cases.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "PTL reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": format_percent(
            scenario_b_outputs["What if 50% delivery"]["ptl_reduction_pct"]
        ),
        "What if 75% delivery": format_percent(
            scenario_b_outputs["What if 75% delivery"]["ptl_reduction_pct"]
        ),
        "What if 100% delivery": format_percent(
            scenario_b_outputs["What if 100% delivery"]["ptl_reduction_pct"]
        ),
        "Notes / calculation": "Additional case volume / current PTL.",
        "Actions": "Use as an indicative reduction only until RTT conversion is validated.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Remaining PTL %",
        "Current baseline": "100.0%",
        "What if 50% delivery": format_percent(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl_pct"]
        ),
        "What if 75% delivery": format_percent(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl_pct"]
        ),
        "What if 100% delivery": format_percent(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl_pct"]
        ),
        "Notes / calculation": "PTL after additional cases / current PTL.",
        "Actions": "Read alongside PTL reduction %, not as a formal RTT forecast.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Indicative financial opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cost_avoidance"]
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cost_avoidance"]
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cost_avoidance"]
        ),
        "Notes / calculation": "Additional utilised minutes x theatre/anaesthetic cost per scheduled minute.",
        "Actions": "Treat as avoided-cost/capacity value unless it displaces budgeted spend.",
    },
    {
        "Scenario": "Scenario 2 - do more with the same sessions",
        "Metric": "Indicative income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "Notes / calculation": (
            f"Additional case volume x agreed average income per theatre case "
            f"({format_currency(THEATRE_CASE_VALUE_DEFAULT)})."
        ),
        "Actions": "Gross income lens only; Finance should confirm tariff/payment treatment and whether the activity is genuinely incremental.",
    },
]

for row in theatre_scenario_rows:
    if row["Scenario"].startswith("Scenario 1"):
        row["Scenario interpretation"] = (
            "Same cases, fewer sessions: cases are fixed and sessions are "
            "recalculated to show efficiency and capacity release."
        )
    else:
        row["Scenario interpretation"] = (
            "Same sessions, more activity: sessions are fixed and extra "
            "utilised time is converted into cases and PTL reduction."
        )

baseline_input_rows = [
    {
        "Input area": "Baseline sessions",
        "Metric": "Actual sessions used for utilisation",
        "Value": format_number(full_year_elective_actual_sessions_used),
        "Definition / calculation": (
            "Apr 2025-Mar 2026 elective sessions with recorded touch time, "
            "completed cases, actual start, or actual finish, after invalid/"
            "24-hour scheduled sessions are removed."
        ),
        "Why fixed": (
            "This is the full-year elective actual-session count. It is not "
            "scaled to 43 weeks."
        ),
        "Actions": "Confirm actual-session definition with theatre operations.",
    },
    {
        "Input area": "Standardised sessions",
        "Metric": "Full-year elective 240-min session equivalents",
        "Value": format_number(full_year_elective_240_session_equivalents),
        "Definition / calculation": (
            "Apr 2025-Mar 2026 elective actual-session scheduled minutes / 240."
        ),
        "Why fixed": (
            "This is the full-year elective standardised session baseline. It "
            "is not scaled to 43 weeks."
        ),
        "Actions": "Use this as the session-equivalent denominator in the pack.",
    },
    {
        "Input area": "Activity baseline",
        "Metric": "Total cases per year",
        "Value": format_number(full_year_elective_completed_cases),
        "Definition / calculation": (
            "Completed elective theatre cases from Apr 2025-Mar 2026."
        ),
        "Why fixed": (
            "This is the full-year elective case baseline. It is not scaled to "
            "43 weeks."
        ),
        "Actions": "Confirm whether all elective cases should be treated as RTT-relevant before using for backlog commitments.",
    },
    {
        "Input area": "Activity assumption",
        "Metric": "Average procedure time",
        "Value": f"{format_decimal(avg_procedure_time, 1)} mins",
        "Definition / calculation": (
            "Model Hospital touch minutes / cases with valid touch time. Touch "
            "starts at anaesthetic start where available and ends at recovery "
            "where available; rows over 720 minutes are excluded as outliers."
        ),
        "Why fixed": (
            "This is an assumption used to convert additional utilised minutes "
            "into additional cases."
        ),
        "Actions": "Validate whether specialty-level case duration should be used for final modelling.",
    },
]

theatre_utilisation_rows = []
for row in theatre_baseline_rows:
    if row["Opportunity"] == "Theatre utilisation" or row["Metric"] == "Estate time utilised":
        display_row = {**row}
        display_row["Opportunity"] = "Theatre utilisation"
        theatre_utilisation_rows.append(display_row)

theatre_utilisation_impact_rows = [
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "PTL after additional cases",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl"]
        ),
        "Notes / calculation": (
            "Current PTL less additional case volume from higher theatre utilisation. "
            "Assumes one additional case removes one pathway."
        ),
        "Actions": "Validate RTT relevance before using as a backlog commitment.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Additional utilised minutes from the target utilisation level / "
            "average procedure time."
        ),
        "Actions": "Use as the operational bridge between utilisation and backlog impact.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "PTL variance / reduction",
        "Current baseline": "0",
        "What if 50% delivery": f"-{format_number(scenario_b_outputs['What if 50% delivery']['cases_unlocked'])}",
        "What if 75% delivery": f"-{format_number(scenario_b_outputs['What if 75% delivery']['cases_unlocked'])}",
        "What if 100% delivery": f"-{format_number(scenario_b_outputs['What if 100% delivery']['cases_unlocked'])}",
        "Notes / calculation": (
            "Negative number shows the reduction from current PTL. It equals "
            "additional case volume under the one-case-one-pathway assumption."
        ),
        "Actions": "Confirm whether all unlocked cases remove RTT pathways.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "PTL reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": format_percent(
            scenario_b_outputs["What if 50% delivery"]["ptl_reduction_pct"]
        ),
        "What if 75% delivery": format_percent(
            scenario_b_outputs["What if 75% delivery"]["ptl_reduction_pct"]
        ),
        "What if 100% delivery": format_percent(
            scenario_b_outputs["What if 100% delivery"]["ptl_reduction_pct"]
        ),
        "Notes / calculation": "Additional case volume / current PTL.",
        "Actions": "Treat as indicative until RTT conversion is validated.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "Remaining PTL %",
        "Current baseline": "100.0%",
        "What if 50% delivery": format_percent(
            scenario_b_outputs["What if 50% delivery"]["remaining_ptl_pct"]
        ),
        "What if 75% delivery": format_percent(
            scenario_b_outputs["What if 75% delivery"]["remaining_ptl_pct"]
        ),
        "What if 100% delivery": format_percent(
            scenario_b_outputs["What if 100% delivery"]["remaining_ptl_pct"]
        ),
        "Notes / calculation": "PTL after additional cases / current PTL.",
        "Actions": "Read alongside PTL after additional cases.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "Indicative financial opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cost_avoidance"]
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cost_avoidance"]
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cost_avoidance"]
        ),
        "Notes / calculation": (
            "Additional utilised minutes x theatre/anaesthetic cost per "
            "scheduled minute."
        ),
        "Actions": "Treat as an opportunity proxy unless Finance confirms cashability.",
    },
    {
        "Opportunity": "Theatre utilisation impact",
        "Metric": "Indicative income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            scenario_b_outputs["What if 50% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 75% delivery": format_currency(
            scenario_b_outputs["What if 75% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 100% delivery": format_currency(
            scenario_b_outputs["What if 100% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "Notes / calculation": (
            f"Additional case volume x agreed average income per theatre case "
            f"({format_currency(THEATRE_CASE_VALUE_DEFAULT)})."
        ),
        "Actions": "Gross income lens only; Finance should confirm tariff/payment treatment and whether the activity is genuinely incremental.",
    },
]
theatre_utilisation_rows.extend(theatre_utilisation_impact_rows)

theatre_target_rows = [
    {
        "What-if lens / group": "Baseline and utilisation target",
        "Metric": "Elective utilisation - actual sessions",
        "Current baseline": format_percent(full_year_elective_utilisation),
        "What if 50% delivery": "78.5%",
        "What if 75% delivery": "81.8%",
        "What if 100% delivery": "85.0%",
        "Notes / calculation": (
            "Apr 2025-Mar 2026 elective actual-session touch time / elective "
            "actual-session scheduled minutes."
        ),
        "Actions": "Use as the utilisation target row.",
    },
    {
        "What-if lens / group": "Baseline and utilisation target",
        "Metric": "Full-year elective 240-min session equivalents",
        "Current baseline": format_number(full_year_elective_240_session_equivalents),
        "What if 50% delivery": format_number(full_year_elective_240_session_equivalents),
        "What if 75% delivery": format_number(full_year_elective_240_session_equivalents),
        "What if 100% delivery": format_number(full_year_elective_240_session_equivalents),
        "Notes / calculation": (
            f"Apr 2025-Mar 2026 elective actual-session scheduled minutes / {SESSION_STANDARD_MINUTES}. "
            "This is the full-year baseline, not the 43-week scaled value."
        ),
        "Actions": "Use this as the session-equivalent denominator in the pack.",
    },
    {
        "What-if lens / group": "Baseline and utilisation target",
        "Metric": "Average procedure time",
        "Current baseline": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 50% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 75% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "What if 100% delivery": f"{format_decimal(avg_procedure_time, 1)} mins",
        "Notes / calculation": "Touch minutes / completed elective cases with valid touch time.",
        "Actions": "Held constant unless specialty-level modelling is introduced.",
    },
    {
        "What-if lens / group": "Estate capacity impact - actual sessions",
        "Metric": "Estate time utilised from actual sessions",
        "Current baseline": format_percent(current_estate_time_utilisation),
        "What if 50% delivery": format_percent(
            scenario_estate_time_utilisation["What if 50% delivery"]
        ),
        "What if 75% delivery": format_percent(
            scenario_estate_time_utilisation["What if 75% delivery"]
        ),
        "What if 100% delivery": format_percent(
            scenario_estate_time_utilisation["What if 100% delivery"]
        ),
        "Notes / calculation": (
            "Current = elective actual-session touch minutes / available estate "
            "minutes. What-if = elective actual-session scheduled minutes x "
            "target utilisation / available estate minutes."
        ),
        "Actions": (
            "Estate denominator uses confirmed/assumed theatre estate settings; "
            "replace with PAH-confirmed theatre capacity when supplied."
        ),
    },
    {
        "What-if lens / group": "What if more throughput - same sessions, more cases",
        "Metric": "Additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            full_year_more_throughput_outputs["What if 50% delivery"]["cases_unlocked"]
        ),
        "What if 75% delivery": format_number(
            full_year_more_throughput_outputs["What if 75% delivery"]["cases_unlocked"]
        ),
        "What if 100% delivery": format_number(
            full_year_more_throughput_outputs["What if 100% delivery"]["cases_unlocked"]
        ),
        "Notes / calculation": (
            "Extra utilised minutes inside the same actual-session base / "
            "average procedure time."
        ),
        "Actions": "Main throughput impact row.",
    },
    {
        "What-if lens / group": "What if more throughput - same sessions, more cases",
        "Metric": "Total case volume",
        "Current baseline": format_number(full_year_elective_completed_cases),
        "What if 50% delivery": format_number(
            full_year_more_throughput_outputs["What if 50% delivery"]["total_cases"]
        ),
        "What if 75% delivery": format_number(
            full_year_more_throughput_outputs["What if 75% delivery"]["total_cases"]
        ),
        "What if 100% delivery": format_number(
            full_year_more_throughput_outputs["What if 100% delivery"]["total_cases"]
        ),
        "Notes / calculation": "Baseline elective case volume + additional case volume.",
        "Actions": "Use with additional case volume to show total activity.",
    },
    {
        "What-if lens / group": "RTT backlog impact - from same sessions, more cases",
        "Metric": "Opening RTT backlog",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(current_ptl),
        "What if 75% delivery": format_number(current_ptl),
        "What if 100% delivery": format_number(current_ptl),
        "Notes / calculation": (
            f"Latest RTT/PTL backlog from {latest_ptl_month.strftime('%b %Y')}."
        ),
        "Actions": "Use as the starting backlog before theatre-utilisation impact.",
    },
    {
        "What-if lens / group": "RTT backlog impact - from same sessions, more cases",
        "Metric": "RTT backlog reduction from additional case volume",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            full_year_more_throughput_outputs["What if 50% delivery"]["rtt_backlog_reduction"]
        ),
        "What if 75% delivery": format_number(
            full_year_more_throughput_outputs["What if 75% delivery"]["rtt_backlog_reduction"]
        ),
        "What if 100% delivery": format_number(
            full_year_more_throughput_outputs["What if 100% delivery"]["rtt_backlog_reduction"]
        ),
        "Notes / calculation": (
            "Additional case volume capped at the opening RTT backlog. Assumes "
            "one additional elective theatre case removes one RTT pathway."
        ),
        "Actions": "Validate RTT relevance and specialty mix before using as a commitment.",
    },
    {
        "What-if lens / group": "RTT backlog impact - from same sessions, more cases",
        "Metric": "Closing RTT backlog after theatre utilisation",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": format_number(
            full_year_more_throughput_outputs["What if 50% delivery"]["remaining_ptl"]
        ),
        "What if 75% delivery": format_number(
            full_year_more_throughput_outputs["What if 75% delivery"]["remaining_ptl"]
        ),
        "What if 100% delivery": format_number(
            full_year_more_throughput_outputs["What if 100% delivery"]["remaining_ptl"]
        ),
        "Notes / calculation": "Opening RTT backlog - RTT backlog reduction from additional case volume.",
        "Actions": "This is the modelled closing backlog position.",
    },
    {
        "What-if lens / group": "RTT backlog impact - from same sessions, more cases",
        "Metric": "RTT backlog reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": format_percent(
            full_year_more_throughput_outputs["What if 50% delivery"]["ptl_reduction_pct"]
        ),
        "What if 75% delivery": format_percent(
            full_year_more_throughput_outputs["What if 75% delivery"]["ptl_reduction_pct"]
        ),
        "What if 100% delivery": format_percent(
            full_year_more_throughput_outputs["What if 100% delivery"]["ptl_reduction_pct"]
        ),
        "Notes / calculation": "RTT backlog reduction / opening RTT backlog.",
        "Actions": "Use alongside the closing RTT backlog row.",
    },
    {
        "What-if lens / group": "What if same throughput - fewer sessions, same cases",
        "Metric": "Case volume held constant",
        "Current baseline": format_number(full_year_elective_completed_cases),
        "What if 50% delivery": format_number(full_year_elective_completed_cases),
        "What if 75% delivery": format_number(full_year_elective_completed_cases),
        "What if 100% delivery": format_number(full_year_elective_completed_cases),
        "Notes / calculation": (
            "This scenario keeps Apr 2025-Mar 2026 elective case volume fixed "
            "and asks how many fewer 240-minute sessions would be needed."
        ),
        "Actions": "Do not use this row to claim RTT backlog reduction.",
    },
    {
        "What-if lens / group": "What if same throughput - fewer sessions, same cases",
        "Metric": "Full-year 240-min sessions required",
        "Current baseline": format_number(full_year_elective_240_session_equivalents),
        "What if 50% delivery": format_number(
            full_year_same_throughput_outputs["What if 50% delivery"]["required_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            full_year_same_throughput_outputs["What if 75% delivery"]["required_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            full_year_same_throughput_outputs["What if 100% delivery"]["required_240_sessions"]
        ),
        "Notes / calculation": (
            "Full-year elective touch minutes / target utilisation / 240."
        ),
        "Actions": "This is the capacity-release scenario output.",
    },
    {
        "What-if lens / group": "What if same throughput - fewer sessions, same cases",
        "Metric": "Full-year 240-min sessions freed",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            full_year_same_throughput_outputs["What if 50% delivery"]["freed_240_sessions"]
        ),
        "What if 75% delivery": format_number(
            full_year_same_throughput_outputs["What if 75% delivery"]["freed_240_sessions"]
        ),
        "What if 100% delivery": format_number(
            full_year_same_throughput_outputs["What if 100% delivery"]["freed_240_sessions"]
        ),
        "Notes / calculation": (
            "Full-year elective 240-minute equivalents - 240-minute sessions required."
        ),
        "Actions": "Use as the simple same-throughput/fewer-sessions impact row.",
    },
    {
        "What-if lens / group": "What if same throughput - fewer sessions, same cases",
        "Metric": "Indicative cost opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            full_year_same_throughput_outputs["What if 50% delivery"]["cost_opportunity"]
        ),
        "What if 75% delivery": format_currency(
            full_year_same_throughput_outputs["What if 75% delivery"]["cost_opportunity"]
        ),
        "What if 100% delivery": format_currency(
            full_year_same_throughput_outputs["What if 100% delivery"]["cost_opportunity"]
        ),
        "Notes / calculation": (
            "Full-year 240-minute sessions freed x 240 minutes x theatre/"
            "anaesthetic cost per scheduled minute."
        ),
        "Actions": "Treat as capacity value unless Finance confirms cashability.",
    },
    {
        "What-if lens / group": "What if more throughput - same sessions, more cases",
        "Metric": "Indicative income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            full_year_more_throughput_outputs["What if 50% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 75% delivery": format_currency(
            full_year_more_throughput_outputs["What if 75% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "What if 100% delivery": format_currency(
            full_year_more_throughput_outputs["What if 100% delivery"]["cases_unlocked"]
            * THEATRE_CASE_VALUE_DEFAULT
        ),
        "Notes / calculation": (
            f"Additional case volume x agreed average income per theatre case "
            f"({format_currency(THEATRE_CASE_VALUE_DEFAULT)})."
        ),
        "Actions": "Gross income lens only; Finance should confirm tariff/payment treatment and whether the activity is genuinely incremental.",
    },
]
table_rows = theatre_target_rows
theatre_baseline_column = "Baseline (Apr 2025-Mar 2026)"
table_df = pd.DataFrame(table_rows).rename(
    columns={
        **DISPLAY_SCENARIO_LABELS,
        "Current baseline": theatre_baseline_column,
    }
)
baseline_input_df = pd.DataFrame(baseline_input_rows)
capacity_assumption_df = pd.DataFrame(capacity_assumption_rows)
theatre_scenario_columns = [
    "Scenario",
    "Scenario interpretation",
    "Metric",
    "Current baseline",
    "What if 50% delivery",
    "What if 75% delivery",
    "What if 100% delivery",
    "Notes / calculation",
    "Actions",
]
theatre_scenario_df = pd.DataFrame(theatre_scenario_rows)[
    theatre_scenario_columns
].rename(
    columns=DISPLAY_SCENARIO_LABELS
)
theatre_metric_coverage_df = pd.DataFrame(
    [
        {
            "Requested metric": "Elective vs emergency split",
            "Shown in": "Total Activity Context",
            "Status": "Included",
        },
        {
            "Requested metric": "Current utilisation",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Effective utilised sessions",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Actual sessions run vs available estate theatre capacity",
            "Shown in": "Estate and Workforce Capacity Assumptions",
            "Status": "Included",
        },
        {
            "Requested metric": "Estate time utilised from actual sessions",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Total substantive operating sessions from job plans",
            "Shown in": "Estate and Workforce Capacity Assumptions",
            "Status": "Included",
        },
        {
            "Requested metric": "Additional case volume",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Unutilised session equivalent remaining",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Total cases per year",
            "Shown in": "Theatre Baseline Inputs and Theatre Scenario Output Table",
            "Status": "Included",
        },
        {
            "Requested metric": "Cases per list",
            "Shown in": "Theatre Scenario Output Table",
            "Status": "Included",
        },
        {
            "Requested metric": "Average procedure time",
            "Shown in": "Theatre Baseline Inputs",
            "Status": "Included",
        },
        {
            "Requested metric": "Opening RTT backlog",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "RTT backlog reduction from additional case volume",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Closing RTT backlog after theatre utilisation",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "RTT backlog reduction %",
            "Shown in": "Theatre Utilisation Target View",
            "Status": "Included",
        },
        {
            "Requested metric": "Remaining PTL %",
            "Shown in": "Theatre Utilisation Target View and Theatre Scenario Output Table",
            "Status": "Included",
        },
        {
            "Requested metric": "Indicative financial opportunity",
            "Shown in": "Theatre Utilisation Target View and Theatre Scenario Output Table",
            "Status": "Included",
        },
    ]
)
theatre_action_evidence_df = pd.DataFrame(
    [
        {
            "Action / question": "Does the refreshed PAH extract support elective vs emergency and full timestamp logic?",
            "Answer from available data": "Yes.",
            "Evidence": (
                "The extract includes `Elective/Emergency`, anaesthetic start, "
                "operation/procedure timestamps and patient-in-recovery timestamp. "
                "The model uses anaesthetic start to recovery where available."
            ),
            "Still needs confirmation": (
                "Confirm with PAH that these are the preferred fields for the final Model Hospital-aligned calculation."
            ),
        },
        {
            "Action / question": "Does the theatre extract include cancellation data?",
            "Answer from available data": "Yes, for cancelled cases recorded in the extract.",
            "Evidence": (
                f"{format_number(cancelled_case_rows)} rows have cancelled cases; "
                f"{format_number(cancellation_date_rows)} rows have a cancellation date; "
                f"{format_number(cancellation_reason_rows)} rows have a populated cancellation reason."
            ),
            "Still needs confirmation": (
                "Whether the extract includes theatre sessions with no booked case rows at all."
            ),
        },
        {
            "Action / question": "Can actual sessions used for utilisation be defined from the data?",
            "Answer from available data": "Yes, using an inferred delivered-session flag.",
            "Evidence": (
                "Full-year elective actual sessions used for utilisation = "
                f"{format_number(full_year_elective_actual_sessions_used)} "
                "for Apr 2025-Mar 2026, using touch "
                "time, completed cases, actual start, or actual finish."
            ),
            "Still needs confirmation": (
                "Theatre operations should confirm this is the agreed local definition of a session that ran."
            ),
        },
        {
            "Action / question": "Can actual sessions vs estate capacity be calculated?",
            "Answer from available data": "Partially.",
            "Evidence": (
                f"The model can calculate {format_percent(actual_sessions_vs_estate_capacity)} "
                "using elective actual 240-minute sessions divided by the estate denominator."
            ),
            "Still needs confirmation": (
                f"Estate denominator currently uses {estate_theatres} theatres x "
                f"{format_decimal(estate_sessions_per_day, 1)} sessions/day x "
                f"{format_decimal(estate_days_per_week, 1)} days/week x "
                f"{format_decimal(estate_capacity_weeks, 1)} weeks. PAH should confirm "
                "physical theatre count and routine operating pattern."
            ),
        },
        {
            "Action / question": "Can substantive workforce DCC operating sessions be calculated?",
            "Answer from available data": "Partially.",
            "Evidence": (
                f"Job-planning file {job_plan_capacity['source']} has "
                f"{format_number(job_plan_capacity['rows'])} substantive rows and "
                f"{format_number(job_plan_capacity['theatre_weekly'])} operating "
                f"sessions per week, giving {format_number(workforce_theatre_capacity_240_sessions)} "
                f"over {format_decimal(active_delivery_weeks, 1)} weeks."
            ),
            "Still needs confirmation": (
                "Confirm `Operating sessions` maps to weekly theatre DCC sessions "
                "and agree whether anaesthetic rows should be included."
            ),
        },
        {
            "Action / question": "Can current theatre utilisation be calculated?",
            "Answer from available data": "Yes.",
            "Evidence": (
                f"Current elective utilisation = {format_percent(current_utilisation)} = "
                "anaesthetic-to-recovery touch minutes / scheduled minutes for "
                "elective, non-obstetric actual sessions in Apr 2025-Mar 2026."
            ),
            "Still needs confirmation": (
                "PAH should confirm whether any additional local Model Hospital exclusions apply."
            ),
        },
        {
            "Action / question": "Can total cases and average procedure time be calculated?",
            "Answer from available data": "Yes.",
            "Evidence": (
                f"Baseline cases = {format_number(baseline_delivery_cases)}; "
                f"average procedure time = {format_decimal(avg_procedure_time, 1)} minutes."
            ),
            "Still needs confirmation": (
                "Confirm whether all elective cases should be treated as RTT-relevant."
            ),
        },
        {
            "Action / question": "Can RTT/PTL impact be proven from the theatre extract alone?",
            "Answer from available data": "No, only modelled indicatively.",
            "Evidence": (
                "No RTT/pathway/waiting-list columns were found in the theatre extract."
                if not rtt_columns
                else f"Potential RTT columns found: {', '.join(rtt_columns)}."
            ),
            "Still needs confirmation": (
                "Validate the one additional case = one pathway reduction assumption against RTT/PTL data."
            ),
        },
        {
            "Action / question": "Can financial opportunity be calculated from available data?",
            "Answer from available data": "Yes as an opportunity proxy, not as confirmed cashable benefit.",
            "Evidence": (
                f"The model uses {format_currency(theatre_cost_2526)} of theatre/anaesthetic "
                "cost from the trial balance to derive a scheduled-minute value."
            ),
            "Still needs confirmation": (
                "Finance must confirm whether any value is cashable, cost avoidance, or only capacity value."
            ),
        },
    ]
)

if not theatre_activity_split_df.empty:
    theatre_activity_split_display_df = theatre_activity_split_df.copy()
    theatre_activity_split_display_df = theatre_activity_split_display_df.drop(
        columns=["Planned sessions", "Actual sessions delivered"],
        errors="ignore",
    ).rename(
        columns={
            "Valid actual sessions used for utilisation": "Actual sessions used for utilisation"
        }
    )
    for col in [
        "Actual sessions used for utilisation",
        "Actual 240-min session equivalents",
        "Completed cases",
        "Touch minutes used",
        "Scheduled minutes used",
        "Obstetric sessions",
        "Invalid / 24hr sessions",
    ]:
        theatre_activity_split_display_df[col] = theatre_activity_split_display_df[
            col
        ].map(format_number)
    theatre_activity_split_display_df["Utilisation"] = theatre_activity_split_df[
        "Utilisation"
    ].map(format_percent)
else:
    theatre_activity_split_display_df = pd.DataFrame()

st.subheader("Theatre Baseline Inputs")
with st.expander("Core definitions and guardrails", expanded=True):
    st.markdown(
        f"""
- Total activity is retained for context, but the utilisation baseline and what-if model are elective-only.
- Elective utilisation excludes emergency/mixed sessions, obstetrics sessions, cancelled/not-run sessions, and invalid/24-hour scheduled sessions.
- Touch time uses anaesthetic start to patient into recovery where available; otherwise it falls back to the next available theatre timestamps and then the supplied case touch-time field.
- Actual sessions used for utilisation: elective, non-obstetric sessions with recorded touch time, completed cases, actual start, or actual finish, after invalid/24-hour scheduled sessions are removed.
- Theatre utilisation: elective actual-session touch time divided by elective actual-session scheduled minutes.
- Session standard: one session equivalent = {SESSION_STANDARD_MINUTES} minutes.
- Baseline: April 2025 to March 2026 full-year theatre utilisation.
- Estate capacity default: {DEFAULT_ESTATE_THEATRES} elective theatres, within a wider estate of {TOTAL_ESTATE_THEATRES} theatres, based on the latest PAH information provided. The raw extract has {raw_theatre_count} distinct theatre/location labels, which is not the same as confirmed available elective estate.
- Substantive DCC capacity: the workforce capacity layer uses `Operating sessions` from {job_plan_capacity['source']} in the job-planning splits folder, excluding Locum rows.
- The utilisation target table shows how theatre-utilisation metrics and their indicative PTL/financial impact move at 78.5%, 81.75%, and 85%.
        """
    )

with st.expander("Theatre activity context: whole set, elective and non-elective", expanded=True):
    st.caption(
        "The first rows show the whole theatre set, elective sessions and "
        "non-elective sessions. Non-elective includes emergency, mixed "
        "elective/emergency and unknown session types. The model baseline remains "
        "elective-only and excludes obstetrics."
    )
    if theatre_activity_split_display_df.empty:
        st.warning("No session-type split could be calculated from the theatre extract.")
    else:
        st.dataframe(
            theatre_activity_split_display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Session type": st.column_config.TextColumn(
                    "Session type",
                    width="medium",
                ),
                "Actual sessions used for utilisation": st.column_config.TextColumn(
                    "Actual sessions used for utilisation",
                    width="medium",
                ),
                "Actual 240-min session equivalents": st.column_config.TextColumn(
                    "Actual 240-min session equivalents",
                    width="medium",
                ),
                "Completed cases": st.column_config.TextColumn(
                    "Completed cases",
                    width="small",
                ),
                "Touch minutes used": st.column_config.TextColumn(
                    "Touch minutes used",
                    width="small",
                ),
                "Scheduled minutes used": st.column_config.TextColumn(
                    "Scheduled minutes used",
                    width="small",
                ),
                "Utilisation": st.column_config.TextColumn(
                    "Utilisation",
                    width="small",
                ),
                "Obstetric sessions": st.column_config.TextColumn(
                    "Obstetric sessions",
                    width="small",
                ),
                "Invalid / 24hr sessions": st.column_config.TextColumn(
                    "Invalid / 24hr sessions",
                    width="small",
                ),
            },
        )

with st.expander("Where the requested theatre metrics appear", expanded=False):
    st.dataframe(
        theatre_metric_coverage_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Requested metric": st.column_config.TextColumn(
                "Requested metric",
                width="large",
            ),
            "Shown in": st.column_config.TextColumn("Shown in", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

with st.expander("Action answers from available data", expanded=True):
    st.dataframe(
        theatre_action_evidence_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Action / question": st.column_config.TextColumn(
                "Action / question",
                width="large",
            ),
            "Answer from available data": st.column_config.TextColumn(
                "Answer from available data",
                width="medium",
            ),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            "Still needs confirmation": st.column_config.TextColumn(
                "Still needs confirmation",
                width="large",
            ),
        },
    )

with st.expander("Formula traceability - Excel-style calculations", expanded=True):
    st.caption(
        "These formulas describe the calculation chain using Excel-style structured "
        "references. In practice, the app calculates the same logic in Python, but "
        "this table gives a traceable audit trail from source fields to outputs."
    )

    theatre_formula_rows = [
        {
            "Area": "Theatre",
            "Metric / helper field": "Session minutes",
            "Source fields": (
                "Theatre session ID; Scheduled start time(Session); Scheduled finish time(Session)"
            ),
            "Excel-style formula": (
                "=MOD([@[Scheduled finish time(Session)]]-[@[Scheduled start time(Session)]],1)*1440"
            ),
            "Connected output": (
                "Feeds scheduled minutes used and 240-minute session equivalents."
            ),
            "Current value / note": (
                "Calculated once per session ID, then summed after filters."
            ),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Actual ran flag",
            "Source fields": (
                "Case Touch time (minutes); Number of cases completed; Actual start time(Session); Actual finish time(Session)"
            ),
            "Excel-style formula": (
                '=OR([@[Case Touch time (minutes)]]>0,[@[Number of cases completed]]>0,'
                '[@[Actual start time(Session)]]<>"",[@[Actual finish time(Session)]]<>"")'
            ),
            "Connected output": "Defines which sessions are counted as delivered.",
            "Current value / note": "Used before utilisation is calculated.",
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Valid utilisation session flag",
            "Source fields": (
                "Elective/Emergency; Specialty (standardised); cancellation/actual ran status; session minutes"
            ),
            "Excel-style formula": (
                '=AND([@[Elective/Emergency]]="Elective",'
                'ISERROR(SEARCH("obstetric",[@[Specialty (standardised)]])),'
                "[@[Actual ran flag]]=TRUE,[@[Session minutes]]>=30,[@[Session minutes]]<=720)"
            ),
            "Connected output": (
                "Filters to elective, non-obstetric, valid actual sessions."
            ),
            "Current value / note": (
                "Excludes emergency/mixed, obstetrics, cancelled/not-run and invalid/24-hour sessions."
            ),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Actual sessions used for utilisation",
            "Source fields": "Theatre session ID; valid utilisation session flag",
            "Excel-style formula": (
                "=ROWS(UNIQUE(FILTER([Theatre session ID],[Valid utilisation session flag]=TRUE)))"
            ),
            "Connected output": "Count of valid elective sessions actually delivered.",
            "Current value / note": format_number(full_year_elective_actual_sessions_used),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Scheduled minutes used",
            "Source fields": "Session-level scheduled minutes after valid utilisation filters",
            "Excel-style formula": (
                "=SUM(FILTER([Session minutes],[Valid utilisation session flag]=TRUE))"
            ),
            "Connected output": (
                "Denominator for utilisation and numerator for 240-minute equivalents."
            ),
            "Current value / note": format_number(full_year_elective_scheduled_minutes),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Actual 240-min session equivalents",
            "Source fields": "Scheduled minutes used",
            "Excel-style formula": "=[Scheduled minutes used]/240",
            "Connected output": (
                "Standardises sessions of different lengths into 4-hour equivalent units."
            ),
            "Current value / note": format_number(full_year_elective_240_session_equivalents),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Touch minutes used",
            "Source fields": (
                "Operation Anaesthetic Start Datetime; Operation Patient in Recovery Datetime; fallback theatre timestamps; Case Touch time (minutes)"
            ),
            "Excel-style formula": (
                "=SUM(FILTER([Model Hospital touch minutes],[Valid utilisation session flag]=TRUE))"
            ),
            "Connected output": "Numerator for theatre utilisation.",
            "Current value / note": format_number(full_year_elective_touch_minutes),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Current utilisation",
            "Source fields": "Touch minutes used; scheduled minutes used",
            "Excel-style formula": "=[Touch minutes used]/[Scheduled minutes used]",
            "Connected output": "Baseline for the 78.5%, 81.75% and 85% target states.",
            "Current value / note": format_percent(full_year_elective_utilisation),
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Average procedure time",
            "Source fields": "Touch minutes used; cases with valid touch time",
            "Excel-style formula": "=[Touch minutes used]/[Cases with valid touch time]",
            "Connected output": "Converts additional utilised minutes into additional cases.",
            "Current value / note": f"{format_decimal(avg_procedure_time, 1)} mins",
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Additional utilised minutes",
            "Source fields": (
                "Baseline scheduled minutes over delivery weeks; current utilisation; target utilisation"
            ),
            "Excel-style formula": (
                "=MAX([Target utilisation]-[Current utilisation],0)*[Baseline scheduled minutes over delivery weeks]"
            ),
            "Connected output": "First step in the same-sessions, more-throughput scenario.",
            "Current value / note": "Calculated separately for 78.5%, 81.75% and 85%.",
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Additional case volume",
            "Source fields": "Additional utilised minutes; average procedure time",
            "Excel-style formula": "=[Additional utilised minutes]/[Average procedure time]",
            "Connected output": "Feeds PTL/backlog impact and indicative finance proxy.",
            "Current value / note": "Scenario output.",
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Closing PTL/backlog",
            "Source fields": "Opening PTL; additional case volume; RTT conversion",
            "Excel-style formula": (
                "=MAX([Opening PTL]-([Additional case volume]*[RTT conversion]),0)"
            ),
            "Connected output": "Shows modelled backlog after theatre additional cases.",
            "Current value / note": f"Opening PTL: {format_number(current_ptl)}",
        },
        {
            "Area": "Theatre",
            "Metric / helper field": "Estate capacity, 240-min sessions",
            "Source fields": (
                "Elective theatre count; sessions per theatre per day; operating days per week; estate weeks"
            ),
            "Excel-style formula": (
                "=[Elective theatres]*[Sessions per theatre per day]*[Days per week]*[Estate weeks]"
            ),
            "Connected output": "Denominator for actual sessions run vs estate capacity.",
            "Current value / note": format_number(estate_capacity_240_sessions),
        },
    ]

    outpatient_formula_rows = [
        {
            "Area": "Outpatients",
            "Metric / helper field": "Planned appointment records",
            "Source fields": "Contact_ID",
            "Excel-style formula": "=ROWS(UNIQUE([Contact_ID]))",
            "Connected output": "Denominator for attendance/fill proxy and DNA rate.",
            "Current value / note": format_number(outpatient_baseline["planned_appointments"]),
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Actual attended appointments",
            "Source fields": "Contact_ID; Status",
            "Excel-style formula": (
                '=ROWS(UNIQUE(FILTER([Contact_ID],ISNUMBER(MATCH([Status],{"Checked In","Checked Out"},0)))))'
            ),
            "Connected output": "Numerator for attendance/fill proxy and baseline activity.",
            "Current value / note": format_number(outpatient_baseline["attended_appointments"]),
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "DNA / no-show appointments",
            "Source fields": "Contact_ID; Status",
            "Excel-style formula": (
                '=ROWS(UNIQUE(FILTER([Contact_ID],[Status]="No Show")))'
            ),
            "Connected output": "Numerator for DNA/no-show rate.",
            "Current value / note": format_number(outpatient_baseline["dna_appointments"]),
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Attendance / fill proxy",
            "Source fields": "Actual attended appointments; planned appointment records",
            "Excel-style formula": "=[Actual attended appointments]/[Planned appointment records]",
            "Connected output": (
                "Proxy only. Used because true empty-slot/template capacity is not available."
            ),
            "Current value / note": f"{outpatient_model_current_fill_pct:.1f}%",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "DNA rate",
            "Source fields": "DNA / no-show appointments; planned appointment records",
            "Excel-style formula": "=[DNA / no-show appointments]/[Planned appointment records]",
            "Connected output": "Baseline for DNA-reduction opportunity.",
            "Current value / note": f"{outpatient_model_current_dna_rate_pct:.1f}%",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Clinic-session proxy",
            "Source fields": "Contact_Start; ContactClinicPerfUnit; Status",
            "Excel-style formula": (
                "=ROWS(UNIQUE(FILTER([Date]&[AM/PM]&[ContactClinicPerfUnit],[Attended flag]=TRUE)))"
            ),
            "Connected output": "Used as actual outpatient session proxy where template data is not available.",
            "Current value / note": format_number(outpatient_baseline["actual_sessions"]),
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Template-fill opportunity",
            "Source fields": (
                "Planned appointments/week; template target fill; current attendance/fill proxy; RTT relevant share; delivery weeks"
            ),
            "Excel-style formula": (
                "=([Planned appts/week]*[RTT relevant share]*MAX([Target fill]-[Current fill proxy],0)*[Delivery weeks])*[State share]"
            ),
            "Connected output": "One component of total additional outpatient appointments.",
            "Current value / note": "State share = 50%, 75%, or 100%.",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "DNA-reduction opportunity",
            "Source fields": (
                "Eligible new + follow-up appointments/week; current DNA rate; target DNA rate; delivery weeks"
            ),
            "Excel-style formula": (
                "=([Eligible appts/week]*MAX([Current DNA rate]-[Target DNA rate],0)*[Delivery weeks])*[State share]"
            ),
            "Connected output": "One component of total additional outpatient appointments.",
            "Current value / note": "State share = 50%, 75%, or 100%.",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "PIFU opportunity",
            "Source fields": "Follow-up attendances/week; PIFU conversion rate; delivery weeks",
            "Excel-style formula": (
                "=([Follow-up/week]*[PIFU conversion rate]*[Delivery weeks])*[State share]"
            ),
            "Connected output": "One component of total additional outpatient appointments.",
            "Current value / note": "Uses follow-up activity as the current proxy base.",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "F:N opportunity",
            "Source fields": "First attendances/week; F:N improvement rate; delivery weeks",
            "Excel-style formula": (
                "=([First attendances/week]*[F:N improvement rate]*[Delivery weeks])*[State share]"
            ),
            "Connected output": "One component of total additional outpatient appointments.",
            "Current value / note": "Uses first-attendance activity as the current proxy base.",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Total additional appointments",
            "Source fields": "Template fill; DNA; PIFU; F:N opportunities",
            "Excel-style formula": (
                "=[Template-fill opportunity]+[DNA-reduction opportunity]+[PIFU opportunity]+[F:N opportunity]"
            ),
            "Connected output": "Feeds RTT backlog impact and outpatient finance proxy.",
            "Current value / note": "Scenario output.",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "RTT backlog impact",
            "Source fields": "Total additional appointments; outpatient RTT conversion",
            "Excel-style formula": "=[Total additional appointments]*[RTT conversion]",
            "Connected output": "Shows pathways reduced by outpatient appointment levers.",
            "Current value / note": f"Current conversion setting: {outpatient_rtt_conversion_pct:.0f}%",
        },
        {
            "Area": "Outpatients",
            "Metric / helper field": "Indicative finance proxy",
            "Source fields": "Total additional appointments; value per appointment",
            "Excel-style formula": (
                "=[Total additional appointments]*[Value per appointment]"
            ),
            "Connected output": "Indicative outpatient cost/capacity opportunity.",
            "Current value / note": format_currency(outpatient_value_per_appointment),
        },
    ]

    formula_traceability_df = pd.DataFrame(
        [*theatre_formula_rows, *outpatient_formula_rows]
    )
    theatre_formula_tab, outpatient_formula_tab, all_formula_tab = st.tabs(
        ["Theatre formulas", "Outpatient formulas", "All formulas"]
    )
    formula_column_config = {
        "Area": st.column_config.TextColumn("Area", width="small"),
        "Metric / helper field": st.column_config.TextColumn(
            "Metric / helper field",
            width="medium",
        ),
        "Source fields": st.column_config.TextColumn("Source fields", width="large"),
        "Excel-style formula": st.column_config.TextColumn(
            "Excel-style formula",
            width="large",
        ),
        "Connected output": st.column_config.TextColumn(
            "Connected output",
            width="large",
        ),
        "Current value / note": st.column_config.TextColumn(
            "Current value / note",
            width="medium",
        ),
    }
    with theatre_formula_tab:
        st.dataframe(
            formula_traceability_df[formula_traceability_df["Area"] == "Theatre"],
            use_container_width=True,
            hide_index=True,
            column_config=formula_column_config,
        )
    with outpatient_formula_tab:
        st.dataframe(
            formula_traceability_df[
                formula_traceability_df["Area"] == "Outpatients"
            ],
            use_container_width=True,
            hide_index=True,
            column_config=formula_column_config,
        )
    with all_formula_tab:
        st.dataframe(
            formula_traceability_df,
            use_container_width=True,
            hide_index=True,
            column_config=formula_column_config,
        )
    st.download_button(
        "Download formula traceability as CSV",
        data=formula_traceability_df.to_csv(index=False).encode("utf-8"),
        file_name="formula_traceability_map.csv",
        mime="text/csv",
    )

st.caption(
    "These are the fixed Apr 2025-Mar 2026 baseline inputs. They are separated "
    "from the utilisation target view so actual-session inputs are not shown as "
    "if they were scenario outputs."
)
st.dataframe(
    baseline_input_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Input area": st.column_config.TextColumn("Input area", width="small"),
        "Metric": st.column_config.TextColumn("Metric", width="medium"),
        "Value": st.column_config.TextColumn("Value", width="small"),
        "Definition / calculation": st.column_config.TextColumn(
            "Definition / calculation",
            width="large",
        ),
        "Why fixed": st.column_config.TextColumn("Why fixed", width="large"),
        "Actions": st.column_config.TextColumn("Actions", width="large"),
    },
)

baseline_input_csv = baseline_input_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download baseline inputs as CSV",
    data=baseline_input_csv,
    file_name="theatre_baseline_inputs.csv",
    mime="text/csv",
)

st.subheader("Theatre Utilisation Target View")
st.caption(
    "One grouped table: the first column shows whether each metric is baseline "
    "context, estate capacity impact, same sessions with more cases, same "
    "throughput with fewer sessions, or RTT backlog impact. The session-equivalent "
    "denominator is the full Apr 2025-Mar 2026 elective 240-minute equivalent."
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "What-if lens / group": st.column_config.TextColumn(
            "What-if lens / group",
            width="large",
        ),
        "Metric": st.column_config.TextColumn("Metric", width="medium"),
        theatre_baseline_column: st.column_config.TextColumn(
            theatre_baseline_column,
            width="small",
        ),
        DISPLAY_SCENARIO_LABELS["What if 50% delivery"]: st.column_config.TextColumn(
            DISPLAY_SCENARIO_LABELS["What if 50% delivery"],
            width="small",
        ),
        DISPLAY_SCENARIO_LABELS["What if 75% delivery"]: st.column_config.TextColumn(
            DISPLAY_SCENARIO_LABELS["What if 75% delivery"],
            width="small",
        ),
        DISPLAY_SCENARIO_LABELS[
            "What if 100% delivery"
        ]: st.column_config.TextColumn(
            DISPLAY_SCENARIO_LABELS["What if 100% delivery"],
            width="small",
        ),
        "Notes / calculation": st.column_config.TextColumn(
            "Notes / calculation",
            width="large",
        ),
        "Actions": st.column_config.TextColumn(
            "Actions",
            width="large",
        ),
    },
)

csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download utilisation target view as CSV",
    data=csv,
    file_name="theatre_baseline_target_utilisation.csv",
    mime="text/csv",
)

st.subheader("Estate and Workforce Capacity Assumptions")
st.caption(
    "These rows are fixed capacity denominators. They do not move when the "
    "utilisation target changes; the utilisation-sensitive row is Estate time "
    "utilised in the main table."
)
st.dataframe(
    capacity_assumption_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Capacity layer": st.column_config.TextColumn(
            "Capacity layer",
            width="small",
        ),
        "Metric": st.column_config.TextColumn("Metric", width="medium"),
        "Value": st.column_config.TextColumn("Value", width="small"),
        "Calculation / source": st.column_config.TextColumn(
            "Calculation / source",
            width="large",
        ),
        "Why fixed": st.column_config.TextColumn("Why fixed", width="large"),
        "Actions": st.column_config.TextColumn("Actions", width="large"),
    },
)

capacity_csv = capacity_assumption_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download capacity assumptions as CSV",
    data=capacity_csv,
    file_name="theatre_capacity_assumptions.csv",
    mime="text/csv",
)

with st.expander("Detailed scenario output table", expanded=False):
    st.caption(
        "Optional detail. The main theatre table above uses the full Apr 2025-Mar "
        "2026 elective 240-minute session-equivalent baseline."
    )
    st.dataframe(
        theatre_scenario_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="large"),
            "Scenario interpretation": st.column_config.TextColumn(
                "Scenario interpretation",
                width="large",
            ),
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Current baseline": st.column_config.TextColumn(
                "Current baseline",
                width="small",
            ),
            DISPLAY_SCENARIO_LABELS[
                "What if 50% delivery"
            ]: st.column_config.TextColumn(
                DISPLAY_SCENARIO_LABELS["What if 50% delivery"],
                width="small",
            ),
            DISPLAY_SCENARIO_LABELS[
                "What if 75% delivery"
            ]: st.column_config.TextColumn(
                DISPLAY_SCENARIO_LABELS["What if 75% delivery"],
                width="small",
            ),
            DISPLAY_SCENARIO_LABELS[
                "What if 100% delivery"
            ]: st.column_config.TextColumn(
                DISPLAY_SCENARIO_LABELS["What if 100% delivery"],
                width="small",
            ),
            "Notes / calculation": st.column_config.TextColumn(
                "Notes / calculation",
                width="large",
            ),
            "Actions": st.column_config.TextColumn(
                "Actions",
                width="large",
            ),
        },
    )

scenario_csv = theatre_scenario_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scenario table as CSV",
    data=scenario_csv,
    file_name="theatre_scenario_outputs.csv",
    mime="text/csv",
)

st.subheader("Scenario Comparison")

chart_df = pd.DataFrame(
    [
        {
            "Scenario": "Current baseline",
            "Scenario A sessions freed": 0,
            "Scenario B additional cases": 0,
            "Remaining PTL": current_ptl,
            "Opportunity value": 0,
        }
    ]
    + [
        {
            "Scenario": DISPLAY_SCENARIO_LABELS[scenario].replace(
                "What if utilisation improves to ",
                "",
            ),
            "Scenario A sessions freed": scenario_a_outputs[scenario][
                "freed_240_sessions"
            ],
            "Scenario B additional cases": scenario_b_outputs[scenario][
                "cases_unlocked"
            ],
            "Remaining PTL": scenario_b_outputs[scenario]["remaining_ptl"],
            "Opportunity value": scenario_b_outputs[scenario]["cost_avoidance"],
        }
        for scenario in scenario_outputs
    ]
)

col1, col2, col3 = st.columns(3)

with col1:
    fig_sessions = px.bar(
        chart_df,
        x="Scenario",
        y="Scenario A sessions freed",
        title="Scenario A: Sessions Freed",
    )
    fig_sessions.update_layout(
        template="plotly_white",
        xaxis_title="Scenario",
        yaxis_title="240-min session equivalents",
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(fig_sessions, use_container_width=True)

with col2:
    fig_cases = px.bar(
        chart_df,
        x="Scenario",
        y="Scenario B additional cases",
        title="Scenario B: Additional Cases",
    )
    fig_cases.update_layout(
        template="plotly_white",
        xaxis_title="Scenario",
        yaxis_title="Additional cases",
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(fig_cases, use_container_width=True)

with col3:
    fig_ptl = px.line(
        chart_df,
        x="Scenario",
        y="Remaining PTL",
        markers=True,
        title="Scenario B: Remaining PTL",
    )
    fig_ptl.update_layout(
        template="plotly_white",
        xaxis_title="Scenario",
        yaxis_title="Remaining PTL",
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(fig_ptl, use_container_width=True)

st.subheader("Specialty Breakdown")
st.caption(
    "Baseline period is April 2025 to March 2026. Theatre is elective-only; "
    "outpatients use the raw contact file and the same appointment-lever logic "
    "as the main outpatient table."
)

theatre_specialty_source = theatre_df[
    (theatre_df["Booked Operation Date"] >= THEATRE_BASELINE_START)
    & (theatre_df["Booked Operation Date"] <= THEATRE_BASELINE_END)
].copy()
specialty_monthly_df, specialty_overall_df = build_specialty_utilisation(
    theatre_specialty_source,
    recent_months=FULL_BASELINE_MONTHS,
)
outpatient_specialty_df = build_outpatient_specialty_breakdown(outpatient_baseline_df)

theatre_specialty_tab, outpatient_specialty_tab = st.tabs(
    ["Theatre specialties", "Outpatient specialties"]
)

with theatre_specialty_tab:
    if specialty_overall_df.empty:
        st.warning("No specialty-level theatre utilisation data is available.")
    else:
        theatre_specialty_display = specialty_overall_df.copy()
        if exclude_unknown_specialty:
            theatre_specialty_display = theatre_specialty_display[
                theatre_specialty_display["Specialty (standardised)"].str.lower()
                != "unknown"
            ].copy()

        theatre_specialty_display["Actual 240-min session equivalents"] = (
            theatre_specialty_display["Scheduled_Minutes"] / SESSION_STANDARD_MINUTES
        )
        theatre_specialty_display["Cases per 240-min session"] = (
            theatre_specialty_display["Completed_Cases"]
            / theatre_specialty_display["Actual 240-min session equivalents"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)
        theatre_specialty_display["Average case duration (mins)"] = (
            theatre_specialty_display["Touch_Minutes"]
            / theatre_specialty_display["Completed_Cases"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        for scenario, target_utilisation in SCENARIO_TARGETS.items():
            label = DISPLAY_SCENARIO_LABELS[scenario].replace(
                "What if utilisation improves to ",
                "Additional cases at ",
            )
            additional_minutes = (
                theatre_specialty_display["Scheduled_Minutes"] * target_utilisation
                - theatre_specialty_display["Touch_Minutes"]
            ).clip(lower=0)
            theatre_specialty_display[label] = (
                additional_minutes
                / theatre_specialty_display["Average case duration (mins)"].replace(
                    0,
                    pd.NA,
                )
            ).fillna(0)

        theatre_specialty_display["Utilisation %"] = (
            theatre_specialty_display["Utilisation"] * 100
        )
        theatre_specialty_display = theatre_specialty_display.sort_values(
            "Utilisation",
            ascending=True,
        )

        theatre_specialty_preview = theatre_specialty_display[
            [
                "Specialty (standardised)",
                "Specialty_Sessions",
                "Actual 240-min session equivalents",
                "Scheduled_Minutes",
                "Touch_Minutes",
                "Completed_Cases",
                "Utilisation %",
                "Cases per 240-min session",
                "Average case duration (mins)",
                "Additional cases at 78.5%",
                "Additional cases at 81.75%",
                "Additional cases at 85%",
            ]
        ].head(40).copy()
        theatre_specialty_preview = theatre_specialty_preview.rename(
            columns={
                "Specialty (standardised)": "Specialty",
                "Specialty_Sessions": "Session-specialty records",
                "Actual 240-min session equivalents": "Actual 240-min sessions",
                "Scheduled_Minutes": "Allocated scheduled minutes",
                "Touch_Minutes": "Touch minutes",
                "Completed_Cases": "Completed cases",
                "Utilisation %": "Utilisation %",
                "Cases per 240-min session": "Cases / 240-min session",
                "Average case duration (mins)": "Avg case mins",
                "Additional cases at 78.5%": "Addl cases @ 78.5%",
                "Additional cases at 81.75%": "Addl cases @ 81.8%",
                "Additional cases at 85%": "Addl cases @ 85%",
            }
        )
        for col in [
            "Session-specialty records",
            "Allocated scheduled minutes",
            "Touch minutes",
            "Completed cases",
            "Addl cases @ 78.5%",
            "Addl cases @ 81.8%",
            "Addl cases @ 85%",
        ]:
            theatre_specialty_preview[col] = theatre_specialty_preview[col].map(
                lambda value: format_number(float(value))
            )
        for col in [
            "Actual 240-min sessions",
            "Utilisation %",
            "Cases / 240-min session",
            "Avg case mins",
        ]:
            theatre_specialty_preview[col] = theatre_specialty_preview[col].map(
                lambda value: format_decimal(float(value), 1)
            )
        st.markdown(
            theatre_specialty_preview.to_html(index=False, escape=True),
            unsafe_allow_html=True,
        )
        st.caption(
            "Showing the first 40 rows as a static preview to avoid browser "
            "rendering errors. Use the download for the full specialty file."
        )
        st.download_button(
            "Download theatre specialty breakdown",
            data=theatre_specialty_display.to_csv(index=False).encode("utf-8"),
            file_name="theatre_specialty_breakdown.csv",
            mime="text/csv",
        )
        st.caption(
            "Theatre specialty view uses the same elective, non-obstetric valid "
            "actual-session baseline as the agreed theatre table. Scheduled minutes "
            "are allocated to specialty by touch-time share; where touch time is "
            "unavailable, case share is used. Session-specialty records can sum "
            "above the total session count where one actual session contains more "
            "than one specialty."
        )

        elective_golden = theatre_activity_split_df[
            theatre_activity_split_df["Session type"] == "Elective"
        ]
        model_golden = theatre_activity_split_df[
            theatre_activity_split_df["Session type"]
            == "Elective excl obstetrics (model baseline)"
        ]
        if not elective_golden.empty and not model_golden.empty:
            elective_golden_row = elective_golden.iloc[0]
            model_golden_row = model_golden.iloc[0]

            golden_scheduled_minutes = float(
                elective_golden_row["Scheduled minutes used"]
            )
            golden_touch_minutes = float(elective_golden_row["Touch minutes used"])
            golden_completed_cases = float(elective_golden_row["Completed cases"])
            golden_240_sessions = float(
                elective_golden_row["Actual 240-min session equivalents"]
            )
            golden_current_utilisation = (
                golden_touch_minutes / golden_scheduled_minutes
                if golden_scheduled_minutes > 0
                else 0
            )
            golden_average_procedure_time = (
                float(model_golden_row["Touch minutes used"])
                / float(model_golden_row["Completed cases"])
                if float(model_golden_row["Completed cases"]) > 0
                else 0
            )
            golden_cost_per_scheduled_minute = (
                theatre_cost_2526 / float(model_golden_row["Scheduled minutes used"])
                if float(model_golden_row["Scheduled minutes used"]) > 0
                else 0
            )

            specialty_quantum = theatre_specialty_display.copy()
            total_specialty_scheduled_minutes = specialty_quantum[
                "Scheduled_Minutes"
            ].sum()
            specialty_quantum["Golden-source allocation share"] = (
                specialty_quantum["Scheduled_Minutes"]
                / total_specialty_scheduled_minutes
                if total_specialty_scheduled_minutes > 0
                else 0
            )
            specialty_quantum["Allocation share %"] = (
                specialty_quantum["Golden-source allocation share"] * 100
            )
            specialty_quantum["Observed valid-session completed cases"] = specialty_quantum[
                "Completed_Cases"
            ]
            specialty_quantum["Allocated elective completed cases"] = (
                golden_completed_cases
                * specialty_quantum["Golden-source allocation share"]
            )
            specialty_quantum["Allocated elective 240-min sessions"] = (
                golden_240_sessions
                * specialty_quantum["Golden-source allocation share"]
            )
            specialty_quantum["Baseline utilisation %"] = (
                specialty_quantum["Utilisation"] * 100
            )

            for scenario, target_utilisation in SCENARIO_TARGETS.items():
                target_label = DISPLAY_SCENARIO_LABELS[scenario].replace(
                    "What if utilisation improves to ",
                    "",
                )
                full_additional_minutes = golden_scheduled_minutes * max(
                    target_utilisation - golden_current_utilisation,
                    0,
                )
                full_additional_cases = (
                    full_additional_minutes / golden_average_procedure_time
                    if golden_average_procedure_time > 0
                    else 0
                )
                full_required_240_sessions = (
                    golden_touch_minutes
                    / target_utilisation
                    / SESSION_STANDARD_MINUTES
                    if target_utilisation > 0
                    else 0
                )
                full_sessions_freed = max(
                    golden_240_sessions - full_required_240_sessions,
                    0,
                )
                full_capacity_value = (
                    full_sessions_freed
                    * SESSION_STANDARD_MINUTES
                    * golden_cost_per_scheduled_minute
                )

                specialty_quantum[f"Additional cases @ {target_label}"] = (
                    full_additional_cases
                    * specialty_quantum["Golden-source allocation share"]
                )
                specialty_quantum[f"Total cases @ {target_label}"] = (
                    specialty_quantum["Allocated elective completed cases"]
                    + specialty_quantum[f"Additional cases @ {target_label}"]
                )
                specialty_quantum[f"Sessions freed @ {target_label}"] = (
                    full_sessions_freed
                    * specialty_quantum["Golden-source allocation share"]
                )
                specialty_quantum[f"RTT reduction @ {target_label}"] = (
                    specialty_quantum[f"Additional cases @ {target_label}"]
                )
                specialty_quantum[f"Closing backlog @ {target_label}"] = (
                    current_ptl - specialty_quantum[f"RTT reduction @ {target_label}"]
                )
                specialty_quantum[f"Capacity value @ {target_label}"] = (
                    full_capacity_value
                    * specialty_quantum["Golden-source allocation share"]
                )
                specialty_quantum[f"Per-case value @ {target_label}"] = (
                    specialty_quantum[f"Additional cases @ {target_label}"]
                    * THEATRE_CASE_VALUE_DEFAULT
                )

            specialty_quantum_display = specialty_quantum[
                [
                    "Specialty (standardised)",
                    "Allocation share %",
                    "Specialty_Sessions",
                    "Allocated elective 240-min sessions",
                    "Observed valid-session completed cases",
                    "Allocated elective completed cases",
                    "Baseline utilisation %",
                    "Additional cases @ 78.5%",
                    "Additional cases @ 81.75%",
                    "Additional cases @ 85%",
                    "Total cases @ 78.5%",
                    "Total cases @ 81.75%",
                    "Total cases @ 85%",
                    "Sessions freed @ 78.5%",
                    "Sessions freed @ 81.75%",
                    "Sessions freed @ 85%",
                    "RTT reduction @ 78.5%",
                    "RTT reduction @ 81.75%",
                    "RTT reduction @ 85%",
                    "Closing backlog @ 78.5%",
                    "Closing backlog @ 81.75%",
                    "Closing backlog @ 85%",
                    "Capacity value @ 78.5%",
                    "Capacity value @ 81.75%",
                    "Capacity value @ 85%",
                    "Per-case value @ 78.5%",
                    "Per-case value @ 81.75%",
                    "Per-case value @ 85%",
                ]
            ].sort_values("Additional cases @ 85%", ascending=False)

            with st.expander(
                "Theatre specialty view using golden-source full quantum",
                expanded=True,
            ):
                st.caption(
                    "This view takes the agreed full-theatre quantum and allocates "
                    "it to specialties by each specialty's share of elective scheduled "
                    "minutes. That keeps the specialty totals reconcilable to the "
                    "golden-source table while still showing the specialty split. "
                    f"Golden-source inputs: elective utilisation {format_percent(golden_current_utilisation)}, "
                    f"elective scheduled minutes {format_number(golden_scheduled_minutes)}, "
                    f"elective completed cases {format_number(golden_completed_cases)}, "
                    f"elective 240-min sessions {format_decimal(golden_240_sessions, 1)}, "
                    f"average procedure time {format_decimal(golden_average_procedure_time, 1)} mins, "
                    f"cost per scheduled minute {format_currency(golden_cost_per_scheduled_minute)}."
                )

                top_summary = specialty_quantum_display.head(5)
                top_summary_lines = []
                for _, row in top_summary.iterrows():
                    top_summary_lines.append(
                        "- "
                        f"{row['Specialty (standardised)']}: "
                        f"{format_decimal(float(row['Additional cases @ 85%']), 1)} "
                        "additional cases at 85%, "
                        f"{format_decimal(float(row['Sessions freed @ 85%']), 1)} "
                        "240-min sessions freed, "
                        f"{format_currency(float(row['Capacity value @ 85%']))} "
                        "capacity value."
                    )
                st.markdown(
                    "**Top specialties by additional case volume at 85%**\n"
                    + "\n".join(top_summary_lines)
                )
                st.info(
                    "The full specialty-level reconciliation is available as a CSV "
                    "download below. The on-page preview is deliberately lightweight "
                    "to avoid the browser rendering issue."
                )
                st.download_button(
                    "Download theatre specialty golden-source view",
                    data=specialty_quantum_display.to_csv(index=False).encode("utf-8"),
                    file_name="theatre_specialty_golden_source_view.csv",
                    mime="text/csv",
                )
                st.markdown(
                    """
**Justification**
- The full quantum is calculated once from the agreed theatre table, then allocated to specialties so the specialty values reconcile back to the total.
- Allocation uses each specialty's share of elective scheduled minutes because utilisation improvement is a time-capacity opportunity.
- RTT reduction assumes one additional case removes one RTT pathway.
- Closing backlog is shown as an independent specialty contribution, not a cumulative specialty-by-specialty run-down.
- Capacity value uses sessions-freed value; per-case value uses additional case volume x GBP250.
                    """
                )

with outpatient_specialty_tab:
    if outpatient_error is not None:
        st.warning(f"Outpatient specialty data could not be loaded: {outpatient_error}.")
    elif outpatient_specialty_df.empty:
        st.warning("No specialty-level outpatient data is available.")
    else:
        outpatient_specialty_display = outpatient_specialty_df.copy()
        observed_weeks_for_specialty = max(outpatient_baseline["observed_weeks"], 1)
        outpatient_specialty_display["Planned appointments / week"] = (
            outpatient_specialty_display["Planned appointment records"]
            / observed_weeks_for_specialty
        )
        outpatient_specialty_display["Eligible new + follow-up / week"] = (
            outpatient_specialty_display["First attendances"]
            + outpatient_specialty_display["Follow-up attendances"]
        ) / observed_weeks_for_specialty
        outpatient_specialty_display["Follow-up / week"] = (
            outpatient_specialty_display["Follow-up attendances"]
            / observed_weeks_for_specialty
        )
        outpatient_specialty_display["First attendances / week"] = (
            outpatient_specialty_display["First attendances"]
            / observed_weeks_for_specialty
        )

        outpatient_specialty_display["Template-fill opportunity"] = (
            outpatient_specialty_display["Planned appointments / week"]
            * (outpatient_template_rtt_share_pct / 100)
            * (
                outpatient_template_target_fill_pct / 100
                - outpatient_specialty_display["Attendance / fill proxy"]
            ).clip(lower=0)
            * active_delivery_weeks
        )
        outpatient_specialty_display["DNA-reduction opportunity"] = (
            outpatient_specialty_display["Eligible new + follow-up / week"]
            * (
                outpatient_specialty_display["DNA rate"]
                - outpatient_target_dna_rate_pct / 100
            ).clip(lower=0)
            * active_delivery_weeks
        )
        outpatient_specialty_display["PIFU opportunity"] = (
            outpatient_specialty_display["Follow-up / week"]
            * outpatient_pifu_conversion_pct
            / 100
            * active_delivery_weeks
        )
        outpatient_specialty_display["F:N opportunity"] = (
            outpatient_specialty_display["First attendances / week"]
            * outpatient_fn_ratio_improvement_pct
            / 100
            * active_delivery_weeks
        )
        outpatient_specialty_display["Full additional appointments"] = (
            outpatient_specialty_display["Template-fill opportunity"]
            + outpatient_specialty_display["DNA-reduction opportunity"]
            + outpatient_specialty_display["PIFU opportunity"]
            + outpatient_specialty_display["F:N opportunity"]
        )
        outpatient_specialty_display["RTT backlog impact"] = (
            outpatient_specialty_display["Full additional appointments"]
            * outpatient_rtt_conversion_pct
            / 100
        )
        outpatient_specialty_display["Indicative finance proxy"] = (
            outpatient_specialty_display["Full additional appointments"]
            * outpatient_value_per_appointment
        )
        outpatient_specialty_display["State 1 additional appts"] = (
            outpatient_specialty_display["Full additional appointments"] * 0.5
        )
        outpatient_specialty_display["State 2 additional appts"] = (
            outpatient_specialty_display["Full additional appointments"] * 0.75
        )
        outpatient_specialty_display["State 3 additional appts"] = (
            outpatient_specialty_display["Full additional appointments"]
        )

        outpatient_specialty_display["DNA rate %"] = (
            outpatient_specialty_display["DNA rate"] * 100
        )
        outpatient_specialty_display["Attendance / fill proxy %"] = (
            outpatient_specialty_display["Attendance / fill proxy"] * 100
        )

        st.dataframe(
            outpatient_specialty_display[
                [
                    "Specialty",
                    "Planned appointment records",
                    "Actual attended appointments",
                    "DNA / no-show appointments",
                    "DNA rate %",
                    "Attendance / fill proxy %",
                    "First attendances",
                    "Follow-up attendances",
                    "Actual clinic-session proxies",
                    "Attended appointments / actual session",
                    "Template-fill opportunity",
                    "DNA-reduction opportunity",
                    "PIFU opportunity",
                    "F:N opportunity",
                    "State 1 additional appts",
                    "State 2 additional appts",
                    "State 3 additional appts",
                    "RTT backlog impact",
                    "Indicative finance proxy",
                ]
            ].sort_values("State 3 additional appts", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Specialty": st.column_config.TextColumn("Specialty", width="large"),
                "Planned appointment records": st.column_config.NumberColumn(
                    "Planned appts",
                    format="%.0f",
                ),
                "Actual attended appointments": st.column_config.NumberColumn(
                    "Attended appts",
                    format="%.0f",
                ),
                "DNA / no-show appointments": st.column_config.NumberColumn(
                    "DNA/no-show",
                    format="%.0f",
                ),
                "DNA rate %": st.column_config.NumberColumn(
                    "DNA rate %",
                    format="%.1f%%",
                ),
                "Attendance / fill proxy %": st.column_config.NumberColumn(
                    "Attendance/fill proxy %",
                    format="%.1f%%",
                ),
                "Actual clinic-session proxies": st.column_config.NumberColumn(
                    "Actual clinic-session proxies",
                    format="%.0f",
                ),
                "Attended appointments / actual session": st.column_config.NumberColumn(
                    "Attended / actual session",
                    format="%.1f",
                ),
                "Template-fill opportunity": st.column_config.NumberColumn(
                    "Template-fill opportunity",
                    format="%.0f",
                ),
                "DNA-reduction opportunity": st.column_config.NumberColumn(
                    "DNA opportunity",
                    format="%.0f",
                ),
                "PIFU opportunity": st.column_config.NumberColumn(
                    "PIFU opportunity",
                    format="%.0f",
                ),
                "F:N opportunity": st.column_config.NumberColumn(
                    "F:N opportunity",
                    format="%.0f",
                ),
                "State 1 additional appts": st.column_config.NumberColumn(
                    "State 1 addl appts",
                    format="%.0f",
                ),
                "State 2 additional appts": st.column_config.NumberColumn(
                    "State 2 addl appts",
                    format="%.0f",
                ),
                "State 3 additional appts": st.column_config.NumberColumn(
                    "State 3 addl appts",
                    format="%.0f",
                ),
                "RTT backlog impact": st.column_config.NumberColumn(
                    "RTT backlog impact",
                    format="%.0f",
                ),
                "Indicative finance proxy": st.column_config.NumberColumn(
                    "Finance proxy",
                    format="£%.0f",
                ),
            },
        )
        st.download_button(
            "Download outpatient specialty breakdown",
            data=outpatient_specialty_display.to_csv(index=False).encode("utf-8"),
            file_name="outpatient_specialty_breakdown.csv",
            mime="text/csv",
        )
        st.caption(
            "Outpatient specialty view uses Standardised_Specialty from the outpatient loader. "
            "Template fill remains an attended/planned appointment-record proxy because the current extract does not include empty template slots or full clinic-template capacity."
        )

st.subheader("Lowest Utilisation Specialties")

if specialty_monthly_df.empty or specialty_overall_df.empty:
    st.warning("No specialty-level theatre utilisation data is available.")
else:
    if exclude_unknown_specialty:
        specialty_monthly_df = specialty_monthly_df[
            specialty_monthly_df["Specialty (standardised)"].str.lower()
            != "unknown"
        ].copy()
        specialty_overall_df = specialty_overall_df[
            specialty_overall_df["Specialty (standardised)"].str.lower()
            != "unknown"
        ].copy()

    eligible_specialties = specialty_overall_df[
        specialty_overall_df["Specialty_Sessions"] >= min_heatmap_sessions
    ].copy()

    if eligible_specialties.empty:
        st.warning(
            "No specialties meet the current minimum session threshold for the heatmap."
        )
    else:
        lowest_specialties = eligible_specialties.sort_values(
            "Utilisation",
            ascending=True,
        ).head(5)
        specialty_order = lowest_specialties["Specialty (standardised)"].tolist()

        heatmap_source = specialty_monthly_df[
            specialty_monthly_df["Specialty (standardised)"].isin(specialty_order)
        ].copy()
        heatmap_source["Month_Label"] = heatmap_source["Month"].dt.strftime("%b %Y")
        month_order = (
            heatmap_source[["Month", "Month_Label"]]
            .drop_duplicates()
            .sort_values("Month")["Month_Label"]
            .tolist()
        )
        heatmap_source["Utilisation_Percent"] = (
            heatmap_source["Utilisation"] * 100
        )

        heatmap_pivot = (
            heatmap_source.pivot_table(
                index="Specialty (standardised)",
                columns="Month_Label",
                values="Utilisation_Percent",
                aggfunc="mean",
            )
            .reindex(index=specialty_order, columns=month_order)
        )

        st.markdown("**Monthly utilisation for five lowest-utilisation specialties**")
        heatmap_display = heatmap_pivot.copy().round(1)
        heatmap_display = heatmap_display.map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1f}%"
        )
        st.markdown(
            heatmap_display.to_html(escape=True),
            unsafe_allow_html=True,
        )

        specialty_summary_display = lowest_specialties[
            [
                "Specialty (standardised)",
                "Specialty_Sessions",
                "Scheduled_Minutes",
                "Touch_Minutes",
                "Completed_Cases",
                "Utilisation",
            ]
        ].copy()
        specialty_summary_display["Utilisation"] = (
            specialty_summary_display["Utilisation"] * 100
        )
        specialty_summary_display = specialty_summary_display.rename(
            columns={
                "Specialty (standardised)": "Specialty",
                "Specialty_Sessions": "Session-specialty records",
                "Scheduled_Minutes": "Allocated scheduled minutes",
                "Touch_Minutes": "Touch minutes",
                "Completed_Cases": "Completed cases",
                "Utilisation": "Utilisation %",
            }
        )
        for col in [
            "Session-specialty records",
            "Allocated scheduled minutes",
            "Touch minutes",
            "Completed cases",
        ]:
            specialty_summary_display[col] = specialty_summary_display[col].map(
                lambda value: format_number(float(value))
            )
        specialty_summary_display["Utilisation %"] = specialty_summary_display[
            "Utilisation %"
        ].map(lambda value: f"{float(value):.1f}%")

        st.markdown("**Lowest-utilisation specialty summary**")
        st.markdown(
            specialty_summary_display.to_html(index=False, escape=True),
            unsafe_allow_html=True,
        )

        st.caption(
            "Specialty heatmap uses elective, non-obstetric sessions from the full theatre period. Touch time uses anaesthetic-to-recovery where available; touch-time rows over 720 minutes are excluded. Where a session has multiple specialties, scheduled minutes are allocated by touch-time share; if no touch time exists, case share is used; if neither exists, the session is split equally."
        )

st.subheader("Outpatient Appointment Impact Table")
st.caption(
    "Baseline is fixed to Apr 2025-Mar 2026. The three outpatient states show what happens as the appointment opportunity dial moves from partial to full delivery."
)

if outpatient_error is not None:
    st.warning(
        f"Outpatient raw baseline could not be loaded: {outpatient_error}. "
        "The lever calculations still run from the planning assumptions."
    )

outpatient_display_scenario_labels = {
    scenario: (
        f"Appointment state {index} "
        f"(+{format_number(output['additional_monthly'])} appts/month)"
    )
    for index, (scenario, output) in enumerate(outpatient_outputs.items(), start=1)
}

st.info(
    "Outpatient baseline activity is calculated from Apr 2025-Mar 2026 raw outpatient data. "
    "Planned appointments, attended appointments, DNA rate and clinic-session proxies are data-led. True room/template estate capacity still requires a clinic-template or room-capacity extract."
)

outpatient_rows = [
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Baseline label",
        "Current baseline": (
            f"Average of {outpatient_baseline_period_label}, annualised over "
            f"{format_decimal(active_delivery_weeks, 1)} weeks"
            if not outpatient_baseline_df.empty
            else f"No outpatient records found for {outpatient_baseline_period_label}"
        ),
        "What if 50% delivery": "Same baseline",
        "What if 75% delivery": "Same baseline",
        "What if 100% delivery": "Same baseline",
        "Notes / calculation": (
            "Uses the raw outpatient contact records and status values. "
            "This is now data-led rather than purely top-down target modelling."
        ),
        "Actions": "Baseline period fixed to Apr 2025-Mar 2026 as requested.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Planned appointment records",
        "Current baseline": format_number(outpatient_planned_appointments_horizon),
        "What if 50% delivery": "Input to target model",
        "What if 75% delivery": "Input to target model",
        "What if 100% delivery": "Input to target model",
        "Notes / calculation": (
            "Unique outpatient Contact_ID records annualised over the delivery "
            "period. Includes attended, DNA/no-show, confirmed/hold/scheduled "
            "records present in the extract."
        ),
        "Actions": "Validate that Contact_ID records represent booked/planned appointments and not only completed activity.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Actual attended appointments",
        "Current baseline": format_number(outpatient_attended_appointments_horizon),
        "What if 50% delivery": "Input to target model",
        "What if 75% delivery": "Input to target model",
        "What if 100% delivery": "Input to target model",
        "Notes / calculation": (
            "Attended appointments = Status Checked In or Checked Out, "
            "counted once per Contact_ID and annualised over the delivery period."
        ),
        "Actions": "Answered: Checked In and Checked Out count as attended; retain one attendance per Contact_ID.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Planned clinic-session proxies",
        "Current baseline": format_number(outpatient_planned_sessions_horizon),
        "What if 50% delivery": "Held constant in throughput view",
        "What if 75% delivery": "Held constant in throughput view",
        "What if 100% delivery": "Held constant in throughput view",
        "Notes / calculation": (
            "Proxy session = clinic/performance unit + date + AM/PM. This gives "
            "a 240-minute equivalent clinic-session view where a template extract "
            "is not yet available."
        ),
        "Actions": "Replace with Cerner clinic template/session extract when available.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Actual clinic-session proxies",
        "Current baseline": format_number(outpatient_actual_sessions_horizon),
        "What if 50% delivery": "Held constant in throughput view",
        "What if 75% delivery": "Held constant in throughput view",
        "What if 100% delivery": "Held constant in throughput view",
        "Notes / calculation": (
            "Planned clinic-session proxies with at least one Checked In or "
            "Checked Out appointment."
        ),
        "Actions": "Validate proxy against actual clinic-session reporting.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "Booked appointment fill / attendance rate",
        "Current baseline": f"{outpatient_model_current_fill_pct:.1f}%",
        "What if 50% delivery": outpatient_value(
            "template_fill_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "template_fill_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "template_fill_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Data-led rate = attended appointments / planned appointment records. "
            "This is not true template fill. The current outpatient extract has "
            "no empty template-slot, booked-slot, room, or clinic-template "
            "capacity fields, so measured template fill is not available."
        ),
        "Actions": "No true template-fill data available: request Cerner template slots, booked slots, empty slots, rooms and planned clinics.",
    },
    {
        "Opportunity": "Outpatients - data-led baseline",
        "Metric": "DNA / no-show rate",
        "Current baseline": f"{outpatient_model_current_dna_rate_pct:.1f}%",
        "What if 50% delivery": outpatient_value(
            "dna_rate_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "dna_rate_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "dna_rate_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "DNA rate = Status No Show / planned appointment records."
        ),
        "Actions": "Validate whether short-notice cancellations should be added to this rate if available.",
    },
    {
        "Opportunity": "Outpatients - capacity layer",
        "Metric": "Estate capacity: rooms/templates",
        "Current baseline": "Not in current raw file",
        "What if 50% delivery": "Not modelled",
        "What if 75% delivery": "Not modelled",
        "What if 100% delivery": "Not modelled",
        "Notes / calculation": (
            "The outpatient activity file has clinic/performance-unit labels but "
            "does not contain room capacity or full template slot availability."
        ),
        "Actions": "Request Cerner clinic-template extract: rooms, planned sessions, template slots, booked slots, cancelled slots.",
    },
    {
        "Opportunity": "Outpatients - capacity layer",
        "Metric": "Workforce capacity from job plans",
        "Current baseline": format_number(workforce_outpatient_capacity_240_sessions),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            "Latest substantive job-plan Out-patient activities x delivery weeks. "
            f"Source: {job_plan_capacity['source']}."
        ),
        "Actions": "Validate whether Out-patient activities maps to weekly 240-minute DCC sessions and agree specialty scope.",
    },
    {
        "Opportunity": "Outpatients - capacity layer",
        "Metric": "Workforce utilisation",
        "Current baseline": format_percent(workforce_outpatient_utilisation),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Notes / calculation": (
            "Actual clinic-session proxies / substantive outpatient job-plan "
            "activity capacity."
        ),
        "Actions": "Treat as directional until clinic sessions and job-plan scope are reconciled.",
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "Template fill target (proxy only - no template-slot data)",
        "Current baseline": f"{outpatient_model_current_fill_pct:.1f}%",
        "What if 50% delivery": outpatient_value(
            "template_fill_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "template_fill_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "template_fill_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "This is an attendance/fill proxy from the outpatient contact file, "
            f"not true template fill. No empty-slot or template-capacity data is "
            f"available in the current extract. The proxy is calculated as "
            f"{format_number(outpatient_baseline['attended_appointments'])} "
            "Checked In/Checked Out contacts / "
            f"{format_number(outpatient_baseline['planned_appointments'])} "
            f"Contact_ID records = {outpatient_model_current_fill_pct:.1f}%. "
            f"The user target is {outpatient_template_target_fill_pct:.1f}%; "
            "if the observed proxy is already above the target, the model holds "
            "the target columns at the observed rate and creates no template-fill uplift."
        ),
        "Actions": "Answered: no measured template-fill data is available; 90.7% is only the attended/planned Contact_ID proxy.",
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "DNA rate target",
        "Current baseline": f"{outpatient_model_current_dna_rate_pct:.1f}%",
        "What if 50% delivery": outpatient_value(
            "dna_rate_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "dna_rate_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "dna_rate_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Appointment-state DNA reduction. It moves from "
            f"{outpatient_model_current_dna_rate_pct:.1f}% toward "
            f"{outpatient_target_dna_rate_pct:.1f}% across the three appointment states."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "PIFU conversion target",
        "Current baseline": "0.0%",
        "What if 50% delivery": outpatient_value(
            "pifu_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "pifu_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "pifu_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Target share of eligible follow-up slots released or redirected "
            f"through PIFU, moving toward {outpatient_pifu_conversion_pct:.1f}%."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "F:N improvement target",
        "Current baseline": "0.0%",
        "What if 50% delivery": outpatient_value(
            "fn_ratio_target",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "fn_ratio_target",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "fn_ratio_target",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Target improvement applied to eligible new appointment capacity, "
            f"moving toward {outpatient_fn_ratio_improvement_pct:.1f}%."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "Current outpatient attendances / month",
        "Current baseline": (
            format_number(outpatient_baseline_monthly)
            if outpatient_error is None
            else "Not available"
        ),
        "What if 50% delivery": outpatient_value(
            "activity_after_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "activity_after_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "activity_after_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Baseline is the average monthly attended outpatient contacts from "
            f"the raw outpatient files across {outpatient_month_count} months; "
            f"latest month loaded is {outpatient_latest_month_label}. What-if "
            "columns add the modelled additional appointments per month."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "Template fill additional appts / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "template_fill_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "template_fill_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "template_fill_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Template fill = clinic sessions/week x patients/session x "
            "RTT-relevant template share x template fill gap x in-year delivery "
            "weeks x target share / horizon months."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "DNA reduction additional appts / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "dna_reduction_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "dna_reduction_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "dna_reduction_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "DNA reduction = eligible new + follow-up appointments/week x DNA "
            "rate improvement x in-year delivery weeks x target share / "
            "horizon months."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "PIFU additional appts / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "pifu_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "pifu_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "pifu_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "PIFU = eligible follow-up appointments/week x FU slots moved via "
            "PIFU x in-year delivery weeks x target share / horizon months."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "F:N ratio additional appts / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "fn_ratio_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "fn_ratio_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "fn_ratio_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "F:N ratio improvement = eligible new appointments/week x F:N "
            "improvement rate x in-year delivery weeks x target share / "
            "horizon months."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "Total additional appts / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "additional_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "additional_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "additional_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Sum of template fill, DNA reduction, PIFU, and F:N ratio "
            "additional appointments per month."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": f"Total additional appts over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "additional_total",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "additional_total",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "additional_total",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Total additional appointments = monthly additional appointments x "
            "horizon months. With defaults this aligns to the planning view of "
            "approximately 1,730 / 2,590 / 3,460 over 10 months."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": f"Indicative cost / capacity opportunity over {outpatient_horizon_months} months",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "financial_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "financial_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "financial_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Total additional appointments over the horizon x "
            f"{format_currency(outpatient_value_per_appointment)} per extra "
            "appointment. This is the cost/capacity lens. The default value is a planning proxy based on the "
            "Opportunity model reference range of GBP150-GBP250 per outpatient "
            "slot; it is not automatically cashable."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": f"Indicative income opportunity over {outpatient_horizon_months} months",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "income_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "income_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "income_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Total additional appointments over the horizon x observed 25/26 "
            f"outpatient income per attended contact ({format_currency(outpatient_income_per_attendance)}). "
            f"Evidence: {outpatient_finance_evidence}"
        ),
        "Actions": "Income is a gross income lens; Finance should confirm tariff/payment treatment and whether activity is genuinely incremental.",
    },
    {
        "Opportunity": "Outpatients",
        "Metric": f"Appointment activity after levers over {outpatient_horizon_months} months",
        "Current baseline": (
            format_number(outpatient_baseline_total)
            if outpatient_error is None
            else "Not available"
        ),
        "What if 50% delivery": outpatient_value(
            "activity_after_total",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "activity_after_total",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "activity_after_total",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Baseline outpatient attendances over the horizon plus the total "
            "additional appointments unlocked by the four outpatient levers."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": f"RTT/PTL pathways reduced over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Total additional appointments x outpatient RTT/PTL conversion %. "
            "Default assumes 100% conversion, so one additional outpatient slot "
            "removes one RTT/PTL pathway."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "PTL after outpatient levers",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": outpatient_value(
            "remaining_ptl",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "remaining_ptl",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "remaining_ptl",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            f"Latest PTL month: {latest_ptl_month.strftime('%B %Y')}. "
            "What-if PTL = current PTL - converted RTT/PTL pathway reduction."
        ),
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "PTL reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": outpatient_value(
            "ptl_reduction_pct",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "ptl_reduction_pct",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "ptl_reduction_pct",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Converted RTT/PTL pathway reduction divided by current PTL.",
    },
    {
        "Opportunity": "Outpatients",
        "Metric": "Remaining PTL %",
        "Current baseline": "100.0%",
        "What if 50% delivery": outpatient_value(
            "remaining_ptl_pct",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "remaining_ptl_pct",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "remaining_ptl_pct",
            "What if 100% delivery",
        ),
        "Notes / calculation": "PTL after outpatient levers / current PTL.",
    },
]

for row in outpatient_rows:
    row.setdefault(
        "Actions",
        "Validate planning assumption and confirm owner before using as committed benefit.",
    )

outpatient_baseline_label = (
    f"{outpatient_baseline_period_label}, annualised over "
    f"{format_decimal(active_delivery_weeks, 1)} weeks"
    if not outpatient_baseline_df.empty
    else f"No outpatient records found for {outpatient_baseline_period_label}"
)
outpatient_template_capacity_columns = [
    col
    for col in outpatient_df.columns
    if any(token in col.lower() for token in ["template", "slot", "room"])
]
outpatient_rtt_columns = [
    col
    for col in outpatient_df.columns
    if any(token in col.lower() for token in ["rtt", "pathway", "waiting", "ptl"])
]
outpatient_status_counts = (
    outpatient_baseline_df["Status"].value_counts().head(6).to_dict()
    if "Status" in outpatient_baseline_df.columns and not outpatient_baseline_df.empty
    else {}
)
outpatient_source_files = (
    ", ".join(sorted(outpatient_baseline_df["Source_File"].dropna().unique()))
    if "Source_File" in outpatient_baseline_df.columns and not outpatient_baseline_df.empty
    else "Not available"
)

outpatient_baseline_input_rows = [
    {
        "Input area": "Baseline period",
        "Metric": "Baseline label",
        "Value": outpatient_baseline_label,
        "Definition / calculation": (
            "Uses outpatient contact records from Apr 2025 to Mar 2026 and "
            "annualises the observed baseline over the selected delivery weeks."
        ),
        "Why fixed": "This is the starting point before outpatient appointment states are applied.",
        "Actions": "Baseline period fixed to Apr 2025-Mar 2026 as requested.",
    },
    {
        "Input area": "Appointment baseline",
        "Metric": "Planned appointment records",
        "Value": format_number(outpatient_planned_appointments_horizon),
        "Definition / calculation": (
            "Unique Contact_ID records annualised over the delivery period."
        ),
        "Why fixed": (
            "This is the booked/contact baseline. Appointment states change the additional "
            "appointments unlocked, not the original baseline record count."
        ),
        "Actions": "Validate that Contact_ID represents booked/planned appointments and not only completed activity.",
    },
    {
        "Input area": "Appointment baseline",
        "Metric": "Actual attended appointments",
        "Value": format_number(outpatient_attended_appointments_horizon),
        "Definition / calculation": (
            "Unique Contact_ID records with Status containing Checked In or Checked Out."
        ),
        "Why fixed": "This is the actual attended baseline before appointment-state improvement.",
        "Actions": "Answered: Checked In and Checked Out count as attended; retain one attendance per Contact_ID.",
    },
    {
        "Input area": "Appointment baseline",
        "Metric": "DNA / no-show appointments",
        "Value": format_number(
            outpatient_baseline["dna_appointments"]
            / outpatient_baseline["observed_weeks"]
            * active_delivery_weeks
        ),
        "Definition / calculation": (
            "Unique Contact_ID records with Status containing No Show, annualised "
            "over the delivery period."
        ),
        "Why fixed": "This is the baseline DNA count before the DNA planning value is applied.",
        "Actions": "Confirm whether short-notice cancellations should be included in the DNA opportunity.",
    },
    {
        "Input area": "Clinic-session proxy",
        "Metric": "Booked/planned clinic-session context",
        "Value": format_number(outpatient_planned_sessions_horizon),
        "Definition / calculation": (
            "Proxy session = clinic/performance unit + date + AM/PM."
        ),
        "Why fixed": (
            "Shown only as context. Scenario modelling uses actual clinic-session "
            "proxies as the session base."
        ),
        "Actions": "Replace with Cerner clinic-template/session extract when available.",
    },
    {
        "Input area": "Clinic-session proxy",
        "Metric": "Actual clinic-session proxies",
        "Value": format_number(outpatient_actual_sessions_horizon),
        "Definition / calculation": (
            "Actual clinic-session proxy with at least one Checked In or Checked Out appointment."
        ),
        "Why fixed": "This is the delivered-session proxy used as the scenario session base.",
        "Actions": "Validate proxy against actual clinic-session reporting.",
    },
    {
        "Input area": "Rates",
        "Metric": "Booked appointment fill / attendance rate",
        "Value": f"{outpatient_model_current_fill_pct:.1f}%",
        "Definition / calculation": "Attended appointments / planned appointment records.",
        "Why fixed": "This is the observed starting rate before appointment-state improvement.",
        "Actions": "True template fill still needs a template-slot extract because empty slots may be absent.",
    },
    {
        "Input area": "Rates",
        "Metric": "DNA / no-show rate",
        "Value": f"{outpatient_model_current_dna_rate_pct:.1f}%",
        "Definition / calculation": "No Show appointments / planned appointment records.",
        "Why fixed": "This is the observed starting DNA rate before modelled DNA reduction.",
        "Actions": "Validate status mapping and cancellation treatment.",
    },
    {
        "Input area": "Activity mix",
        "Metric": "Observed follow-up appointments per week",
        "Value": format_number(outpatient_model_follow_up_per_week),
        "Definition / calculation": "Attended Contact_ID records mapped to Follow Up / observed weeks.",
        "Why fixed": "Used as the baseline eligible volume for the PIFU lever.",
        "Actions": "Confirm ContactVisitType mapping to follow-up activity.",
    },
    {
        "Input area": "Activity mix",
        "Metric": "Observed first appointments per week",
        "Value": format_number(outpatient_model_first_attendance_per_week),
        "Definition / calculation": "Attended Contact_ID records mapped to First attendance / observed weeks.",
        "Why fixed": "Used as the baseline eligible volume for F:N improvement.",
        "Actions": "Confirm ContactVisitType mapping to first-attendance activity.",
    },
]

outpatient_capacity_assumption_rows = [
    {
        "Capacity layer": "Estate",
        "Metric": "Estate capacity: rooms/templates",
        "Value": "Not available in current raw file",
        "Calculation / source": (
            "Current outpatient extract has clinic/performance-unit labels but "
            "no room count, template slots, empty slots, or full template capacity."
        ),
        "Why fixed": (
            "Cannot be modelled as true estate capacity until room/template data is supplied."
        ),
        "Actions": "Request Cerner clinic-template extract: rooms, planned sessions, template slots, booked slots, cancelled slots.",
    },
    {
        "Capacity layer": "Workforce",
        "Metric": "Substantive outpatient DCC activity capacity",
        "Value": format_number(workforce_outpatient_capacity_240_sessions),
        "Calculation / source": (
            f"`Out-patient activities` from {job_plan_capacity['source']} x "
            f"{format_decimal(active_delivery_weeks, 1)} weeks, excluding Locum rows."
        ),
        "Why fixed": (
            "This is planned workforce supply. It changes only if job plans, DCC "
            "assumptions, or delivery weeks change."
        ),
        "Actions": "Validate whether Out-patient activities maps to weekly 240-minute DCC sessions and agree specialty scope.",
    },
    {
        "Capacity layer": "Workforce",
        "Metric": "Workforce utilisation proxy",
        "Value": format_percent(workforce_outpatient_utilisation),
        "Calculation / source": (
            "Actual clinic-session proxies / substantive outpatient job-plan activity capacity."
        ),
        "Why fixed": (
            "This is a baseline diagnostic, not an appointment-state output."
        ),
        "Actions": "Treat as directional until clinic sessions and job-plan scope are reconciled.",
    },
]

outpatient_target_metric_order = [
    "Template fill target (proxy only - no template-slot data)",
    "DNA rate target",
    "PIFU conversion target",
    "F:N improvement target",
    "Current outpatient attendances / month",
    "Template fill additional appts / month",
    "DNA reduction additional appts / month",
    "PIFU additional appts / month",
    "F:N ratio additional appts / month",
    "Total additional appts / month",
    f"Total additional appts over {outpatient_horizon_months} months",
    f"Appointment activity after levers over {outpatient_horizon_months} months",
    f"RTT/PTL pathways reduced over {outpatient_horizon_months} months",
    "PTL after outpatient levers",
    "PTL reduction %",
    "Remaining PTL %",
    f"Indicative cost / capacity opportunity over {outpatient_horizon_months} months",
    f"Indicative income opportunity over {outpatient_horizon_months} months",
]
outpatient_target_rows = []
for metric in outpatient_target_metric_order:
    matched_rows = [row for row in outpatient_rows if row["Metric"] == metric]
    for row in matched_rows:
        display_row = {**row}
        display_row["Opportunity"] = "Outpatient appointment impact"
        outpatient_target_rows.append(display_row)

outpatient_more_throughput_rows = [
    {
        "Scenario lens": "What if more throughput",
        "Metric": "Definition",
        "Current baseline": "Clinic sessions broadly fixed",
        "What if 50% delivery": "More patients seen",
        "What if 75% delivery": "More patients seen",
        "What if 100% delivery": "More patients seen",
        "Notes / calculation": (
            "Keeps clinic-session capacity broadly the same and converts better "
            "use of capacity into additional appointments."
        ),
    },
    {
        "Scenario lens": "What if more throughput",
        "Metric": "Additional appointments / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "additional_monthly",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "additional_monthly",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "additional_monthly",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Template proxy + DNA + PIFU + F:N additional appointments.",
    },
    {
        "Scenario lens": "What if more throughput",
        "Metric": f"Additional appointments over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "additional_total",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "additional_total",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "additional_total",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Additional appointments / month x horizon months.",
    },
    {
        "Scenario lens": "What if more throughput",
        "Metric": f"RTT/PTL pathways reduced over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Additional appointments x RTT/PTL conversion assumption.",
    },
    {
        "Scenario lens": "What if more throughput",
        "Metric": "Cost / capacity opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "financial_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "financial_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "financial_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional appointments x cost/capacity value per appointment."
        ),
    },
    {
        "Scenario lens": "What if more throughput",
        "Metric": "Income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "income_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "income_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "income_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": (
            "Additional appointments x observed outpatient income per attended contact."
        ),
    },
]

outpatient_baseline_sessions_for_horizon = outpatient_actual_sessions_horizon
outpatient_baseline_activity_for_horizon = outpatient_baseline_total
outpatient_baseline_productivity = (
    outpatient_baseline_activity_for_horizon / outpatient_baseline_sessions_for_horizon
    if outpatient_baseline_sessions_for_horizon > 0
    else 0
)
outpatient_efficiency_outputs = {}

for scenario, output in outpatient_outputs.items():
    target_activity = outpatient_baseline_activity_for_horizon + output["additional_total"]
    target_productivity = (
        target_activity / outpatient_baseline_sessions_for_horizon
        if outpatient_baseline_sessions_for_horizon > 0
        else 0
    )
    required_sessions = (
        outpatient_baseline_activity_for_horizon / target_productivity
        if target_productivity > 0
        else 0
    )
    sessions_freed = max(outpatient_baseline_sessions_for_horizon - required_sessions, 0)
    released_appointment_capacity = sessions_freed * target_productivity
    outpatient_efficiency_outputs[scenario] = {
        "target_productivity": target_productivity,
        "required_sessions": required_sessions,
        "sessions_freed": sessions_freed,
        "released_appointment_capacity": released_appointment_capacity,
        "capacity_value": released_appointment_capacity * outpatient_value_per_appointment,
    }

outpatient_same_throughput_rows = [
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Definition",
        "Current baseline": "Activity fixed",
        "What if 50% delivery": "Fewer sessions needed",
        "What if 75% delivery": "Fewer sessions needed",
        "What if 100% delivery": "Fewer sessions needed",
        "Notes / calculation": (
            "Keeps the same number of attended appointments and estimates how "
            "many fewer proxy clinic sessions are required if productivity improves."
        ),
    },
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Baseline attended appointments",
        "Current baseline": format_number(outpatient_baseline_activity_for_horizon),
        "What if 50% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "What if 75% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "What if 100% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "Notes / calculation": "Activity is deliberately held constant in this lens.",
    },
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Appointments per proxy clinic session",
        "Current baseline": format_decimal(outpatient_baseline_productivity, 2),
        "What if 50% delivery": format_decimal(
            outpatient_efficiency_outputs["What if 50% delivery"]["target_productivity"],
            2,
        ),
        "What if 75% delivery": format_decimal(
            outpatient_efficiency_outputs["What if 75% delivery"]["target_productivity"],
            2,
        ),
        "What if 100% delivery": format_decimal(
            outpatient_efficiency_outputs["What if 100% delivery"]["target_productivity"],
            2,
        ),
        "Notes / calculation": (
            "Target productivity = baseline activity plus modelled additional "
            "appointments, divided by fixed proxy clinic sessions."
        ),
    },
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Proxy clinic sessions required",
        "Current baseline": format_number(outpatient_baseline_sessions_for_horizon),
        "What if 50% delivery": format_number(
            outpatient_efficiency_outputs["What if 50% delivery"]["required_sessions"]
        ),
        "What if 75% delivery": format_number(
            outpatient_efficiency_outputs["What if 75% delivery"]["required_sessions"]
        ),
        "What if 100% delivery": format_number(
            outpatient_efficiency_outputs["What if 100% delivery"]["required_sessions"]
        ),
        "Notes / calculation": "Baseline attended appointments / appointment-state productivity.",
    },
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Proxy clinic sessions freed",
        "Current baseline": "0",
        "What if 50% delivery": format_number(
            outpatient_efficiency_outputs["What if 50% delivery"]["sessions_freed"]
        ),
        "What if 75% delivery": format_number(
            outpatient_efficiency_outputs["What if 75% delivery"]["sessions_freed"]
        ),
        "What if 100% delivery": format_number(
            outpatient_efficiency_outputs["What if 100% delivery"]["sessions_freed"]
        ),
        "Notes / calculation": "Baseline proxy clinic sessions - required proxy clinic sessions.",
    },
    {
        "Scenario lens": "What if same throughput, fewer sessions",
        "Metric": "Capacity value released",
        "Current baseline": format_currency(0),
        "What if 50% delivery": format_currency(
            outpatient_efficiency_outputs["What if 50% delivery"]["capacity_value"]
        ),
        "What if 75% delivery": format_currency(
            outpatient_efficiency_outputs["What if 75% delivery"]["capacity_value"]
        ),
        "What if 100% delivery": format_currency(
            outpatient_efficiency_outputs["What if 100% delivery"]["capacity_value"]
        ),
        "Notes / calculation": (
            "Released appointment-equivalent capacity x cost/capacity value per appointment. "
            "No income uplift is assumed because activity is held constant."
        ),
    },
]

outpatient_rtt_opportunity_rows = [
    {
        "RTT opportunity": "Backlog reduction",
        "Metric": f"Additional appointments over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "additional_total",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "additional_total",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "additional_total",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Total additional appointment capacity from the more-throughput lens.",
    },
    {
        "RTT opportunity": "Backlog reduction",
        "Metric": "RTT/PTL conversion assumption",
        "Current baseline": f"{outpatient_rtt_conversion_pct:.0f}%",
        "What if 50% delivery": f"{outpatient_rtt_conversion_pct:.0f}%",
        "What if 75% delivery": f"{outpatient_rtt_conversion_pct:.0f}%",
        "What if 100% delivery": f"{outpatient_rtt_conversion_pct:.0f}%",
        "Notes / calculation": "Sidebar assumption applied to additional appointments.",
    },
    {
        "RTT opportunity": "Backlog reduction",
        "Metric": f"RTT/PTL pathways reduced over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "converted_ptl_impact",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Additional appointments x RTT/PTL conversion assumption.",
    },
    {
        "RTT opportunity": "Backlog reduction",
        "Metric": "PTL after outpatient levers",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": outpatient_value("remaining_ptl", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("remaining_ptl", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("remaining_ptl", "What if 100% delivery"),
        "Notes / calculation": "Current PTL - converted RTT/PTL pathway reduction.",
    },
    {
        "RTT opportunity": "Finance",
        "Metric": "Cost / capacity opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "financial_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "financial_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "financial_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Additional appointments x cost/capacity value per appointment.",
    },
    {
        "RTT opportunity": "Finance",
        "Metric": "Income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value(
            "income_opportunity",
            "What if 50% delivery",
        ),
        "What if 75% delivery": outpatient_value(
            "income_opportunity",
            "What if 75% delivery",
        ),
        "What if 100% delivery": outpatient_value(
            "income_opportunity",
            "What if 100% delivery",
        ),
        "Notes / calculation": "Additional appointments x observed income per attended contact.",
    },
]

outpatient_scenario_rows = [
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": (
            "Baseline outpatient activity is fixed; planning improvement across "
            "template fill, DNA, PIFU and F:N creates additional appointments."
        ),
        "Metric": "Target share delivered",
        "Current baseline": "0%",
        "What if 50% delivery": "50%",
        "What if 75% delivery": "75%",
        "What if 100% delivery": "100%",
        "Notes / calculation": "Appointment states 1, 2 and 3 scale linearly to the full planning state.",
        "Actions": "Agree target ownership before using as committed delivery.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Four levers combined into one outpatient impact view.",
        "Metric": "Total additional appointments / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("additional_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("additional_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("additional_monthly", "What if 100% delivery"),
        "Notes / calculation": "Template fill + DNA reduction + PIFU + F:N ratio improvement.",
        "Actions": "Use as the main operational outpatient capacity-impact row.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Four levers combined into one outpatient impact view.",
        "Metric": f"Total additional appointments over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("additional_total", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("additional_total", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("additional_total", "What if 100% delivery"),
        "Notes / calculation": "Monthly additional appointments x outpatient horizon.",
        "Actions": "Use for planning scale; validate deliverability by specialty.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Additional appointments are converted to RTT/PTL impact using the selected conversion assumption.",
        "Metric": f"RTT/PTL pathways reduced over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("converted_ptl_impact", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("converted_ptl_impact", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("converted_ptl_impact", "What if 100% delivery"),
        "Notes / calculation": "Total additional appointments x outpatient RTT/PTL conversion percentage.",
        "Actions": "Validate whether additional appointments remove RTT pathways one-for-one.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Additional appointments are converted to RTT/PTL impact using the selected conversion assumption.",
        "Metric": "PTL after outpatient levers",
        "Current baseline": format_number(current_ptl),
        "What if 50% delivery": outpatient_value("remaining_ptl", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("remaining_ptl", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("remaining_ptl", "What if 100% delivery"),
        "Notes / calculation": "Current PTL - converted RTT/PTL pathway reduction.",
        "Actions": "Use as indicative until RTT conversion is validated.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Additional appointments are converted to RTT/PTL impact using the selected conversion assumption.",
        "Metric": "PTL reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": outpatient_value("ptl_reduction_pct", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("ptl_reduction_pct", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("ptl_reduction_pct", "What if 100% delivery"),
        "Notes / calculation": "Converted RTT/PTL pathway reduction / current PTL.",
        "Actions": "Read alongside remaining PTL percentage.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Additional appointments are converted to RTT/PTL impact using the selected conversion assumption.",
        "Metric": "Remaining PTL %",
        "Current baseline": "100.0%",
        "What if 50% delivery": outpatient_value("remaining_ptl_pct", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("remaining_ptl_pct", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("remaining_ptl_pct", "What if 100% delivery"),
        "Notes / calculation": "PTL after outpatient levers / current PTL.",
        "Actions": "Read alongside PTL after outpatient levers.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Financial view values additional appointments; it is not automatically cashable.",
        "Metric": "Indicative cost / capacity opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value("financial_opportunity", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("financial_opportunity", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("financial_opportunity", "What if 100% delivery"),
        "Notes / calculation": f"Additional appointments over the horizon x {format_currency(outpatient_value_per_appointment)}.",
        "Actions": "Finance should confirm whether this is cashable, cost avoidance, or capacity value.",
    },
    {
        "Scenario": "Outpatient appointment-state delivery",
        "Scenario interpretation": "Income view values additional activity; it is not automatically incremental income.",
        "Metric": "Indicative income opportunity",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value("income_opportunity", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("income_opportunity", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("income_opportunity", "What if 100% delivery"),
        "Notes / calculation": f"Additional appointments over the horizon x {format_currency(outpatient_income_per_attendance)} observed outpatient income per attended contact.",
        "Actions": "Finance should confirm tariff/payment treatment and whether this activity creates incremental income.",
    },
]

rtt_opening_backlog = {
    "0_18": latest_rtt_wait_bands["waiting_0_18"],
    "18_51": latest_rtt_wait_bands["waiting_18_51"],
    "52_plus": latest_rtt_wait_bands["waiting_52_plus"],
    "total": latest_rtt_wait_bands["total"],
}
rtt_lever_order = [
    (
        "DNA reduction",
        "dna_reduction_total",
        "Observed first + follow-up appointments x DNA rate improvement.",
    ),
    (
        "PIFU",
        "pifu_total",
        "Observed attended follow-up appointments x PIFU conversion assumption.",
    ),
    (
        "F:N improvement",
        "fn_ratio_total",
        "Observed attended first appointments x F:N improvement assumption.",
    ),
    (
        "Template fill proxy",
        "template_fill_total",
        "Template-capacity proxy. Kept last because true template-slot data is not available.",
    ),
]
outpatient_rtt_lever_bridge_rows = []

for scenario, output in outpatient_outputs.items():
    state_label = outpatient_display_scenario_labels[scenario].split(" (")[0]
    running_backlog = rtt_opening_backlog.copy()

    outpatient_rtt_lever_bridge_rows.append(
        {
            "Appointment state": state_label,
            "Step": "Opening",
            "Lever": "Current RTT backlog",
            "Appointment capacity over horizon": "-",
            "RTT conversion applied": "-",
            "Converted RTT capacity": "-",
            "Applied RTT reduction": "-",
            "Unused converted capacity": "-",
            "Closing 52+ weeks": format_number(running_backlog["52_plus"]),
            "Closing 18-51 weeks": format_number(running_backlog["18_51"]),
            "Closing 0-18 weeks": format_number(running_backlog["0_18"]),
            "Closing total backlog": format_number(running_backlog["total"]),
            "Closing % within 18 weeks": format_percent(
                running_backlog["0_18"] / running_backlog["total"]
                if running_backlog["total"] > 0
                else 0
            ),
            "Calculation / evidence": (
                f"Latest RTT incomplete-pathway extract: {latest_rtt_wait_bands['month']} "
                f"from {latest_rtt_wait_bands['source']}."
            ),
        }
    )

    for step_number, (lever_name, output_key, evidence) in enumerate(
        rtt_lever_order,
        start=1,
    ):
        appointment_capacity = output[output_key]
        converted_capacity = appointment_capacity * outpatient_rtt_conversion_pct / 100
        allocation = allocate_rtt_backlog_reduction(
            running_backlog,
            converted_capacity,
        )
        running_backlog = allocation["closing"]

        outpatient_rtt_lever_bridge_rows.append(
            {
                "Appointment state": state_label,
                "Step": str(step_number),
                "Lever": lever_name,
                "Appointment capacity over horizon": format_number(appointment_capacity),
                "RTT conversion applied": f"{outpatient_rtt_conversion_pct:.0f}%",
                "Converted RTT capacity": format_number(converted_capacity),
                "Applied RTT reduction": format_number(allocation["applied_reduction"]),
                "Unused converted capacity": format_number(allocation["unused_capacity"]),
                "Closing 52+ weeks": format_number(
                    allocation["closing"]["52_plus"]
                ),
                "Closing 18-51 weeks": format_number(
                    allocation["closing"]["18_51"]
                ),
                "Closing 0-18 weeks": format_number(allocation["closing"]["0_18"]),
                "Closing total backlog": format_number(allocation["closing"]["total"]),
                "Closing % within 18 weeks": format_percent(
                    allocation["closing"]["0_18"] / allocation["closing"]["total"]
                    if allocation["closing"]["total"] > 0
                    else 0
                ),
                "Calculation / evidence": (
                    f"{evidence} Converted capacity is applied longest-waits-first: "
                    "52+ weeks, then 18-51 weeks, then 0-18 weeks."
                ),
            }
        )

outpatient_rtt_simple_impact = {}
for scenario, output in outpatient_outputs.items():
    remaining_backlog = rtt_opening_backlog["total"]
    lever_impacts = {}

    for lever_name, output_key, _ in rtt_lever_order:
        appointment_capacity = output[output_key]
        converted_capacity = appointment_capacity * outpatient_rtt_conversion_pct / 100
        applied_reduction = min(converted_capacity, remaining_backlog)
        remaining_backlog -= applied_reduction
        lever_impacts[lever_name] = {
            "appointment_capacity": appointment_capacity,
            "converted_capacity": converted_capacity,
            "applied_reduction": applied_reduction,
            "closing_backlog_after_lever": remaining_backlog,
        }

    total_reduction = sum(
        item["applied_reduction"] for item in lever_impacts.values()
    )
    outpatient_rtt_simple_impact[scenario] = {
        "lever_impacts": lever_impacts,
        "total_reduction": total_reduction,
        "closing_backlog": remaining_backlog,
        "backlog_reduction_pct": (
            total_reduction / rtt_opening_backlog["total"]
            if rtt_opening_backlog["total"] > 0
            else 0
        ),
        "remaining_backlog_pct": (
            remaining_backlog / rtt_opening_backlog["total"]
            if rtt_opening_backlog["total"] > 0
            else 0
        ),
    }


def outpatient_rtt_lever_value(scenario: str, lever: str) -> str:
    return format_number(
        outpatient_rtt_simple_impact[scenario]["lever_impacts"][lever][
            "applied_reduction"
        ]
    )


def outpatient_rtt_total_value(scenario: str, key: str) -> str:
    value = outpatient_rtt_simple_impact[scenario][key]
    if key.endswith("_pct"):
        return format_percent(value)
    return format_number(value)


outpatient_impact_rows = [
    {
        "Impact bucket": "Baseline context",
        "Metric": "Baseline period",
        "Current baseline": outpatient_baseline_label,
        "What if 50% delivery": "Same baseline",
        "What if 75% delivery": "Same baseline",
        "What if 100% delivery": "Same baseline",
        "Calculation / evidence": (
            "All outpatient baseline metrics are calculated from raw outpatient "
            "records dated Apr 2025 to Mar 2026. Scenario columns do not change "
            "the baseline period; they only apply the modelled appointment uplift."
        ),
        "Interpretation / action": "Use this as the fixed starting point for the outpatient impact view.",
    },
    {
        "Impact bucket": "Baseline context",
        "Metric": "Planned appointment records",
        "Current baseline": format_number(outpatient_baseline["planned_appointments"]),
        "What if 50% delivery": format_number(outpatient_baseline["planned_appointments"]),
        "What if 75% delivery": format_number(outpatient_baseline["planned_appointments"]),
        "What if 100% delivery": format_number(outpatient_baseline["planned_appointments"]),
        "Calculation / evidence": (
            "Count of unique Contact_ID records in Apr 2025-Mar 2026. This is a "
            "booked/planned appointment-record proxy, not full template capacity, "
            "because empty template slots are not present in the outpatient file."
        ),
        "Interpretation / action": "This stays fixed because the what-if states add capacity above baseline.",
    },
    {
        "Impact bucket": "Baseline context",
        "Metric": "Actual attended appointments",
        "Current baseline": format_number(outpatient_baseline["attended_appointments"]),
        "What if 50% delivery": format_number(outpatient_baseline["attended_appointments"]),
        "What if 75% delivery": format_number(outpatient_baseline["attended_appointments"]),
        "What if 100% delivery": format_number(outpatient_baseline["attended_appointments"]),
        "Calculation / evidence": (
            "Count of unique Contact_ID records where Status contains Checked In "
            "or Checked Out. Each Contact_ID is counted once."
        ),
        "Interpretation / action": "This is the observed attended baseline before any additional appointment opportunity.",
    },
    {
        "Impact bucket": "Baseline context",
        "Metric": "Booked appointment fill / attendance proxy",
        "Current baseline": f"{outpatient_model_current_fill_pct:.1f}%",
        "What if 50% delivery": outpatient_value("template_fill_target", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("template_fill_target", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("template_fill_target", "What if 100% delivery"),
        "Calculation / evidence": (
            f"{format_number(outpatient_baseline['attended_appointments'])} attended contacts / "
            f"{format_number(outpatient_baseline['planned_appointments'])} planned Contact_ID records. "
            "This is only an attendance/fill proxy because the extract does not include empty template slots."
        ),
        "Interpretation / action": "Do not describe this as true template fill until template-slot data is supplied.",
    },
    {
        "Impact bucket": "Baseline context",
        "Metric": "DNA / no-show rate",
        "Current baseline": f"{outpatient_model_current_dna_rate_pct:.1f}%",
        "What if 50% delivery": outpatient_value("dna_rate_target", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("dna_rate_target", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("dna_rate_target", "What if 100% delivery"),
        "Calculation / evidence": (
            f"{format_number(outpatient_baseline['dna_appointments'])} No Show Contact_ID records / "
            f"{format_number(outpatient_baseline['planned_appointments'])} planned Contact_ID records. "
            "Scenario values move from the observed DNA rate toward the selected DNA planning value."
        ),
        "Interpretation / action": "Confirm whether short-notice cancellations should also be counted in the opportunity.",
    },
    {
        "Impact bucket": "Baseline context",
        "Metric": "Actual clinic sessions used as scenario base",
        "Current baseline": format_number(outpatient_actual_sessions_horizon),
        "What if 50% delivery": format_number(outpatient_actual_sessions_horizon),
        "What if 75% delivery": format_number(outpatient_actual_sessions_horizon),
        "What if 100% delivery": format_number(outpatient_actual_sessions_horizon),
        "Calculation / evidence": (
            "Actual session proxy = clinic/performance unit + appointment date + "
            "AM/PM with at least one Checked In or Checked Out appointment, "
            "converted to the selected delivery weeks."
        ),
        "Interpretation / action": "This is the session metric used by the scenario logic; planned sessions are context only.",
    },
    {
        "Impact bucket": "Capacity / data gap",
        "Metric": "Estate capacity: rooms/templates",
        "Current baseline": "Not available in current raw file",
        "What if 50% delivery": "Not modelled",
        "What if 75% delivery": "Not modelled",
        "What if 100% delivery": "Not modelled",
        "Calculation / evidence": (
            "Current outpatient extract has clinic/performance-unit labels, but "
            "does not contain room counts, full clinic templates, empty slots, "
            "booked-slot capacity, or cancelled template slots."
        ),
        "Interpretation / action": "Request Cerner clinic-template/room-capacity data before modelling estate capacity.",
    },
    {
        "Impact bucket": "Capacity / data gap",
        "Metric": "Substantive outpatient DCC capacity",
        "Current baseline": format_number(workforce_outpatient_capacity_240_sessions),
        "What if 50% delivery": "Diagnostic only",
        "What if 75% delivery": "Diagnostic only",
        "What if 100% delivery": "Diagnostic only",
        "Calculation / evidence": (
            f"`Out-patient activities` from {job_plan_capacity['source']} x "
            f"{format_decimal(active_delivery_weeks, 1)} delivery weeks, excluding Locum rows."
        ),
        "Interpretation / action": "Validate that Out-patient activities maps to weekly clinic DCC sessions.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "Actual clinic sessions held constant",
        "Current baseline": format_number(outpatient_actual_sessions_horizon),
        "What if 50% delivery": format_number(outpatient_actual_sessions_horizon),
        "What if 75% delivery": format_number(outpatient_actual_sessions_horizon),
        "What if 100% delivery": format_number(outpatient_actual_sessions_horizon),
        "Calculation / evidence": (
            "More-throughput lens uses actual clinic-session proxies as the "
            "fixed session base."
        ),
        "Interpretation / action": "Sessions stay fixed; additional appointment volume changes.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "Appointment opportunity dial",
        "Current baseline": "0%",
        "What if 50% delivery": "50% of modelled appointment opportunity",
        "What if 75% delivery": "75% of modelled appointment opportunity",
        "What if 100% delivery": "100% of modelled appointment opportunity",
        "Calculation / evidence": (
            "The three columns scale the same four outpatient levers: template-fill "
            "proxy, DNA reduction, PIFU conversion and F:N improvement. They are "
            "appointment uplift states, not separate baselines."
        ),
        "Interpretation / action": "Use these columns to show what happens as the outpatient dial is moved.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "Template fill - additional appointment volume / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("template_fill_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("template_fill_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("template_fill_monthly", "What if 100% delivery"),
        "Calculation / evidence": (
            "Clinic sessions/week x patients/session x RTT-relevant template share "
            "x max(template planning value - current fill proxy, 0) x delivery weeks "
            "x appointment opportunity dial / horizon months. If the current proxy "
            "is already above the planning value, this row is zero."
        ),
        "Interpretation / action": "Proxy only; true template-fill uplift needs template-slot data.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "DNA reduction - additional appointment volume / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("dna_reduction_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("dna_reduction_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("dna_reduction_monthly", "What if 100% delivery"),
        "Calculation / evidence": (
            "(Observed first appointments/week + observed follow-up appointments/week) "
            "x max(current DNA rate - DNA planning value, 0) x delivery weeks "
            "x appointment opportunity dial / horizon months."
        ),
        "Interpretation / action": "This estimates extra attendances released by reducing no-shows.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "PIFU - additional appointment volume / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("pifu_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("pifu_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("pifu_monthly", "What if 100% delivery"),
        "Calculation / evidence": (
            "Observed attended follow-up appointments/week x PIFU conversion % "
            "x delivery weeks x appointment opportunity dial / horizon months."
        ),
        "Interpretation / action": "This treats eligible follow-up activity as capacity that can be released or redirected.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "F:N ratio - additional appointment volume / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("fn_ratio_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("fn_ratio_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("fn_ratio_monthly", "What if 100% delivery"),
        "Calculation / evidence": (
            "Observed attended first appointments/week x F:N improvement % "
            "x delivery weeks x appointment opportunity dial / horizon months."
        ),
        "Interpretation / action": "This models new-patient capacity released through F:N improvement.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "Total additional appointment volume / month",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("additional_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("additional_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("additional_monthly", "What if 100% delivery"),
        "Calculation / evidence": "Template fill uplift + DNA reduction uplift + PIFU uplift + F:N uplift.",
        "Interpretation / action": "Use this as the headline operational throughput row.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": f"Total additional appointment volume over {outpatient_horizon_months} months",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_value("additional_total", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("additional_total", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("additional_total", "What if 100% delivery"),
        "Calculation / evidence": "Total additional appointment volume/month x selected outpatient horizon months.",
        "Interpretation / action": "Use this for planning the total capacity impact over the period.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": "Appointment volume / month after levers",
        "Current baseline": format_number(outpatient_baseline_monthly),
        "What if 50% delivery": outpatient_value("activity_after_monthly", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("activity_after_monthly", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("activity_after_monthly", "What if 100% delivery"),
        "Calculation / evidence": (
            f"Baseline monthly attendances = {format_number(outpatient_baseline['attended_appointments'])} "
            f"attended contacts divided across {outpatient_month_count} months. "
            "Scenario values add the modelled additional appointment volume/month."
        ),
        "Interpretation / action": "Shows throughput movement using actual sessions as the fixed base.",
    },
    {
        "Impact bucket": "What if more throughput",
        "Metric": f"Appointment volume after levers over {outpatient_horizon_months} months",
        "Current baseline": format_number(outpatient_baseline_total),
        "What if 50% delivery": outpatient_value("activity_after_total", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("activity_after_total", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("activity_after_total", "What if 100% delivery"),
        "Calculation / evidence": (
            "Baseline monthly attendances x horizon months + total additional "
            "appointments over the horizon."
        ),
        "Interpretation / action": "Shows the full activity view after the outpatient dial is applied.",
    },
    {
        "Impact bucket": "What if same throughput, fewer sessions",
        "Metric": "Actual clinic sessions baseline",
        "Current baseline": format_number(outpatient_baseline_sessions_for_horizon),
        "What if 50% delivery": format_number(outpatient_baseline_sessions_for_horizon),
        "What if 75% delivery": format_number(outpatient_baseline_sessions_for_horizon),
        "What if 100% delivery": format_number(outpatient_baseline_sessions_for_horizon),
        "Calculation / evidence": (
            "Baseline actual clinic-session proxies. The fewer-sessions lens "
            "then calculates how many of these are required at improved productivity."
        ),
        "Interpretation / action": "Actual sessions are the starting session metric.",
    },
    {
        "Impact bucket": "What if same throughput, fewer sessions",
        "Metric": "Attended appointment volume held constant",
        "Current baseline": format_number(outpatient_baseline_activity_for_horizon),
        "What if 50% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "What if 75% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "What if 100% delivery": format_number(outpatient_baseline_activity_for_horizon),
        "Calculation / evidence": (
            "This lens keeps the same attended appointment volume and asks how "
            "many fewer clinic-session proxies would be needed if productivity improves."
        ),
        "Interpretation / action": "No RTT or income uplift is assumed here because appointment volume is held constant.",
    },
    {
        "Impact bucket": "What if same throughput, fewer sessions",
        "Metric": "Appointments per proxy clinic session",
        "Current baseline": format_decimal(outpatient_baseline_productivity, 2),
        "What if 50% delivery": format_decimal(outpatient_efficiency_outputs["What if 50% delivery"]["target_productivity"], 2),
        "What if 75% delivery": format_decimal(outpatient_efficiency_outputs["What if 75% delivery"]["target_productivity"], 2),
        "What if 100% delivery": format_decimal(outpatient_efficiency_outputs["What if 100% delivery"]["target_productivity"], 2),
        "Calculation / evidence": (
            "Baseline activity plus modelled additional appointment capacity, "
            "divided by fixed actual clinic-session proxies."
        ),
        "Interpretation / action": "Higher productivity means fewer sessions are needed to deliver the same baseline activity.",
    },
    {
        "Impact bucket": "What if same throughput, fewer sessions",
        "Metric": "Proxy clinic sessions required",
        "Current baseline": format_number(outpatient_baseline_sessions_for_horizon),
        "What if 50% delivery": format_number(outpatient_efficiency_outputs["What if 50% delivery"]["required_sessions"]),
        "What if 75% delivery": format_number(outpatient_efficiency_outputs["What if 75% delivery"]["required_sessions"]),
        "What if 100% delivery": format_number(outpatient_efficiency_outputs["What if 100% delivery"]["required_sessions"]),
        "Calculation / evidence": "Baseline attended appointments / productivity under each appointment state.",
        "Interpretation / action": "This is the fewer-sessions output, not the more-throughput output.",
    },
    {
        "Impact bucket": "What if same throughput, fewer sessions",
        "Metric": "Proxy clinic sessions freed",
        "Current baseline": "0",
        "What if 50% delivery": format_number(outpatient_efficiency_outputs["What if 50% delivery"]["sessions_freed"]),
        "What if 75% delivery": format_number(outpatient_efficiency_outputs["What if 75% delivery"]["sessions_freed"]),
        "What if 100% delivery": format_number(outpatient_efficiency_outputs["What if 100% delivery"]["sessions_freed"]),
        "Calculation / evidence": "Baseline actual clinic-session proxies - proxy clinic sessions required.",
        "Interpretation / action": "This is the capacity-release view if the same activity can be delivered more efficiently.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "Opening RTT backlog",
        "Current baseline": format_number(rtt_opening_backlog["total"]),
        "What if 50% delivery": format_number(rtt_opening_backlog["total"]),
        "What if 75% delivery": format_number(rtt_opening_backlog["total"]),
        "What if 100% delivery": format_number(rtt_opening_backlog["total"]),
        "Calculation / evidence": (
            f"Latest RTT incomplete-pathway extract: {latest_rtt_wait_bands['month']} "
            f"from {latest_rtt_wait_bands['source']}. Uses total incomplete pathways only."
        ),
        "Interpretation / action": "This is the backlog baseline before outpatient lever reductions.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "DNA reduction - backlog impact",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_rtt_lever_value("What if 50% delivery", "DNA reduction"),
        "What if 75% delivery": outpatient_rtt_lever_value("What if 75% delivery", "DNA reduction"),
        "What if 100% delivery": outpatient_rtt_lever_value("What if 100% delivery", "DNA reduction"),
        "Calculation / evidence": (
            "Calculated amount = DNA additional appointment volume over the horizon x outpatient RTT/PTL "
            f"conversion ({outpatient_rtt_conversion_pct:.0f}%), capped at remaining backlog."
        ),
        "Interpretation / action": "This is the calculated backlog reduction from this lever only.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "PIFU - backlog impact",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_rtt_lever_value("What if 50% delivery", "PIFU"),
        "What if 75% delivery": outpatient_rtt_lever_value("What if 75% delivery", "PIFU"),
        "What if 100% delivery": outpatient_rtt_lever_value("What if 100% delivery", "PIFU"),
        "Calculation / evidence": (
            "Calculated amount = PIFU additional appointment volume over the horizon x outpatient RTT/PTL "
            f"conversion ({outpatient_rtt_conversion_pct:.0f}%), capped at remaining backlog."
        ),
        "Interpretation / action": "This is the calculated backlog reduction from this lever only.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "F:N improvement - backlog impact",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_rtt_lever_value("What if 50% delivery", "F:N improvement"),
        "What if 75% delivery": outpatient_rtt_lever_value("What if 75% delivery", "F:N improvement"),
        "What if 100% delivery": outpatient_rtt_lever_value("What if 100% delivery", "F:N improvement"),
        "Calculation / evidence": (
            "Calculated amount = F:N additional appointment volume over the horizon x outpatient RTT/PTL "
            f"conversion ({outpatient_rtt_conversion_pct:.0f}%), capped at remaining backlog."
        ),
        "Interpretation / action": "This is the calculated backlog reduction from this lever only.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "Template fill proxy - backlog impact",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_rtt_lever_value("What if 50% delivery", "Template fill proxy"),
        "What if 75% delivery": outpatient_rtt_lever_value("What if 75% delivery", "Template fill proxy"),
        "What if 100% delivery": outpatient_rtt_lever_value("What if 100% delivery", "Template fill proxy"),
        "Calculation / evidence": (
            "Calculated amount = template-fill proxy appointments over the horizon x outpatient RTT/PTL "
            f"conversion ({outpatient_rtt_conversion_pct:.0f}%), capped at remaining backlog."
        ),
        "Interpretation / action": "This is the calculated backlog reduction from this lever only. Proxy only because true outpatient template-fill data is not available.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "Total backlog impact from outpatient levers",
        "Current baseline": "0",
        "What if 50% delivery": outpatient_rtt_total_value("What if 50% delivery", "total_reduction"),
        "What if 75% delivery": outpatient_rtt_total_value("What if 75% delivery", "total_reduction"),
        "What if 100% delivery": outpatient_rtt_total_value("What if 100% delivery", "total_reduction"),
        "Calculation / evidence": "Sum of applied backlog reductions from DNA, PIFU, F:N and template-fill proxy, capped at opening backlog.",
        "Interpretation / action": "This is the calculated total backlog reduction across the outpatient levers.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "Final RTT backlog after outpatient levers",
        "Current baseline": format_number(rtt_opening_backlog["total"]),
        "What if 50% delivery": outpatient_rtt_total_value("What if 50% delivery", "closing_backlog"),
        "What if 75% delivery": outpatient_rtt_total_value("What if 75% delivery", "closing_backlog"),
        "What if 100% delivery": outpatient_rtt_total_value("What if 100% delivery", "closing_backlog"),
        "Calculation / evidence": "Opening RTT backlog - total applied backlog reduction.",
        "Interpretation / action": "This is the actual closing backlog position after all outpatient levers.",
    },
    {
        "Impact bucket": "RTT / backlog opportunity",
        "Metric": "Backlog reduction %",
        "Current baseline": "0.0%",
        "What if 50% delivery": outpatient_rtt_total_value("What if 50% delivery", "backlog_reduction_pct"),
        "What if 75% delivery": outpatient_rtt_total_value("What if 75% delivery", "backlog_reduction_pct"),
        "What if 100% delivery": outpatient_rtt_total_value("What if 100% delivery", "backlog_reduction_pct"),
        "Calculation / evidence": "Total applied backlog reduction / opening RTT backlog.",
        "Interpretation / action": "Use with closing RTT backlog so the percentage has a clear denominator.",
    },
    {
        "Impact bucket": "Financial opportunity",
        "Metric": f"Cost / capacity opportunity over {outpatient_horizon_months} months",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value("financial_opportunity", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("financial_opportunity", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("financial_opportunity", "What if 100% delivery"),
        "Calculation / evidence": (
            f"Total additional appointment volume over the horizon x {format_currency(outpatient_value_per_appointment)} "
            "cost/capacity value per extra appointment."
        ),
        "Interpretation / action": "Opportunity proxy only; Finance should confirm cashability and overlap with other benefits.",
    },
    {
        "Impact bucket": "Financial opportunity",
        "Metric": f"Income opportunity over {outpatient_horizon_months} months",
        "Current baseline": format_currency(0),
        "What if 50% delivery": outpatient_value("income_opportunity", "What if 50% delivery"),
        "What if 75% delivery": outpatient_value("income_opportunity", "What if 75% delivery"),
        "What if 100% delivery": outpatient_value("income_opportunity", "What if 100% delivery"),
        "Calculation / evidence": (
            f"Total additional appointment volume over the horizon x {format_currency(outpatient_income_per_attendance)} "
            f"observed outpatient income per attended contact. {outpatient_finance_evidence}"
        ),
        "Interpretation / action": "Gross income lens only; Finance should confirm tariff/payment treatment.",
    },
]

outpatient_metric_coverage_df = pd.DataFrame(
    [
        {"Requested metric": "Planned appointment records", "Shown in": "Outpatient Baseline Inputs", "Status": "Included"},
        {"Requested metric": "Actual attended appointments", "Shown in": "Outpatient Baseline Inputs", "Status": "Included"},
        {"Requested metric": "Booked appointment fill / attendance rate", "Shown in": "Outpatient Impact Table", "Status": "Included as proxy"},
        {"Requested metric": "DNA / no-show rate", "Shown in": "Outpatient Impact Table", "Status": "Included"},
        {"Requested metric": "Actual clinic-session proxies", "Shown in": "Outpatient Baseline Inputs and More Throughput tab", "Status": "Included as scenario session base"},
        {"Requested metric": "Estate capacity: rooms/templates", "Shown in": "Outpatient Capacity Assumptions", "Status": "Partially included"},
        {"Requested metric": "Substantive outpatient DCC from job plans", "Shown in": "Outpatient Capacity Assumptions", "Status": "Included"},
        {"Requested metric": "Template fill additional appointment volume", "Shown in": "Outpatient Impact Table - More Throughput", "Status": "Proxy only - no measured template-fill data"},
        {"Requested metric": "DNA reduction additional appointment volume", "Shown in": "Outpatient Impact Table - More Throughput", "Status": "Included"},
        {"Requested metric": "PIFU additional appointment volume", "Shown in": "Outpatient Impact Table - More Throughput", "Status": "Included"},
        {"Requested metric": "F:N improvement additional appointment volume", "Shown in": "Outpatient Impact Table - More Throughput", "Status": "Included"},
        {"Requested metric": "Total additional appointment volume", "Shown in": "Outpatient Impact Table - More Throughput", "Status": "Included"},
        {"Requested metric": "RTT backlog impact by lever", "Shown in": "Outpatient Impact Table - RTT/backlog", "Status": "Included"},
        {"Requested metric": "Final RTT backlog after outpatient levers", "Shown in": "Outpatient Impact Table - RTT/backlog", "Status": "Included"},
        {"Requested metric": "Backlog reduction %", "Shown in": "Outpatient Impact Table - RTT/backlog", "Status": "Included"},
        {"Requested metric": "Indicative cost / capacity opportunity", "Shown in": "Outpatient Impact Table", "Status": "Included"},
        {"Requested metric": "Indicative income opportunity", "Shown in": "Outpatient Impact Table", "Status": "Included"},
    ]
)

outpatient_action_evidence_df = pd.DataFrame(
    [
        {
            "Action / question": "Can the outpatient baseline be calculated from raw data?",
            "Answer from available data": "Yes.",
            "Evidence": (
                f"Loaded source files: {outpatient_source_files}. Baseline contains "
                f"{format_number(outpatient_baseline['planned_appointments'])} unique Contact_ID records."
            ),
            "Still needs confirmation": "No period confirmation needed; baseline is fixed to Apr 2025-Mar 2026.",
        },
        {
            "Action / question": "Can planned appointments be defined?",
            "Answer from available data": "Partially.",
            "Evidence": "The model uses unique Contact_ID records as planned/booked appointment records.",
            "Still needs confirmation": "Contact_ID may not include empty template slots, so true template capacity still needs Cerner template data.",
        },
        {
            "Action / question": "Can attended appointments be defined?",
            "Answer from available data": "Yes, using status values.",
            "Evidence": (
                f"Attended = Checked In or Checked Out, counted once per Contact_ID. "
                f"Top statuses observed: {outpatient_status_counts}."
            ),
            "Still needs confirmation": "Only whether any other local status should also be treated as attended.",
        },
        {
            "Action / question": "Can DNA/no-show rate be calculated?",
            "Answer from available data": "Yes, for Status = No Show.",
            "Evidence": f"Observed DNA/no-show rate = {outpatient_model_current_dna_rate_pct:.1f}%.",
            "Still needs confirmation": "Confirm whether cancellations should be included in the opportunity definition.",
        },
        {
            "Action / question": "Can true template fill be calculated?",
            "Answer from available data": "No, not from the current outpatient contact file alone.",
            "Evidence": (
                "Template/slot/room columns found: "
                + (", ".join(outpatient_template_capacity_columns) if outpatient_template_capacity_columns else "none")
                + "."
            ),
            "Still needs confirmation": "Request clinic template slots, booked slots, empty slots and room/session capacity.",
        },
        {
            "Action / question": "Can clinic sessions be estimated?",
            "Answer from available data": "Yes, as a proxy.",
            "Evidence": (
                f"Proxy session = clinic/performance unit + date + AM/PM. Planned proxies = "
                f"{format_number(outpatient_planned_sessions_horizon)} over selected delivery weeks."
            ),
            "Still needs confirmation": "Replace proxy with official clinic-session extract when available.",
        },
        {
            "Action / question": "Can substantive outpatient DCC capacity be calculated?",
            "Answer from available data": "Partially.",
            "Evidence": (
                f"Job-plan file {job_plan_capacity['source']} gives "
                f"{format_number(workforce_outpatient_capacity_240_sessions)} outpatient DCC sessions over "
                f"{format_decimal(active_delivery_weeks, 1)} weeks."
            ),
            "Still needs confirmation": "Confirm Out-patient activities maps to outpatient clinic DCC and agree scope.",
        },
        {
            "Action / question": "Can RTT/PTL impact be proven from outpatient data alone?",
            "Answer from available data": "No, only modelled indicatively.",
            "Evidence": (
                "No RTT/pathway/waiting-list columns were found in the outpatient extract."
                if not outpatient_rtt_columns
                else f"Potential RTT columns found: {', '.join(outpatient_rtt_columns)}."
            ),
            "Still needs confirmation": "Validate outpatient RTT conversion assumption with RTT/PTL data.",
        },
        {
            "Action / question": "Can financial opportunity be calculated from both cost and income lenses?",
            "Answer from available data": "Yes as indicative gross lenses, not confirmed cashability.",
            "Evidence": (
                f"Cost/capacity lens = additional appointments x {format_currency(outpatient_value_per_appointment)}. "
                f"Income lens = additional appointments x {format_currency(outpatient_income_per_attendance)} "
                f"observed outpatient income per attendance. {outpatient_finance_evidence}"
            ),
            "Still needs confirmation": "Finance should confirm cashability, tariff/payment treatment, and whether income and cost opportunities overlap.",
        },
    ]
)

outpatient_baseline_input_df = pd.DataFrame(outpatient_baseline_input_rows)
outpatient_capacity_assumption_df = pd.DataFrame(outpatient_capacity_assumption_rows)
outpatient_impact_columns = [
    "Impact bucket",
    "Metric",
    "Current baseline",
    "What if 50% delivery",
    "What if 75% delivery",
    "What if 100% delivery",
    "Calculation / evidence",
    "Interpretation / action",
]
outpatient_impact_df = pd.DataFrame(outpatient_impact_rows)[
    outpatient_impact_columns
].rename(columns=outpatient_display_scenario_labels)
outpatient_impact_group_col = "What-if lens / group"
outpatient_impact_group_labels = {
    "Baseline context": "Baseline context",
    "Capacity / data gap": "Capacity and data gaps",
    "What if more throughput": "What if more throughput - same sessions, more patients",
    "What if same throughput, fewer sessions": "What if same throughput - fewer sessions, same patients",
    "RTT / backlog opportunity": "RTT/backlog opportunity - from more throughput",
    "Financial opportunity": "Financial opportunity - from more throughput",
}
outpatient_impact_group_order = list(outpatient_impact_group_labels.values())
outpatient_impact_df["Impact bucket"] = outpatient_impact_df[
    "Impact bucket"
].replace(outpatient_impact_group_labels)
outpatient_impact_df = outpatient_impact_df.rename(
    columns={"Impact bucket": outpatient_impact_group_col}
)
outpatient_rtt_lever_bridge_df = pd.DataFrame(outpatient_rtt_lever_bridge_rows)
outpatient_target_df = pd.DataFrame(outpatient_target_rows).rename(
    columns=outpatient_display_scenario_labels
)
outpatient_more_throughput_df = pd.DataFrame(outpatient_more_throughput_rows).rename(
    columns=outpatient_display_scenario_labels
)
outpatient_same_throughput_df = pd.DataFrame(outpatient_same_throughput_rows).rename(
    columns=outpatient_display_scenario_labels
)
outpatient_rtt_opportunity_df = pd.DataFrame(outpatient_rtt_opportunity_rows).rename(
    columns=outpatient_display_scenario_labels
)
outpatient_scenario_columns = [
    "Scenario",
    "Scenario interpretation",
    "Metric",
    "Current baseline",
    "What if 50% delivery",
    "What if 75% delivery",
    "What if 100% delivery",
    "Notes / calculation",
    "Actions",
]
outpatient_scenario_df = pd.DataFrame(outpatient_scenario_rows)[
    outpatient_scenario_columns
].rename(columns=outpatient_display_scenario_labels)

st.subheader("Outpatient Baseline Inputs")
with st.expander("Outpatient core definitions and guardrails", expanded=True):
    st.markdown(
        f"""
- Planned appointment records: unique `Contact_ID` records in the outpatient extract.
- Actual attended appointments: `Status` containing Checked In or Checked Out, counted once per `Contact_ID`.
- DNA/no-show rate: `Status` containing No Show divided by planned appointment records.
- Clinic-session proxy: clinic/performance unit + date + AM/PM. This is not a substitute for a Cerner template extract.
- Baseline: {outpatient_baseline_label}.
- Template fill: true measured template fill is not available in the current outpatient extract because empty template slots, booked slots, room capacity and planned clinic templates are not present.
- Appointment states: State 1, State 2 and State 3 move the outpatient appointment opportunity dial to 50%, 75% and 100% of the modelled lever opportunity. They are not separate baselines.
- Scenario buckets: more throughput keeps actual clinic-session proxies fixed and increases appointment volume; same throughput with fewer sessions keeps attended appointment volume fixed and estimates actual-session equivalents released.
- RTT/PTL impact is indicative: additional appointment volume is converted using the sidebar RTT/PTL conversion percentage.
        """
    )

with st.expander("Where the requested outpatient metrics appear", expanded=False):
    st.dataframe(
        outpatient_metric_coverage_df,
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Outpatient action answers from available data", expanded=True):
    st.dataframe(
        outpatient_action_evidence_df,
        use_container_width=True,
        hide_index=True,
    )

st.dataframe(
    outpatient_baseline_input_df,
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download outpatient baseline inputs as CSV",
    data=outpatient_baseline_input_df.to_csv(index=False).encode("utf-8"),
    file_name="outpatient_baseline_inputs.csv",
    mime="text/csv",
)

st.subheader("Outpatient Impact Table")
st.caption(
    "One grouped table: the first column shows whether each metric is baseline "
    "context, more throughput, same throughput with fewer sessions, RTT/backlog "
    "impact, or finance."
)
st.markdown(
    """
- **What if more throughput - same sessions, more patients:** actual clinic-session proxies are held fixed and the levers create extra appointment volume.
- **What if same throughput - fewer sessions, same patients:** attended appointment volume is held fixed and improved productivity releases actual clinic-session capacity.
- **RTT/backlog opportunity:** linked to the more-throughput lens because backlog only reduces when additional appointments are created.
    """
)
st.warning(
    "No measured outpatient template-fill data is available in the current raw file. "
    "The extract contains appointment/contact records, but not empty template slots, "
    "booked-slot capacity, room capacity or full clinic templates."
)
st.dataframe(
    outpatient_impact_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        outpatient_impact_group_col: st.column_config.TextColumn(
            outpatient_impact_group_col,
            width="large",
        ),
        "Metric": st.column_config.TextColumn("Metric", width="large"),
        "Current baseline": st.column_config.TextColumn(
            "Current baseline",
            width="small",
        ),
        "Calculation / evidence": st.column_config.TextColumn(
            "Calculation / evidence",
            width="large",
        ),
        "Interpretation / action": st.column_config.TextColumn(
            "Interpretation / action",
            width="large",
        ),
    },
)
st.download_button(
    "Download outpatient impact table as CSV",
    data=outpatient_impact_df.to_csv(index=False).encode("utf-8"),
    file_name="outpatient_impact_table.csv",
    mime="text/csv",
)

if rtt_wait_band_error is not None:
    st.warning(
        f"RTT backlog data could not be loaded: {rtt_wait_band_error}. "
        "The impact table falls back to total PTL only."
    )

st.subheader("Outpatient Estate and Workforce Capacity Assumptions")
st.caption(
    "These are fixed capacity or diagnostic inputs. They do not move with appointment-state "
    "delivery unless estate, template, job-plan or delivery-week assumptions change."
)
st.dataframe(
    outpatient_capacity_assumption_df,
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download outpatient capacity assumptions as CSV",
    data=outpatient_capacity_assumption_df.to_csv(index=False).encode("utf-8"),
    file_name="outpatient_capacity_assumptions.csv",
    mime="text/csv",
)

outpatient_lever_chart_df = pd.DataFrame(
    [
        {
            "Scenario": outpatient_display_scenario_labels[scenario].split(" (")[0],
            "Lever": "Template fill",
            "Additional appointments/month": output["template_fill_monthly"],
        }
        for scenario, output in outpatient_outputs.items()
    ]
    + [
        {
            "Scenario": outpatient_display_scenario_labels[scenario].split(" (")[0],
            "Lever": "DNA reduction",
            "Additional appointments/month": output["dna_reduction_monthly"],
        }
        for scenario, output in outpatient_outputs.items()
    ]
    + [
        {
            "Scenario": outpatient_display_scenario_labels[scenario].split(" (")[0],
            "Lever": "PIFU",
            "Additional appointments/month": output["pifu_monthly"],
        }
        for scenario, output in outpatient_outputs.items()
    ]
    + [
        {
            "Scenario": outpatient_display_scenario_labels[scenario].split(" (")[0],
            "Lever": "F:N ratio",
            "Additional appointments/month": output["fn_ratio_monthly"],
        }
        for scenario, output in outpatient_outputs.items()
    ]
)

fig_outpatient = px.bar(
    outpatient_lever_chart_df,
    x="Scenario",
    y="Additional appointments/month",
    color="Lever",
    barmode="stack",
    title="Outpatient Additional Appointments per Month by Lever and Appointment State",
)
fig_outpatient.update_layout(
    template="plotly_white",
    xaxis_title="Appointment uplift state",
    yaxis_title="Additional appointments/month",
    height=420,
    margin=dict(l=20, r=20, t=70, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)
st.plotly_chart(fig_outpatient, use_container_width=True)

with st.expander("Outpatient calculation notes"):
    st.markdown(
        f"""
- User-editable planning assumptions retained for sensitivity: {format_number(outpatient_patients_per_session)} patients/session, template fill planning value {outpatient_template_target_fill_pct:.1f}%, DNA planning value {outpatient_target_dna_rate_pct:.1f}%, PIFU {outpatient_pifu_conversion_pct:.1f}%, and F:N improvement {outpatient_fn_ratio_improvement_pct:.1f}%.
- Delivery period: {format_decimal(active_delivery_weeks, 1)} active weeks over {outpatient_horizon_months} months.
- Data-led baseline: Apr 2025-Mar 2026 planned appointment records {format_number(outpatient_baseline['planned_appointments'])}, attended appointments {format_number(outpatient_baseline['attended_appointments'])}, fill/attendance proxy {outpatient_model_current_fill_pct:.1f}%, DNA/no-show rate {outpatient_model_current_dna_rate_pct:.1f}%.
- Template fill data gap: true template fill cannot be measured from the current outpatient extract because empty slots, booked-slot capacity, room capacity and full clinic templates are not present.
- Template fill lever: the current {outpatient_model_current_fill_pct:.1f}% is only an attended/planned Contact_ID proxy, calculated as {format_number(outpatient_baseline['attended_appointments'])} Checked In/Checked Out contacts divided by {format_number(outpatient_baseline['planned_appointments'])} planned Contact_ID records. The planning value is {outpatient_template_target_fill_pct:.1f}%; if the observed proxy is above the planning value, no template-fill uplift is modelled.
- DNA lever: observed DNA/no-show {outpatient_model_current_dna_rate_pct:.1f}% to planning value {outpatient_target_dna_rate_pct:.1f}% across planned appointment records.
- PIFU lever: {outpatient_pifu_conversion_pct:.1f}% of observed attended follow-up appointments are released or redirected.
- F:N lever: {outpatient_fn_ratio_improvement_pct:.1f}% improvement applied to observed attended first appointments.
- Appointment states: State 1 applies 50% of the modelled appointment opportunity, State 2 applies 75%, and State 3 applies 100%.
- RTT/PTL conversion: {outpatient_rtt_conversion_pct:.0f}% of additional appointment volume is treated as reducing RTT/PTL pathways.
- RTT backlog opportunity: the opening backlog is {format_number(rtt_opening_backlog['total'])} total incomplete pathways from {latest_rtt_wait_bands['month']}. The outpatient impact table now shows one calculated backlog-impact row per lever, plus the total impact and final closing backlog.
- Cost/capacity opportunity: additional appointment volume over the horizon x {format_currency(outpatient_value_per_appointment)} per appointment. Default value is based on the Opportunity model's GBP150-GBP250 outpatient slot reference range.
- Income opportunity: additional appointment volume over the horizon x {format_currency(outpatient_income_per_attendance)} observed 25/26 outpatient income per attended contact. This is a gross income lens, not confirmed incremental income.
        """
    )

st.subheader("Outpatient Indicative Financial Opportunity")
st.caption(
    "Financial view for outpatient appointment states, split into cost/capacity and income lenses."
)

outpatient_finance_df = pd.DataFrame(
    [
        {
            "Appointment state": outpatient_display_scenario_labels[scenario].split(
                " ("
            )[0],
            f"Additional appts over {outpatient_horizon_months} months": format_number(
                output["additional_total"]
            ),
            "Value per extra appointment": format_currency(
                outpatient_value_per_appointment
            ),
            "Observed cost per attendance": format_currency(
                outpatient_cost_per_attendance
            ),
            "Cost / capacity opportunity": format_currency(
                output["financial_opportunity"]
            ),
            "Observed income per attendance": format_currency(
                outpatient_income_per_attendance
            ),
            "Income opportunity": format_currency(
                output["additional_total"] * outpatient_income_per_attendance
            ),
            "Basis": (
                "Cost/capacity lens = additional appointments x value per appointment. "
                "Observed cost per attendance is shown as a trial-balance comparator. "
                "Income lens = additional appointments x observed outpatient income per attended contact. "
                "Finance must confirm cashability, tariff/payment treatment, and overlap."
            ),
        }
        for scenario, output in outpatient_outputs.items()
    ]
)

st.dataframe(
    outpatient_finance_df,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Treat this as an avoided-cost or capacity-value proxy. It becomes cashable only if the released capacity directly replaces budgeted premium capacity, outsourcing, WLI, or other additional spend."
)

st.subheader("Theatre and Raw Finance Opportunity")
st.caption(
    "Scenario-linked theatre utilisation opportunity plus the other quantified finance opportunities from the raw finance data."
)

scenario_finance_df = pd.DataFrame(
    [
        {
            "Scenario": DISPLAY_SCENARIO_LABELS[scenario].replace(
                "What if utilisation improves to ",
                "Utilisation ",
            ),
            "Additional cases": format_number(
                scenario_b_outputs[scenario]["cases_unlocked"]
            ),
            "PTL after additional cases": format_number(
                scenario_b_outputs[scenario]["remaining_ptl"]
            ),
            "PTL reduction": format_percent(
                scenario_b_outputs[scenario]["ptl_reduction_pct"]
            ),
            "Theatre cost / capacity opportunity": format_currency(
                scenario_b_outputs[scenario]["cost_avoidance"]
            ),
            "Theatre income opportunity": format_currency(
                scenario_b_outputs[scenario]["cases_unlocked"]
                * THEATRE_CASE_VALUE_DEFAULT
            ),
            "Other quantified finance opportunity": format_currency(
                other_financial_benefit
            ),
            "Illustrative combined opportunity": format_currency(
                scenario_b_outputs[scenario]["cost_avoidance"]
                + other_financial_benefit
            ),
            "Basis": (
                "Cost/capacity lens = additional utilised minutes x theatre/"
                "anaesthetic cost per scheduled minute. Income lens = additional "
                f"cases x {format_currency(THEATRE_CASE_VALUE_DEFAULT)} agreed "
                "average income per case. Do not add income and cost lenses "
                "together without Finance confirmation."
            ),
        }
        for scenario, output in scenario_outputs.items()
    ]
)

st.dataframe(
    scenario_finance_df,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "The combined opportunity is illustrative and uses the cost/capacity lens. "
    "The theatre income opportunity is shown separately and should not be added "
    "to the cost/capacity total without Finance confirmation. Finance should "
    "also confirm whether the scenario-linked theatre utilisation value overlaps "
    "with WLI, outsourcing, Vanguard, or other avoided-cost lines before using it "
    "as a cashable total."
)

if financial_error is not None:
    st.warning(f"Financial opportunity data could not be loaded: {financial_error}")
else:
    finance_df = pd.DataFrame(financial_rows)[
        [
            "Opportunity",
            "25/26 actual spend",
            "Indicative 26/27 opportunity",
            "Financial category",
            "Calculation / evidence",
            "Backlog / capacity impact",
        ]
    ]

    with st.expander("Raw-data finance opportunity lines", expanded=True):
        st.dataframe(
            finance_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Opportunity": st.column_config.TextColumn(
                    "Opportunity",
                    width="medium",
                ),
                "25/26 actual spend": st.column_config.TextColumn(
                    "25/26 actual spend",
                    width="medium",
                ),
                "Indicative 26/27 opportunity": st.column_config.TextColumn(
                    "Indicative 26/27 opportunity",
                    width="medium",
                ),
                "Financial category": st.column_config.TextColumn(
                    "Financial category",
                    width="medium",
                ),
                "Calculation / evidence": st.column_config.TextColumn(
                    "Calculation / evidence",
                    width="large",
                ),
                "Backlog / capacity impact": st.column_config.TextColumn(
                    "Backlog / capacity impact",
                    width="large",
                ),
            },
        )

with st.expander("Assumptions summary for slide narrative", expanded=True):
    st.markdown(
        f"""
- Theatre baseline is calculated from full-year April 2025 to March 2026 elective theatre activity.
- Theatre utilisation is elective-only and uses actual sessions used for utilisation: anaesthetic-to-recovery touch minutes divided by scheduled minutes for elective, non-obstetric actual sessions after invalid/24-hour scheduled sessions are removed.
- Emergency, mixed elective/emergency, obstetrics, cancelled/not-run and invalid/24-hour sessions are excluded from the utilisation calculation, but shown in the elective/emergency context table.
- Actual sessions used for utilisation are elective, non-obstetric delivered sessions with touch time, completed cases, actual start, or actual finish recorded.
- All theatre sessions are standardised to 240-minute equivalent units.
- Theatre scenario A keeps activity constant and calculates sessions freed. Theatre scenario B keeps actual sessions constant and calculates additional cases and PTL impact.
- Theatre target states are utilisation improving to 78.5%, 81.75% and 85.0%.
- Outpatient baseline is fixed to April 2025 to March 2026: planned appointment records, attended appointments, DNA/no-show rate and clinic-session proxies are calculated from that raw outpatient period.
- Outpatient appointment states are anchored to observed outpatient volumes. State 1, State 2 and State 3 apply 50%, 75% and 100% of the modelled appointment opportunity.
- Outpatient full planning values are: booked-appointment fill proxy {outpatient_model_current_fill_pct:.1f}% to {outpatient_template_target_fill_pct:.1f}%; DNA {outpatient_model_current_dna_rate_pct:.1f}% to {outpatient_target_dna_rate_pct:.1f}%; PIFU conversion {outpatient_pifu_conversion_pct:.1f}% of observed follow-up activity; and F:N improvement {outpatient_fn_ratio_improvement_pct:.1f}% of observed first-attendance activity.
- Estate and workforce capacity layers are shown separately. Theatre estate uses {DEFAULT_ESTATE_THEATRES} elective theatres from PAH feedback, while noting the wider total theatre estate is {TOTAL_ESTATE_THEATRES}; outpatient estate is flagged as requiring a Cerner clinic-template/room-capacity extract.
- Theatre workforce layer shows total substantive operating sessions from {job_plan_capacity['source']}: {format_number(job_plan_capacity['theatre_weekly'])} per week and {format_number(workforce_theatre_capacity_240_sessions)} over {format_decimal(active_delivery_weeks, 1)} weeks.
- Outpatient financial opportunity is calculated as additional appointment volume over the horizon x {format_currency(outpatient_value_per_appointment)} per appointment; the default is a midpoint proxy from the Opportunity model's GBP150-GBP250 outpatient slot value range.
- Theatre income opportunity is calculated as additional theatre case volume x {format_currency(THEATRE_CASE_VALUE_DEFAULT)} agreed average income per case. This is a gross income/activity lens and is shown separately from theatre cost/capacity opportunity.
- RTT/PTL impact assumes each additional theatre case or converted outpatient appointment removes one pathway, adjusted by the selected RTT/PTL conversion percentage. For outpatients, each lever row shows the calculated backlog impact only; the final closing backlog is shown once at the end.
- Vanguard backlog/capacity impact shows 25/26 elective completed cases delivered in Vanguard Theatre 1. If Vanguard spend is reduced, this is the capacity that needs to be replaced or absorbed elsewhere to avoid backlog deterioration.
- Financial values are indicative avoided-cost opportunity proxies. They should not be treated as cashable savings until finance confirms that the activity displaces budgeted WLI, outsourcing, Vanguard, agency or other temporary capacity spend.
        """
    )

with st.expander("Data and calculation notes"):
    st.markdown(
        f"""
- Theatre baseline period: {capacity['Recent_Start_Date'].strftime('%d %b %Y')} to {capacity['Recent_End_Date'].strftime('%d %b %Y')}.
- Current measured elective utilisation: {format_percent(current_utilisation)} = touch minutes {format_number(touch_minutes)} / actual-session scheduled minutes {format_number(scheduled_minutes)}.
- Full-year elective 240-minute session equivalents: {format_number(full_year_elective_240_session_equivalents)} = Apr 2025-Mar 2026 elective actual-session scheduled minutes / {SESSION_STANDARD_MINUTES}.
- What-if utilisation end states: 78.5%, 81.75%, and 85.0%.
- Utilisation uplift: scenario utilisation minus current utilisation.
- Baseline-period scheduled minutes used: {format_number(scheduled_minutes)} for elective actual sessions only.
- Baseline-period touch minutes used: {format_number(float(capacity['Touch_Minutes']))}; start is anaesthetic start where available and end is patient into recovery where available.
- Touch-time outlier rows excluded: {format_number(float(capacity['Invalid_Touch_Time_Rows']))}; invalid/24-hour scheduled sessions excluded: {format_number(float(capacity['Invalid_Scheduled_Sessions']))}.
- Full-year elective actual sessions used for utilisation: {format_number(full_year_elective_actual_sessions_used)}.
- Full-year elective completed cases: {format_number(full_year_elective_completed_cases)}.
- Scenario A sessions required: baseline touch minutes / target utilisation / {SESSION_STANDARD_MINUTES}; cases are held constant.
- Scenario B additional utilised minutes: actual delivered scheduled minutes x utilisation uplift; sessions are held constant.
- Average procedure time: {format_decimal(avg_procedure_time, 1)} minutes = valid touch minutes / cases with valid touch time.
- Scenario B additional case volume: additional utilised minutes / average procedure time.
- In-year baseline cases: {format_number(baseline_delivery_cases)} = {format_decimal(cases_per_week, 1)} cases/week x {format_decimal(active_delivery_weeks, 1)} weeks.
- PTL impact applies only to Scenario B: latest PTL {format_number(current_ptl)} less additional case volume, assuming one additional case removes one pathway.
- Theatre / anaesthetic 25/26 spend used for financial proxy: {format_currency(theatre_cost_2526)}.
- Cost per scheduled minute: {format_currency(cost_per_scheduled_minute)} = 25/26 theatre / anaesthetic spend divided by in-year scheduled minutes.
- Theatre income opportunity: additional case volume x {format_currency(THEATRE_CASE_VALUE_DEFAULT)} agreed average income per case. This uses the theatre procedure/case activity volume and is a gross income lens.
- Scenario A financial opportunity: sessions freed x 240 minutes x cost per scheduled minute. Scenario B value proxy: additional utilised minutes x cost per scheduled minute. Neither should be treated as automatically cashable.
        """
    )
