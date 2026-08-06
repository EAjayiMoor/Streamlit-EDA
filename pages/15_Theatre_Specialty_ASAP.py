import base64

import pandas as pd
import streamlit as st

from src.data.financial_loader import format_currency, load_trial_balance
from src.data.ptl_loader import load_ptl_data, summarise_ptl_by_month
from src.data.theatre_loader import (
    load_theatre_activity_data,
    summarise_theatre_session_type_split,
)


st.set_page_config(page_title="Theatre Specialty ASAP", layout="wide")
st.title("Theatre Specialty View")
st.caption(
    "Static fallback view. No interactive dataframe or chart components are used."
)

BASELINE_START = pd.Timestamp("2025-04-01")
BASELINE_END = pd.Timestamp("2026-03-31")
SESSION_MINUTES = 240
TARGETS = {
    "78.5%": 0.785,
    "81.8%": 0.818,
    "85.0%": 0.85,
}
THEATRE_CASE_VALUE = 250.0


def fmt_num(value: float) -> str:
    return f"{value:,.0f}"


def fmt_1dp(value: float) -> str:
    return f"{value:,.1f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def theatre_cost_base() -> float:
    tb_df = load_trial_balance()
    cost_mask = tb_df["Expenditure Type"].fillna("").astype(str).str.lower() != "income"
    theatre_mask = tb_df["Search_Text"].str.contains(
        "theatre|anaesth",
        regex=True,
        na=False,
    )
    return float(tb_df[cost_mask & theatre_mask]["FY_2526_Total"].abs().sum())


def latest_rtt_backlog() -> float:
    try:
        ptl = load_ptl_data("data/raw/rtt")
        monthly = summarise_ptl_by_month(ptl)
        if monthly.empty:
            return 39763.0
        return float(monthly.sort_values("Month").iloc[-1]["Total"])
    except Exception:
        return 39763.0


def classify_session_type(values: pd.Series) -> str:
    labels = set()
    for value in values.dropna().astype(str):
        text = value.strip().lower()
        if "elective" in text:
            labels.add("Elective")
        elif "emergency" in text or "trauma" in text:
            labels.add("Emergency")
    if labels == {"Elective"}:
        return "Elective"
    if labels == {"Emergency"}:
        return "Emergency"
    if {"Elective", "Emergency"}.issubset(labels):
        return "Mixed elective/emergency"
    return "Unknown"


def build_specialty_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Theatre session ID",
        "Booked Operation Date",
        "Scheduled start time(Session)",
        "Scheduled finish time(Session)",
        "Number of cases completed",
        "Specialty (standardised)",
    ]
    work = df.dropna(subset=required[:4]).copy()
    work["Specialty (standardised)"] = (
        work["Specialty (standardised)"].fillna("Unknown").astype(str).str.strip()
    )
    work.loc[work["Specialty (standardised)"] == "", "Specialty (standardised)"] = (
        "Unknown"
    )
    work["Scheduled_Minutes"] = (
        work["Scheduled finish time(Session)"] - work["Scheduled start time(Session)"]
    ).dt.total_seconds() / 60
    work.loc[work["Scheduled_Minutes"] < 0, "Scheduled_Minutes"] += 24 * 60

    touch_col = (
        "Model_Hospital_Touch_Minutes"
        if "Model_Hospital_Touch_Minutes" in work.columns
        else "Case Touch time (minutes)"
    )
    work[touch_col] = pd.to_numeric(work[touch_col], errors="coerce").fillna(0)
    work["Valid_Touch_Minutes"] = work[touch_col].where(
        work[touch_col].between(0, 720),
        0,
    )
    for col in ["Actual start time(Session)", "Actual finish time(Session)"]:
        if col not in work.columns:
            work[col] = pd.NaT

    work["_Session_Type"] = work.groupby(
        ["Booked Operation Date", "Theatre session ID"]
    )["Elective/Emergency"].transform(classify_session_type)
    work["_Session_Has_Obstetrics"] = (
        work["Specialty (standardised)"]
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
        & session_summary["Actual_Session_Flag"]
        & session_summary["Valid_Scheduled_Session"]
    ][["Booked Operation Date", "Theatre session ID"]]

    work = work.merge(
        eligible_sessions,
        on=["Booked Operation Date", "Theatre session ID"],
        how="inner",
    )
    specialty_session = (
        work.groupby(
            ["Booked Operation Date", "Theatre session ID", "Specialty (standardised)"],
            as_index=False,
        )
        .agg(
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

    summary = (
        specialty_session.groupby("Specialty (standardised)", as_index=False)
        .agg(
            Session_Specialty_Records=("Theatre session ID", "count"),
            Allocated_Scheduled_Minutes=("Allocated_Scheduled_Minutes", "sum"),
            Touch_Minutes=("Touch_Minutes", "sum"),
            Observed_Completed_Cases=("Completed_Cases", "sum"),
        )
    )
    summary["Observed_Utilisation"] = (
        summary["Touch_Minutes"] / summary["Allocated_Scheduled_Minutes"]
    ).fillna(0)
    return summary


theatre_df = load_theatre_activity_data()
baseline_df = theatre_df[
    (theatre_df["Booked Operation Date"] >= BASELINE_START)
    & (theatre_df["Booked Operation Date"] <= BASELINE_END)
].copy()

split = summarise_theatre_session_type_split(
    baseline_df,
    start_date=BASELINE_START,
    end_date=BASELINE_END,
)
elective_row = split[split["Session type"] == "Elective"].iloc[0]
model_row = split[
    split["Session type"] == "Elective excl obstetrics (model baseline)"
].iloc[0]

elective_scheduled_minutes = float(elective_row["Scheduled minutes used"])
elective_touch_minutes = float(elective_row["Touch minutes used"])
elective_cases = float(elective_row["Completed cases"])
elective_240_sessions = float(elective_row["Actual 240-min session equivalents"])
current_utilisation = elective_touch_minutes / elective_scheduled_minutes
avg_case_mins = float(model_row["Touch minutes used"]) / float(
    model_row["Completed cases"]
)
cost_per_minute = theatre_cost_base() / float(model_row["Scheduled minutes used"])
opening_backlog = latest_rtt_backlog()

specialty = build_specialty_summary(baseline_df)
specialty["Allocation share"] = (
    specialty["Allocated_Scheduled_Minutes"]
    / specialty["Allocated_Scheduled_Minutes"].sum()
)
specialty["Allocated elective cases"] = elective_cases * specialty["Allocation share"]
specialty["Allocated elective 240-min sessions"] = (
    elective_240_sessions * specialty["Allocation share"]
)

for label, target in TARGETS.items():
    addl_minutes = elective_scheduled_minutes * max(target - current_utilisation, 0)
    addl_cases = addl_minutes / avg_case_mins
    sessions_required = elective_touch_minutes / target / SESSION_MINUTES
    sessions_freed = max(elective_240_sessions - sessions_required, 0)
    capacity_value = sessions_freed * SESSION_MINUTES * cost_per_minute

    specialty[f"Additional cases @ {label}"] = addl_cases * specialty["Allocation share"]
    specialty[f"Total cases @ {label}"] = (
        specialty["Allocated elective cases"] + specialty[f"Additional cases @ {label}"]
    )
    specialty[f"Sessions freed @ {label}"] = sessions_freed * specialty[
        "Allocation share"
    ]
    specialty[f"Closing backlog @ {label}"] = (
        opening_backlog - specialty[f"Additional cases @ {label}"]
    )
    specialty[f"Capacity value @ {label}"] = capacity_value * specialty[
        "Allocation share"
    ]
    specialty[f"Per-case value @ {label}"] = (
        specialty[f"Additional cases @ {label}"] * THEATRE_CASE_VALUE
    )

specialty = specialty.sort_values("Additional cases @ 85.0%", ascending=False)

recon_rows = [
    [
        "Session measure used",
        "240-minute session equivalents for all visible session/capacity calculations",
    ],
    ["Elective scheduled minutes", fmt_num(elective_scheduled_minutes)],
    ["Elective touch minutes", fmt_num(elective_touch_minutes)],
    ["Elective completed cases", fmt_num(elective_cases)],
    ["Elective 240-min session equivalents", fmt_1dp(elective_240_sessions)],
    ["Current elective utilisation", fmt_pct(current_utilisation)],
    ["Average procedure time", f"{fmt_1dp(avg_case_mins)} mins"],
    ["Opening RTT backlog", fmt_num(opening_backlog)],
    ["Cost per scheduled minute", format_currency(cost_per_minute)],
    ["Specialty allocation share total", fmt_pct(specialty["Allocation share"].sum())],
]
recon = pd.DataFrame(recon_rows, columns=["Metric", "Value"])

display = specialty[
    [
        "Specialty (standardised)",
        "Allocation share",
        "Allocated elective 240-min sessions",
        "Allocated elective cases",
        "Observed_Utilisation",
        "Additional cases @ 78.5%",
        "Additional cases @ 81.8%",
        "Additional cases @ 85.0%",
        "Total cases @ 78.5%",
        "Total cases @ 81.8%",
        "Total cases @ 85.0%",
        "Sessions freed @ 78.5%",
        "Sessions freed @ 81.8%",
        "Sessions freed @ 85.0%",
        "Closing backlog @ 78.5%",
        "Closing backlog @ 81.8%",
        "Closing backlog @ 85.0%",
        "Capacity value @ 78.5%",
        "Capacity value @ 81.8%",
        "Capacity value @ 85.0%",
    ]
].copy()

raw_export = display.copy()
raw_export["Allocation share %"] = raw_export["Allocation share"] * 100

intervention_display = specialty[
    [
        "Specialty (standardised)",
        "Allocation share",
        "Observed_Utilisation",
        "Allocated elective cases",
        "Allocated elective 240-min sessions",
        "Additional cases @ 78.5%",
        "Additional cases @ 81.8%",
        "Additional cases @ 85.0%",
        "Sessions freed @ 78.5%",
        "Sessions freed @ 81.8%",
        "Sessions freed @ 85.0%",
        "Closing backlog @ 78.5%",
        "Closing backlog @ 81.8%",
        "Closing backlog @ 85.0%",
        "Capacity value @ 78.5%",
        "Capacity value @ 81.8%",
        "Capacity value @ 85.0%",
        "Per-case value @ 78.5%",
        "Per-case value @ 81.8%",
        "Per-case value @ 85.0%",
    ]
].copy()
intervention_display["Case contribution: baseline to 78.5%"] = (
    intervention_display["Additional cases @ 78.5%"]
)
intervention_display["Case contribution: 78.5% to 81.8%"] = (
    intervention_display["Additional cases @ 81.8%"]
    - intervention_display["Additional cases @ 78.5%"]
)
intervention_display["Case contribution: 81.8% to 85%"] = (
    intervention_display["Additional cases @ 85.0%"]
    - intervention_display["Additional cases @ 81.8%"]
)
intervention_display["Full additional elective cases at 85%"] = (
    intervention_display["Additional cases @ 85.0%"]
)
intervention_display["RTT backlog reduction at 78.5%"] = intervention_display[
    "Additional cases @ 78.5%"
]
intervention_display["RTT backlog reduction at 81.8%"] = intervention_display[
    "Additional cases @ 81.8%"
]
intervention_display["RTT backlog reduction at 85%"] = intervention_display[
    "Full additional elective cases at 85%"
]
intervention_export = intervention_display.copy()
intervention_display = intervention_display[
    [
        "Specialty (standardised)",
        "Allocation share",
        "Observed_Utilisation",
        "Allocated elective cases",
        "Allocated elective 240-min sessions",
        "Case contribution: baseline to 78.5%",
        "Case contribution: 78.5% to 81.8%",
        "Case contribution: 81.8% to 85%",
        "Full additional elective cases at 85%",
        "RTT backlog reduction at 78.5%",
        "RTT backlog reduction at 81.8%",
        "RTT backlog reduction at 85%",
        "Closing backlog @ 78.5%",
        "Closing backlog @ 81.8%",
        "Closing backlog @ 85.0%",
        "Sessions freed @ 78.5%",
        "Sessions freed @ 81.8%",
        "Sessions freed @ 85.0%",
        "Capacity value @ 78.5%",
        "Capacity value @ 81.8%",
        "Capacity value @ 85.0%",
        "Per-case value @ 78.5%",
        "Per-case value @ 81.8%",
        "Per-case value @ 85.0%",
    ]
].rename(
    columns={
        "Specialty (standardised)": "Specialty",
        "Allocation share": "Share of elective theatre time (%)",
        "Observed_Utilisation": "Current observed utilisation",
        "Allocated elective cases": "Baseline allocated elective cases",
        "Allocated elective 240-min sessions": (
            "Actual sessions delivered (240-min equivalents)"
        ),
        "Closing backlog @ 78.5%": "Closing RTT backlog after 78.5% impact",
        "Closing backlog @ 81.8%": "Closing RTT backlog after 81.8% impact",
        "Closing backlog @ 85.0%": "Closing RTT backlog after 85% impact",
        "Sessions freed @ 78.5%": "240-min sessions released at 78.5%",
        "Sessions freed @ 81.8%": "240-min sessions released at 81.8%",
        "Sessions freed @ 85.0%": "240-min sessions released at 85%",
        "Capacity value @ 78.5%": "Released-time value at 78.5%",
        "Capacity value @ 81.8%": "Released-time value at 81.8%",
        "Capacity value @ 85.0%": "Released-time value at 85%",
        "Per-case value @ 78.5%": "Extra-case income value at 78.5%",
        "Per-case value @ 81.8%": "Extra-case income value at 81.8%",
        "Per-case value @ 85.0%": "Extra-case income value at 85%",
    }
)
intervention_display[
    "Share of elective theatre time (%)"
] = intervention_display["Share of elective theatre time (%)"].map(
    lambda value: f"{float(value):.1%}"
)
intervention_display["Current observed utilisation"] = intervention_display[
    "Current observed utilisation"
].map(lambda value: f"{float(value):.1%}")
for col in intervention_display.columns:
    if col in {
        "Specialty",
        "Share of elective theatre time (%)",
        "Current observed utilisation",
    }:
        continue
    if "value" in col.lower():
        intervention_display[col] = intervention_display[col].map(
            lambda value: format_currency(float(value))
        )
    else:
        intervention_display[col] = intervention_display[col].map(
            lambda value: fmt_1dp(float(value))
        )

intervention_total = {col: "" for col in intervention_display.columns}
intervention_total["Specialty"] = "TOTAL"
intervention_total["Share of elective theatre time (%)"] = fmt_pct(
    specialty["Allocation share"].sum()
)
intervention_total["Current observed utilisation"] = fmt_pct(current_utilisation)
intervention_total["Baseline allocated elective cases"] = fmt_1dp(
    specialty["Allocated elective cases"].sum()
)
intervention_total["Actual sessions delivered (240-min equivalents)"] = fmt_1dp(
    specialty["Allocated elective 240-min sessions"].sum()
)
for raw_col, display_col in [
    (
        "Case contribution: baseline to 78.5%",
        "Case contribution: baseline to 78.5%",
    ),
    (
        "Case contribution: 78.5% to 81.8%",
        "Case contribution: 78.5% to 81.8%",
    ),
    ("Case contribution: 81.8% to 85%", "Case contribution: 81.8% to 85%"),
    (
        "Full additional elective cases at 85%",
        "Full additional elective cases at 85%",
    ),
    ("RTT backlog reduction at 78.5%", "RTT backlog reduction at 78.5%"),
    ("RTT backlog reduction at 81.8%", "RTT backlog reduction at 81.8%"),
    ("RTT backlog reduction at 85%", "RTT backlog reduction at 85%"),
    ("Sessions freed @ 78.5%", "240-min sessions released at 78.5%"),
    ("Sessions freed @ 81.8%", "240-min sessions released at 81.8%"),
    ("Sessions freed @ 85.0%", "240-min sessions released at 85%"),
    ("Capacity value @ 78.5%", "Released-time value at 78.5%"),
    ("Capacity value @ 81.8%", "Released-time value at 81.8%"),
    ("Capacity value @ 85.0%", "Released-time value at 85%"),
    ("Per-case value @ 78.5%", "Extra-case income value at 78.5%"),
    ("Per-case value @ 81.8%", "Extra-case income value at 81.8%"),
    ("Per-case value @ 85.0%", "Extra-case income value at 85%"),
]:
    total_value = intervention_export[raw_col].sum()
    intervention_total[display_col] = (
        format_currency(float(total_value))
        if "value" in display_col.lower()
        else fmt_1dp(float(total_value))
    )
intervention_total["Closing RTT backlog after 78.5% impact"] = "Not additive"
intervention_total["Closing RTT backlog after 81.8% impact"] = "Not additive"
intervention_total["Closing RTT backlog after 85% impact"] = "Not additive"
intervention_display = pd.concat(
    [intervention_display, pd.DataFrame([intervention_total])],
    ignore_index=True,
)

finance_display = specialty[
    [
        "Specialty (standardised)",
        "Additional cases @ 78.5%",
        "Additional cases @ 81.8%",
        "Additional cases @ 85.0%",
        "Sessions freed @ 78.5%",
        "Sessions freed @ 81.8%",
        "Sessions freed @ 85.0%",
        "Capacity value @ 78.5%",
        "Capacity value @ 81.8%",
        "Capacity value @ 85.0%",
        "Per-case value @ 78.5%",
        "Per-case value @ 81.8%",
        "Per-case value @ 85.0%",
    ]
].copy()
finance_export = finance_display.copy()
finance_display["How extra cases are calculated"] = (
    "Total extra cases at target x specialty share"
)
finance_display["How sessions released are calculated"] = (
    "Total sessions released at target x specialty share"
)
finance_display["How released-time value is calculated"] = (
    "Sessions released x 240 x cost per scheduled minute"
)
finance_display["How extra-case value is calculated"] = (
    "Extra elective cases x GBP250"
)
finance_display = finance_display.rename(
    columns={
        "Specialty (standardised)": "Specialty",
        "Additional cases @ 78.5%": (
            "Extra elective cases if utilisation reaches 78.5%"
        ),
        "Additional cases @ 81.8%": (
            "Extra elective cases if utilisation reaches 81.8%"
        ),
        "Additional cases @ 85.0%": (
            "Extra elective cases if utilisation reaches 85%"
        ),
        "Sessions freed @ 78.5%": (
            "240-min sessions released if same cases delivered at 78.5%"
        ),
        "Sessions freed @ 81.8%": (
            "240-min sessions released if same cases delivered at 81.8%"
        ),
        "Sessions freed @ 85.0%": (
            "240-min sessions released if same cases delivered at 85%"
        ),
        "Capacity value @ 78.5%": (
            "Value of released theatre time at 78.5%"
        ),
        "Capacity value @ 81.8%": (
            "Value of released theatre time at 81.8%"
        ),
        "Capacity value @ 85.0%": "Value of released theatre time at 85%",
        "Per-case value @ 78.5%": (
            "Indicative value of extra cases at 78.5%"
        ),
        "Per-case value @ 81.8%": (
            "Indicative value of extra cases at 81.8%"
        ),
        "Per-case value @ 85.0%": "Indicative value of extra cases at 85%",
    }
)
for col in finance_display.columns:
    if col == "Specialty" or col.startswith("How "):
        continue
    if "value" in col.lower():
        finance_display[col] = finance_display[col].map(
            lambda value: format_currency(float(value))
        )
    else:
        finance_display[col] = finance_display[col].map(
            lambda value: fmt_1dp(float(value))
        )

display = display.rename(
    columns={
        "Specialty (standardised)": "Specialty",
        "Allocation share": "Share of elective theatre time (%)",
        "Allocated elective 240-min sessions": (
            "Actual sessions delivered (240-min equivalents)"
        ),
        "Allocated elective cases": "Baseline cases",
        "Observed_Utilisation": "Observed utilisation",
    }
)
display["Share of elective theatre time (%)"] = display[
    "Share of elective theatre time (%)"
].map(lambda value: f"{float(value):.1%}")
display["Observed utilisation"] = display["Observed utilisation"].map(
    lambda value: f"{float(value):.1%}"
)
for col in display.columns:
    if col in {
        "Specialty",
        "Share of elective theatre time (%)",
        "Observed utilisation",
    }:
        continue
    if "value" in col.lower():
        display[col] = display[col].map(lambda value: format_currency(float(value)))
    else:
        display[col] = display[col].map(lambda value: fmt_1dp(float(value)))

total_row = {col: "" for col in display.columns}
total_row["Specialty"] = "TOTAL"
total_row["Share of elective theatre time (%)"] = fmt_pct(
    specialty["Allocation share"].sum()
)
total_row["Actual sessions delivered (240-min equivalents)"] = fmt_1dp(
    specialty["Allocated elective 240-min sessions"].sum()
)
total_row["Baseline cases"] = fmt_1dp(specialty["Allocated elective cases"].sum())
total_row["Observed utilisation"] = fmt_pct(current_utilisation)
for col in display.columns:
    if col in total_row and total_row[col] != "":
        continue
    if col.startswith("Closing backlog"):
        total_row[col] = "Not additive"
    elif col in specialty.columns:
        total_value = specialty[col].sum()
        total_row[col] = (
            format_currency(float(total_value))
            if "value" in col.lower()
            else fmt_1dp(float(total_value))
        )
display = pd.concat([display, pd.DataFrame([total_row])], ignore_index=True)

st.markdown(
    """
<style>
table {border-collapse: collapse; font-size: 13px; width: 100%;}
th, td {border: 1px solid #d7dde5; padding: 6px 8px; text-align: right;}
th:first-child, td:first-child {text-align: left;}
th {background: #f4f6f8;}
</style>
""",
    unsafe_allow_html=True,
)

st.subheader("Reconciliation")
st.markdown(recon.to_html(index=False, escape=True), unsafe_allow_html=True)

calculation_transparency = pd.DataFrame(
    [
        {
            "Output field": "Share of elective theatre time (%)",
            "What is needed": "Specialty allocated scheduled minutes; total elective scheduled minutes.",
            "Calculation": "Specialty allocated scheduled minutes / total elective scheduled minutes.",
            "Source": "Theatre activity extract, Apr 2025-Mar 2026.",
        },
        {
            "Output field": "Current observed utilisation",
            "What is needed": "Specialty touch minutes; specialty allocated scheduled minutes.",
            "Calculation": "Touch minutes / allocated scheduled minutes.",
            "Source": "Theatre activity extract, using anaesthetic-to-recovery touch time where available.",
        },
        {
            "Output field": "Baseline allocated elective cases",
            "What is needed": "Total elective completed cases; specialty share of elective theatre time.",
            "Calculation": "Total elective completed cases x specialty share.",
            "Source": "Elective theatre baseline and specialty allocation share.",
        },
        {
            "Output field": "Actual sessions delivered (240-min equivalents)",
            "What is needed": "Total elective 240-min session equivalents; specialty share of elective theatre time.",
            "Calculation": "Total elective 240-min sessions x specialty share.",
            "Source": "Elective theatre baseline and specialty allocation share.",
        },
        {
            "Output field": "Case contribution: baseline to 78.5%",
            "What is needed": "Specialty scheduled minutes; current utilisation; target utilisation 78.5%; average case time.",
            "Calculation": "Scheduled minutes x (78.5% - current utilisation) / average case time.",
            "Source": "Theatre activity baseline plus agreed utilisation target.",
        },
        {
            "Output field": "Case contribution: 78.5% to 81.8%",
            "What is needed": "Additional cases at 81.8%; additional cases at 78.5%.",
            "Calculation": "Additional cases at 81.8% - additional cases at 78.5%.",
            "Source": "Scenario calculations.",
        },
        {
            "Output field": "Case contribution: 81.8% to 85%",
            "What is needed": "Additional cases at 85%; additional cases at 81.8%.",
            "Calculation": "Additional cases at 85% - additional cases at 81.8%.",
            "Source": "Scenario calculations.",
        },
        {
            "Output field": "RTT backlog reduction at each state",
            "What is needed": "Additional elective cases at the utilisation state; RTT conversion assumption.",
            "Calculation": "Additional cases x RTT conversion. Current assumption is 1 case removes 1 RTT pathway.",
            "Source": "Scenario calculations and RTT assumption.",
        },
        {
            "Output field": "Closing RTT backlog at each state",
            "What is needed": "Opening RTT backlog; RTT backlog reduction at the utilisation state.",
            "Calculation": "Opening RTT backlog - RTT backlog reduction.",
            "Source": "Latest RTT backlog and scenario calculations.",
        },
        {
            "Output field": "240-min sessions released at each state",
            "What is needed": "Baseline 240-min sessions; touch minutes; target utilisation.",
            "Calculation": "Baseline 240-min sessions - (touch minutes / target utilisation / 240).",
            "Source": "Theatre activity baseline and target utilisation.",
        },
        {
            "Output field": "Released-time value at each state",
            "What is needed": "240-min sessions released; theatre/anaesthetic cost per scheduled minute.",
            "Calculation": "Sessions released x 240 x cost per scheduled minute.",
            "Source": "Theatre activity baseline and trial balance theatre/anaesthetic cost proxy.",
        },
        {
            "Output field": "Extra-case income value at each state",
            "What is needed": "Additional elective cases; value per additional case.",
            "Calculation": "Additional cases x GBP250 default value per case.",
            "Source": "Scenario calculations and default value assumption.",
        },
    ]
)
st.subheader("Calculation Transparency")
st.caption(
    "This table shows the inputs needed for each calculation so the specialty "
    "outputs can be traced back and checked."
)
st.markdown(
    calculation_transparency.to_html(index=False, escape=True),
    unsafe_allow_html=True,
)

st.subheader("Specialty Allocation")
st.markdown(display.to_html(index=False, escape=True), unsafe_allow_html=True)

st.subheader("Theatre Productivity Intervention Contribution by Specialty")
st.caption(
    "This table shows the staged contribution of theatre productivity by "
    "specialty. The three case-contribution columns add up to the full "
    "additional elective cases at 85% utilisation."
)
st.markdown(
    intervention_display.to_html(index=False, escape=True),
    unsafe_allow_html=True,
)

st.subheader("Specialty Financial Calculation")
st.caption(
    "Capacity value = sessions freed x 240 minutes x theatre/anaesthetic cost "
    "per scheduled minute. Per-case value = additional cases x GBP250 default "
    "case value."
)
finance_definitions = pd.DataFrame(
    [
        {
            "Field": "Additional cases",
            "Definition": (
                "Estimated extra elective cases that could be completed if theatre "
                "utilisation improves to the stated level."
            ),
            "Calculation": (
                "Total additional cases at target utilisation x specialty share."
            ),
            "Interpretation": (
                "This is the throughput / RTT opportunity. It is not a separate "
                "cash value."
            ),
        },
        {
            "Field": "Sessions freed",
            "Definition": (
                "Estimated 240-minute theatre sessions that would no longer be "
                "needed to deliver the same baseline case volume."
            ),
            "Calculation": (
                "Total sessions freed at target utilisation x specialty share."
            ),
            "Interpretation": (
                "This is a capacity-release view. It shows time that could be "
                "released, redeployed, or avoided if the same activity is done "
                "more efficiently."
            ),
        },
        {
            "Field": "Capacity value",
            "Definition": (
                "Indicative value of released theatre time, using the theatre/"
                "anaesthetic cost base."
            ),
            "Calculation": (
                "Sessions freed x 240 minutes x theatre/anaesthetic cost per "
                "scheduled minute."
            ),
            "Interpretation": (
                "This is a cost/capacity opportunity proxy. It is not automatically "
                "cashable unless sessions, staffing, outsourcing or estate usage "
                "can actually be reduced."
            ),
        },
        {
            "Field": "Per-case value",
            "Definition": (
                "Indicative value of additional elective cases delivered."
            ),
            "Calculation": "Additional cases x GBP250 default case value.",
            "Interpretation": (
                "This is a simple income/activity proxy. Replace GBP250 with an "
                "agreed tariff, income, contribution or avoided outsourcing value "
                "if finance confirms a better rate."
            ),
        },
        {
            "Field": "Cost per scheduled minute",
            "Definition": (
                "The average theatre/anaesthetic cost attached to one scheduled "
                "theatre minute."
            ),
            "Calculation": (
                "Theatre/anaesthetic 25/26 cost base from trial balance / model "
                "baseline scheduled minutes."
            ),
            "Interpretation": (
                "Used only for the capacity value calculation. It is a blended "
                "proxy, not a specialty-specific cost rate."
            ),
        },
    ]
)
st.markdown(
    finance_definitions.to_html(index=False, escape=True),
    unsafe_allow_html=True,
)
st.markdown(
    finance_display.to_html(index=False, escape=True),
    unsafe_allow_html=True,
)

csv = raw_export.to_csv(index=False).encode("utf-8")
csv_link = base64.b64encode(csv).decode("ascii")
st.markdown(
    f'<a download="theatre_specialty_asap.csv" href="data:text/csv;base64,{csv_link}">'
    "Download theatre specialty CSV</a>",
    unsafe_allow_html=True,
)

finance_csv = finance_export.to_csv(index=False).encode("utf-8")
finance_csv_link = base64.b64encode(finance_csv).decode("ascii")
st.markdown(
    f'<a download="theatre_specialty_finance_asap.csv" '
    f'href="data:text/csv;base64,{finance_csv_link}">'
    "Download theatre specialty finance CSV</a>",
    unsafe_allow_html=True,
)

intervention_csv = intervention_export.to_csv(index=False).encode("utf-8")
intervention_csv_link = base64.b64encode(intervention_csv).decode("ascii")
st.markdown(
    f'<a download="theatre_productivity_intervention_by_specialty.csv" '
    f'href="data:text/csv;base64,{intervention_csv_link}">'
    "Download theatre productivity intervention by specialty CSV</a>",
    unsafe_allow_html=True,
)

st.caption(
    "Method: Apr 2025-Mar 2026 elective theatre quantum is allocated to "
    "specialties by each specialty share of valid elective non-obstetric "
    "scheduled minutes. Additional cases use the agreed utilisation targets "
    "78.5%, 81.8% and 85.0%."
)
