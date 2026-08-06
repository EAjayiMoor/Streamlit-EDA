import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.financial_loader import (
    format_currency,
    format_optional_currency,
    load_staff_cost_data,
    load_trial_balance,
    surgical_theatre_mask,
)
from src.data.ptl_loader import load_ptl_data, summarise_ptl_by_month
from src.data.theatre_loader import (
    load_theatre_activity_data,
    summarise_theatre_capacity,
    summarise_vanguard_capacity_impact,
)


st.set_page_config(page_title="Financial Analysis", layout="wide")

st.title("Financial Analysis")
st.caption(
    "Populates elective recovery finance opportunities from the raw trial balance, staff cost, and theatre utilisation files."
)


def sum_abs(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).abs().sum())


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def build_item(label: str, fields_used: list[str], assumptions: list[str]) -> str:
    fields = "; ".join(fields_used)
    assumption_text = "; ".join(assumptions)
    return f"{label}\nFields used: {fields}\nAssumptions: {assumption_text}"


def format_vanguard_backlog_impact(
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
            f"{completed_cases / current_backlog:.1%} of latest PTL "
            f"({current_backlog:,.0f})"
        )

    specialty_text = ""
    if primary_specialty and primary_specialty != "Not available":
        specialty_text = (
            f" Main specialty: {primary_specialty} "
            f"({primary_specialty_cases:,.0f} cases)."
        )

    return (
        f"Vanguard delivered {completed_cases:,.0f} elective completed cases "
        f"in 25/26 across {sessions:,.0f} sessions. If Vanguard capacity is "
        "reduced, this capacity needs to be replaced or absorbed elsewhere to "
        f"avoid backlog deterioration; equivalent to {backlog_text}."
        f"{specialty_text}"
    )


try:
    tb_df = load_trial_balance()
    staff_df = load_staff_cost_data()
except Exception as e:
    st.error(f"Error loading finance data: {e}")
    st.stop()

try:
    theatre_df = load_theatre_activity_data()
    theatre_capacity = summarise_theatre_capacity(theatre_df, recent_months=12)
except Exception:
    theatre_df = pd.DataFrame()
    theatre_capacity = pd.Series(dtype="float64")

try:
    ptl_df = load_ptl_data()
    ptl_monthly_df = summarise_ptl_by_month(ptl_df)
    latest_ptl_size = (
        float(ptl_monthly_df.sort_values("PTL_Month").iloc[-1]["PTL Size"])
        if not ptl_monthly_df.empty
        else 0.0
    )
except Exception:
    latest_ptl_size = 0.0


st.sidebar.header("Benefit Assumptions")

agency_reduction_pct = st.sidebar.slider(
    "Agency spend reduction",
    min_value=0,
    max_value=100,
    value=10,
    step=5,
)

wli_reduction_pct = st.sidebar.slider(
    "WLI / outsourcing reduction",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
)

medinet_reduction_pct = st.sidebar.slider(
    "Endoscopy outsourcing reduction",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
)

lost_time_recovery_pct = st.sidebar.slider(
    "Recoverable lost theatre time",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
)

vanguard_2526_source = st.sidebar.selectbox(
    "Vanguard 25/26 spend source",
    ["Finance-provided value", "Raw trial balance keyword match"],
)

vanguard_2526_finance_value = st.sidebar.number_input(
    "Vanguard 25/26 finance value",
    min_value=0.0,
    max_value=20_000_000.0,
    value=3_000_000.0,
    step=50_000.0,
)

vanguard_2627_commitment = st.sidebar.number_input(
    "Vanguard 26/27 commitment",
    min_value=0.0,
    max_value=10_000_000.0,
    value=1_250_000.0,
    step=50_000.0,
)


# ---------------------------------------------------------
# Shared masks and source extracts
# ---------------------------------------------------------

staff_2526_df = staff_df[staff_df["Financial_Year"] == "25/26"].copy()
staff_surgical_mask = surgical_theatre_mask(staff_2526_df)
staff_surgical_df = staff_2526_df[staff_surgical_mask].copy()

tb_cost_mask = tb_df["Expenditure Type"].fillna("").astype(str).str.lower() != "income"
tb_surgical_mask = surgical_theatre_mask(tb_df)
tb_surgical_cost_df = tb_df[tb_cost_mask & tb_surgical_mask].copy()

agency_df = staff_surgical_df[
    staff_surgical_df["Pay type"].str.lower() == "agency"
].copy()
agency_spend_2526 = sum_abs(agency_df["Total cost"])
agency_benefit_2627 = agency_spend_2526 * agency_reduction_pct / 100

wli_pattern = (
    "waiting list|wli|insourcing|outsourcing|independent sector|independent"
)
wli_df = tb_surgical_cost_df[
    tb_surgical_cost_df["Search_Text"].str.contains(wli_pattern, regex=True, na=False)
].copy()
wli_spend_2526 = sum_abs(wli_df["FY_2526_Total"])
wli_benefit_2627 = wli_spend_2526 * wli_reduction_pct / 100

medinet_df = tb_df[
    tb_df["Search_Text"].str.contains("medinet", regex=False, na=False)
    & tb_cost_mask
].copy()

medinet_source = "Search_Text contains 'medinet'"

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
        "Medinet not separately identifiable; proxy uses Endoscopy + Non-NHS / "
        "NonPat Care / Purchase of Healthcare rows"
    )

medinet_spend_2526 = sum_abs(medinet_df["FY_2526_Total"])
medinet_benefit_2627 = medinet_spend_2526 * medinet_reduction_pct / 100

theatre_tb_df = tb_df[
    tb_df["Search_Text"].str.contains("theatre|anaesth", regex=True, na=False)
    & tb_cost_mask
].copy()
theatre_expenditure_2526 = sum_abs(theatre_tb_df["FY_2526_Total"])

scheduled_minutes = float(theatre_capacity.get("Scheduled_Minutes", 0))
touch_minutes = float(theatre_capacity.get("Touch_Minutes", 0))
lost_minutes = max(scheduled_minutes - touch_minutes, 0)
theatre_utilisation = float(theatre_capacity.get("Utilisation", 0))
lost_theatre_pct = 1 - theatre_utilisation if theatre_utilisation else 0
cost_per_scheduled_minute = (
    theatre_expenditure_2526 / scheduled_minutes if scheduled_minutes > 0 else 0
)
lost_time_value_2526 = lost_minutes * cost_per_scheduled_minute
lost_time_benefit_2627 = lost_time_value_2526 * lost_time_recovery_pct / 100

workforce_df = staff_surgical_df.copy()
workforce_spend_2526 = sum_abs(workforce_df["Total cost"])

monthly_wte_df = (
    workforce_df.groupby("Year/Month", as_index=False)
    .agg(WTE=("WTE equivalent", lambda values: values.abs().sum()))
    .sort_values("Year/Month")
)
average_workforce_wte = (
    float(monthly_wte_df["WTE"].mean()) if not monthly_wte_df.empty else 0
)

vanguard_df = tb_df[
    tb_df["Search_Text"].str.contains("vanguard", regex=False, na=False)
    & tb_cost_mask
].copy()
vanguard_raw_spend_2526 = sum_abs(vanguard_df["FY_2526_Total"])

if vanguard_2526_source == "Finance-provided value":
    vanguard_spend_2526 = vanguard_2526_finance_value
    vanguard_source_note = (
        f"Uses finance-provided 25/26 value of {format_currency(vanguard_2526_finance_value)}; "
        f"raw trial-balance keyword match identifies {format_currency(vanguard_raw_spend_2526)} "
        f"across {len(vanguard_df):,} rows."
    )
else:
    vanguard_spend_2526 = vanguard_raw_spend_2526
    vanguard_source_note = (
        f"Uses raw trial-balance keyword match: {len(vanguard_df):,} rows where "
        "Search_Text contains Vanguard and income is excluded."
    )

vanguard_benefit_2627 = max(vanguard_spend_2526 - vanguard_2627_commitment, 0)
vanguard_capacity = summarise_vanguard_capacity_impact(theatre_df)
vanguard_cases_2526 = float(vanguard_capacity.get("Completed_Cases", 0))
vanguard_sessions_2526 = float(vanguard_capacity.get("Sessions", 0))
vanguard_primary_specialty = str(
    vanguard_capacity.get("Primary_Specialty", "Not available")
)
vanguard_primary_specialty_cases = float(
    vanguard_capacity.get("Primary_Specialty_Cases", 0)
)
vanguard_backlog_impact = format_vanguard_backlog_impact(
    vanguard_cases_2526,
    vanguard_sessions_2526,
    latest_ptl_size,
    vanguard_primary_specialty,
    vanguard_primary_specialty_cases,
)


# ---------------------------------------------------------
# Summary table
# ---------------------------------------------------------

rows = [
    {
        "item": build_item(
            "Agency spend - surgical / theatres",
            [
                "Staff cost files",
                "Financial_Year = 25/26",
                "Pay type = Agency",
                "surgical/theatre specialty and cost-centre filter",
            ],
            [f"26/27 benefit = {agency_reduction_pct}% reduction in agency spend"],
        ),
        "25/26 actual spend": format_currency(agency_spend_2526),
        "26/27 benefit": format_currency(agency_benefit_2627),
        "Calculation / evidence": (
            f"{format_currency(agency_spend_2526)} 25/26 agency spend "
            f"x {agency_reduction_pct}% reduction = "
            f"{format_currency(agency_benefit_2627)}. "
            f"Evidence: {len(agency_df):,} staff-cost rows where Pay type = Agency "
            "and surgical/theatre filter is true."
        ),
        "Financial category": "Cost avoidance / cashable benefit",
        "_actual": agency_spend_2526,
        "_benefit": agency_benefit_2627,
    },
    {
        "item": build_item(
            "Lost theatre time",
            [
                "Theatre session ID",
                "Scheduled start / finish time",
                "Case Touch time (minutes)",
                "trial balance theatre and anaesthetic spend",
            ],
            [
                f"25/26 utilisation = {format_percent(theatre_utilisation)}",
                f"lost time = {format_percent(lost_theatre_pct)}",
                f"26/27 benefit = {lost_time_recovery_pct}% of lost-time value",
            ],
        ),
        "25/26 actual spend": format_currency(lost_time_value_2526),
        "26/27 benefit": format_currency(lost_time_benefit_2627),
        "Calculation / evidence": (
            f"Scheduled minutes {scheduled_minutes:,.0f} less touch minutes "
            f"{touch_minutes:,.0f} = {lost_minutes:,.0f} lost minutes. "
            f"Theatre/anaesthetic spend {format_currency(theatre_expenditure_2526)} "
            f"/ scheduled minutes = {format_currency(cost_per_scheduled_minute)} per minute. "
            f"Lost-time value {format_currency(lost_time_value_2526)} "
            f"x {lost_time_recovery_pct}% recoverable = "
            f"{format_currency(lost_time_benefit_2627)}."
        ),
        "Financial category": "Cost avoidance",
        "_actual": lost_time_value_2526,
        "_benefit": lost_time_benefit_2627,
    },
    {
        "item": build_item(
            "WLI / insourcing / outsourcing / independent sector",
            [
                "Trial balance",
                "surgical/theatre cost filter",
                "keyword filter: waiting list, WLI, insourcing, outsourcing, independent",
            ],
            [f"26/27 benefit = {wli_reduction_pct}% reduction in identified spend"],
        ),
        "25/26 actual spend": format_currency(wli_spend_2526),
        "26/27 benefit": format_currency(wli_benefit_2627),
        "Calculation / evidence": (
            f"{format_currency(wli_spend_2526)} identified 25/26 spend "
            f"x {wli_reduction_pct}% reduction = {format_currency(wli_benefit_2627)}. "
            f"Evidence: {len(wli_df):,} trial-balance rows matched to surgical/theatre "
            "plus waiting-list / WLI / insourcing / outsourcing / independent-sector keywords."
        ),
        "Financial category": "Cost avoidance / cashable if budgeted",
        "_actual": wli_spend_2526,
        "_benefit": wli_benefit_2627,
    },
    {
        "item": build_item(
            "Known: Endoscopy Medinet outsourcing",
            [
                "Trial balance",
                medinet_source,
            ],
            [f"26/27 benefit = {medinet_reduction_pct}% reduction / cease use"],
        ),
        "25/26 actual spend": format_currency(medinet_spend_2526),
        "26/27 benefit": format_currency(medinet_benefit_2627),
        "Calculation / evidence": (
            f"{format_currency(medinet_spend_2526)} identified 25/26 spend "
            f"x {medinet_reduction_pct}% reduction / cease use = "
            f"{format_currency(medinet_benefit_2627)}. "
            f"Evidence: {len(medinet_df):,} trial-balance rows. Source rule: {medinet_source}."
        ),
        "Financial category": "Cost avoidance / cashable if budgeted",
        "_actual": medinet_spend_2526,
        "_benefit": medinet_benefit_2627,
    },
    {
        "item": build_item(
            "Workforce - total FTE in actual, filtered to surgical / theatres",
            [
                "Staff cost files",
                "Financial_Year = 25/26",
                "WTE equivalent",
                "Total cost",
                "surgical/theatre specialty and cost-centre filter",
            ],
            [
                f"average monthly actual WTE = {average_workforce_wte:,.1f}",
                "budgeted FTE is not present in raw staff cost data",
            ],
        ),
        "25/26 actual spend": (
            f"{format_currency(workforce_spend_2526)}; "
            f"avg WTE {average_workforce_wte:,.1f}"
        ),
        "26/27 benefit": "Requires budgeted FTE / establishment data",
        "Calculation / evidence": (
            f"25/26 surgical/theatre workforce spend is {format_currency(workforce_spend_2526)} "
            f"with average monthly actual WTE of {average_workforce_wte:,.1f}. "
            "No 26/27 benefit calculated because budgeted FTE / establishment data is not present "
            "in the raw files."
        ),
        "Financial category": "Cashable",
        "_actual": 0,
        "_benefit": 0,
    },
    {
        "item": build_item(
            "Vanguard theatre capacity",
            [
                "Finance-provided 25/26 Vanguard value",
                "Trial balance Vanguard rows retained as evidence",
            ],
            [
                f"26/27 commitment assumption = {format_currency(vanguard_2627_commitment)}",
                "benefit = 25/26 spend less 26/27 commitment",
            ],
        ),
        "25/26 actual spend": format_currency(vanguard_spend_2526),
        "26/27 benefit": format_currency(vanguard_benefit_2627),
        "Calculation / evidence": (
            f"{format_currency(vanguard_spend_2526)} 25/26 Vanguard spend "
            f"less {format_currency(vanguard_2627_commitment)} 26/27 commitment = "
            f"{format_currency(vanguard_benefit_2627)}. "
            f"{vanguard_source_note}"
        ),
        "Backlog / capacity impact": vanguard_backlog_impact,
        "Financial category": "Cost avoidance / cashable if budgeted",
        "_actual": vanguard_spend_2526,
        "_benefit": vanguard_benefit_2627,
    },
]

for row in rows:
    row.setdefault(
        "Backlog / capacity impact",
        "Not directly quantified from this finance row.",
    )

total_actual = sum(row["_actual"] for row in rows)
total_benefit = sum(row["_benefit"] for row in rows)

rows.append(
    {
        "item": "Total quantified opportunity from available raw data",
        "25/26 actual spend": format_currency(total_actual),
        "26/27 benefit": format_currency(total_benefit),
        "Calculation / evidence": (
            "Sum of quantified opportunity rows only. Workforce baseline is excluded "
            "to avoid double counting pay already captured in agency and other rows."
        ),
        "Backlog / capacity impact": (
            "Not additive. Vanguard capacity impact is shown separately because it is "
            "capacity to replace, not a backlog reduction saving."
        ),
        "Financial category": "",
        "_actual": total_actual,
        "_benefit": total_benefit,
    }
)

summary_df = pd.DataFrame(rows)[
    [
        "item",
        "25/26 actual spend",
        "26/27 benefit",
        "Calculation / evidence",
        "Backlog / capacity impact",
        "Financial category",
    ]
]

st.subheader("Financial Opportunity Table")

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "item": st.column_config.TextColumn("item", width="large"),
        "25/26 actual spend": st.column_config.TextColumn(
            "25/26 actual spend",
            width="medium",
        ),
        "26/27 benefit": st.column_config.TextColumn(
            "26/27 benefit",
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
        "Financial category": st.column_config.TextColumn(
            "Financial category",
            width="medium",
        ),
    },
)

st.info(
    "The total excludes the workforce baseline row to avoid double counting pay already captured in agency and other opportunity lines. Rows marked as not quantified need additional finance inputs, usually budgeted FTE, establishment, or a named supplier code."
)


# ---------------------------------------------------------
# Evidence tabs
# ---------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["Evidence Summary", "Source Rows", "Trend"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantified 25/26 Spend", format_currency(total_actual))
    c2.metric("Quantified 26/27 Benefit", format_currency(total_benefit))
    c3.metric("Theatre Utilisation", format_percent(theatre_utilisation))
    c4.metric("Average Surgical/Theatre WTE", f"{average_workforce_wte:,.1f}")

    evidence_df = pd.DataFrame(
        [
            {
                "Area": "Agency",
                "Source rows": len(agency_df),
                "25/26 value": agency_spend_2526,
                "Benefit value": agency_benefit_2627,
            },
            {
                "Area": "WLI / outsourcing",
                "Source rows": len(wli_df),
                "25/26 value": wli_spend_2526,
                "Benefit value": wli_benefit_2627,
            },
            {
                "Area": "Endoscopy outsourcing",
                "Source rows": len(medinet_df),
                "25/26 value": medinet_spend_2526,
                "Benefit value": medinet_benefit_2627,
            },
            {
                "Area": "Vanguard",
                "Source rows": len(vanguard_df),
                "25/26 value": vanguard_spend_2526,
                "Benefit value": vanguard_benefit_2627,
                "Backlog / capacity impact": vanguard_cases_2526,
            },
        ]
    )

    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "25/26 value": st.column_config.NumberColumn(
                "25/26 value",
                format="£%d",
            ),
            "Benefit value": st.column_config.NumberColumn(
                "Benefit value",
                format="£%d",
            ),
            "Backlog / capacity impact": st.column_config.NumberColumn(
                "Backlog / capacity impact",
                format="%d cases",
            ),
        },
    )

with tab2:
    source_choice = st.selectbox(
        "Source extract",
        [
            "Agency staff cost rows",
            "WLI / outsourcing trial balance rows",
            "Endoscopy outsourcing rows",
            "Vanguard trial balance rows",
        ],
    )

    if source_choice == "Agency staff cost rows":
        display_source_df = agency_df.copy()
    elif source_choice == "WLI / outsourcing trial balance rows":
        display_source_df = wli_df.copy()
    elif source_choice == "Endoscopy outsourcing rows":
        display_source_df = medinet_df.copy()
    else:
        display_source_df = vanguard_df.copy()

    st.dataframe(display_source_df, use_container_width=True)

with tab3:
    trend_df = staff_surgical_df.groupby(["Year/Month", "Pay type"], as_index=False).agg(
        Spend=("Total cost", lambda values: values.abs().sum())
    )

    if trend_df.empty:
        st.warning("No staff cost trend data available after filtering.")
    else:
        fig = px.line(
            trend_df,
            x="Year/Month",
            y="Spend",
            color="Pay type",
            markers=True,
            title="Surgical / Theatre Staff Spend by Pay Type",
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Spend",
            height=520,
            margin=dict(l=20, r=20, t=70, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
