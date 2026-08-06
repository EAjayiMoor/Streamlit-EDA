from pathlib import Path
import re

import pandas as pd
import streamlit as st

from src.data.outpatient_loader import load_outpatient_data
from src.data.ptl_loader import load_ptl_data, summarise_ptl_by_month
from src.data.theatre_loader import load_theatre_activity_data, summarise_theatre_capacity


st.set_page_config(page_title="Opportunity Analysis", layout="wide")

st.title("Opportunity Analysis")
st.caption(
    "Select a theme group from the opportunity model and review the quantified "
    "RTT/backlog and financial impact in a table format."
)


WORKBOOK_CANDIDATES = [
    Path("data/raw/Interventions/PAH_Opportunity_Model_V2.xlsx"),
    Path("/Users/emmanuelajayi/Downloads/PAH_Opportunity_Model_V2.xlsx"),
    Path("data/raw/Interventions/Opportunity model.csv"),
]

SHEET_NAME = "Intervention Opportunity Model"
BASELINE_START = pd.Timestamp("2025-04-01")
BASELINE_END = pd.Timestamp("2026-03-31")
BASELINE_LABEL = "Apr 2025 to Mar 2026"
DEFAULT_STATE_SHARES = [0.50, 0.75, 1.00]
THEATRE_UTILISATION_TARGETS = [0.785, 0.8175, 0.85]
ANNUAL_WEEKS = 52.0
TEN_MONTH_WEEKS = 43.0


def format_number(value: float) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value) if str(value).strip() else ""
    return f"{float(numeric):,.0f}"


def format_decimal(value: float, places: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value) if str(value).strip() else ""
    return f"{float(numeric):,.{places}f}"


def format_currency(value: float) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value) if str(value).strip() else ""
    return f"£{float(numeric):,.0f}"


def format_percent(value: float) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value) if str(value).strip() else ""
    return f"{float(numeric):.0%}"


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalise_column_name(column: object) -> str:
    return " ".join(str(column).replace("\n", " ").split()).strip()


NUMBER_PATTERN = r"\d[\d,]*(?:\.\d+)?"
ACTIVITY_UNITS_PATTERN = (
    r"case|cases|slot|slots|appointment|appointments|patient|patients|"
    r"scan|scans|OPA slots|OPA|day cases"
)


def parse_number_token(number_text: str, suffix: str = "") -> float:
    value = float(str(number_text).replace(",", ""))
    suffix = str(suffix).lower()
    if suffix == "m" or "million" in suffix:
        return value * 1_000_000
    if suffix == "k":
        return value * 1_000
    return value


def midpoint(low: float, high: float | None = None) -> float:
    if high is None:
        return low
    return (low + high) / 2


def extract_activity_quantity(text: str, period_pattern: str) -> float | None:
    text = str(text)
    range_pattern = (
        rf"({NUMBER_PATTERN})\s*(?:[–-]|to)\s*({NUMBER_PATTERN})\s+"
        rf"(?:additional\s+|extra\s+|fewer\s+|new\s+|first\s+)?"
        rf"(?:{ACTIVITY_UNITS_PATTERN})\b[^.]*?(?:{period_pattern})"
    )
    range_match = re.search(range_pattern, text, flags=re.IGNORECASE)
    if range_match:
        return midpoint(
            parse_number_token(range_match.group(1)),
            parse_number_token(range_match.group(2)),
        )

    single_pattern = (
        rf"({NUMBER_PATTERN})\s+"
        rf"(?:additional\s+|extra\s+|fewer\s+|new\s+|first\s+)?"
        rf"(?:{ACTIVITY_UNITS_PATTERN})\b[^.]*?(?:{period_pattern})"
    )
    single_match = re.search(single_pattern, text, flags=re.IGNORECASE)
    if single_match:
        return parse_number_token(single_match.group(1))

    return None


def derive_weekly_opportunity_from_assumption(text: str) -> tuple[float, str]:
    text = str(text)
    if not text.strip() or text.lower() == "nan":
        return 0.0, ""

    over_ten_months = extract_activity_quantity(
        text,
        r"over\s+10\s+months|over\s+ten\s+months",
    )
    if over_ten_months is not None:
        return (
            over_ten_months / TEN_MONTH_WEEKS,
            "Column L narrative: activity over 10 months converted to weekly",
        )

    annual = extract_activity_quantity(
        text,
        r"/\s*yr|/\s*year|per\s+year|annually|yr\b",
    )
    if annual is not None:
        return (
            annual / ANNUAL_WEEKS,
            "Column L narrative: annual activity converted to weekly",
        )

    weekly = extract_activity_quantity(
        text,
        r"/\s*wk|/\s*week|per\s+week|weekly",
    )
    if weekly is not None:
        return weekly, "Column L narrative: weekly activity"

    return 0.0, ""


def derive_financial_opportunity_from_assumption(text: str) -> tuple[float, str]:
    text = str(text)
    if not text.strip() or text.lower() == "nan":
        return 0.0, ""

    equals_match = re.search(
        rf"=\s*£\s*({NUMBER_PATTERN})\s*([mk]?)",
        text,
        flags=re.IGNORECASE,
    )
    if equals_match:
        value = parse_number_token(equals_match.group(1), equals_match.group(2))
        context = text[equals_match.end() : equals_match.end() + 80].lower()
        if value >= 10_000 and re.search(r"sav|benefit|opportun|income|recover", context):
            return value, "Column L narrative: finance value"

    range_match = re.search(
        rf"£\s*({NUMBER_PATTERN})\s*([mk]?)\s*(?:[–-]|to)\s*£?\s*"
        rf"({NUMBER_PATTERN})\s*([mk]?)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        first_suffix = range_match.group(2) or range_match.group(4)
        low = parse_number_token(range_match.group(1), first_suffix)
        high = parse_number_token(range_match.group(3), range_match.group(4))
        value = midpoint(low, high)
        if value >= 10_000:
            return value, "Column L narrative: finance range midpoint"

    single_matches = list(
        re.finditer(
            rf"£\s*({NUMBER_PATTERN})\s*([mk]?)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if single_matches:
        # Prefer values near benefit/opportunity/saving wording, then fall back to
        # the largest material value to avoid small per-slot prices.
        scored = []
        for match in single_matches:
            value = parse_number_token(match.group(1), match.group(2))
            context = text[max(match.start() - 80, 0) : match.end() + 80].lower()
            score = 1 if re.search(r"sav|benefit|opportun|income|recover", context) else 0
            scored.append((score, value))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        value = scored[0][1]
        if value >= 10_000:
            return value, "Column L narrative: finance value"

    return 0.0, ""


def find_default_workbook() -> Path:
    for path in WORKBOOK_CANDIDATES:
        if path.exists():
            return path
    return WORKBOOK_CANDIDATES[0]


def find_header_row(path: Path) -> int:
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, nrows=20)
    for idx, row in raw.iterrows():
        row_values = row.fillna("").astype(str).str.strip().tolist()
        if "Group / Theme" in row_values and "Intervention" in row_values:
            return int(idx)
    return 0


@st.cache_data(show_spinner=False)
def load_opportunity_model(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Opportunity model not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        header_row = find_header_row(path)
        raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=header_row)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("Opportunity model must be CSV or Excel.")

    raw.columns = [normalise_column_name(col) for col in raw.columns]
    raw = raw.dropna(how="all").copy()

    column_map = {
        "Group / Theme": "Group_Theme",
        "Intervention": "Intervention",
        "Current Baseline (PAH position)": "Current_Baseline",
        "Target (E.g. Model Hospital Upper Quartile/GIRFT/NHSE Best Practice Benchmark)": "Benchmark",
        "PAH Target (E.g. Model Hospital Upper Quartile/GIRFT/NHSE Best Practice Benchmark)": "PAH_Target",
        "Total Financial Opportunity (£)": "Financial_Opportunity_Text",
        "Total Performance Opportunity (RTT/PTL)": "Performance_Opportunity_Text",
        "Numerator": "Numerator",
        "Denominator": "Denominator",
        "R/NR, cashable, non cashable , busget and cost avoidance (where spend is in excess of budget)": "Financial_Category",
        "Assumption Detail (lever narrative)": "Assumption_Detail",
        "Additional Cases / Slots per Week": "Additional_Cases_Slots_Per_Week",
        "Weeks to Recover Position": "Weeks_To_Recover",
        "Investment Required (£)": "Investment_Required",
        "WTE Required (peak)": "WTE_Required_Peak",
        "Annual Financial Benefit (£)": "Annual_Financial_Benefit",
        "WTE Released (steady state)": "WTE_Released_Steady_State",
        "Net Position (Benefit minus Cost £)": "Net_Position",
        "Validation (1–5) Does it reflect PAH reality?": "Validation_Score",
        "Evidence (1–5) National data / GIRFT support?": "Evidence_Score",
        "Impact (1–5) Elective recovery benefit?": "Impact_Score",
        "Deliverability (1–5) Feasible at pace?": "Deliverability_Score",
        "Total score": "Total_Score",
        "Recommendation (auto)": "Recommendation",
        "Validation Status": "Validation_Status",
    }
    df = raw.rename(columns=column_map).copy()

    required = ["Group_Theme", "Intervention", "Additional_Cases_Slots_Per_Week"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required opportunity model columns: {missing}")

    df = df.dropna(subset=["Group_Theme", "Intervention"], how="all").copy()
    df["Group_Theme"] = df["Group_Theme"].fillna("").astype(str).str.strip()
    df["Intervention"] = df["Intervention"].fillna("").astype(str).str.strip()
    df = df[df["Intervention"] != ""].copy()
    df = df[~df["Group_Theme"].str.contains("TOTAL", case=False, na=False)].copy()

    text_cols = [
        "Group_Theme",
        "Intervention",
        "Current_Baseline",
        "Benchmark",
        "PAH_Target",
        "Financial_Opportunity_Text",
        "Performance_Opportunity_Text",
        "Numerator",
        "Denominator",
        "Financial_Category",
        "Assumption_Detail",
        "Recommendation",
        "Validation_Status",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_cols = [
        "Additional_Cases_Slots_Per_Week",
        "Weeks_To_Recover",
        "Investment_Required",
        "WTE_Required_Peak",
        "Annual_Financial_Benefit",
        "WTE_Released_Steady_State",
        "Net_Position",
        "Validation_Score",
        "Evidence_Score",
        "Impact_Score",
        "Deliverability_Score",
        "Total_Score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
        else:
            df[col] = pd.NA

    df["Additional_Cases_Slots_Per_Week"] = df[
        "Additional_Cases_Slots_Per_Week"
    ].fillna(0)
    narrative_activity = df["Assumption_Detail"].apply(
        derive_weekly_opportunity_from_assumption
    )
    df["Narrative_Weekly_Opportunity"] = narrative_activity.apply(lambda item: item[0])
    df["Narrative_Opportunity_Source"] = narrative_activity.apply(lambda item: item[1])

    narrative_finance = df["Assumption_Detail"].apply(
        derive_financial_opportunity_from_assumption
    )
    df["Narrative_Financial_Opportunity"] = narrative_finance.apply(lambda item: item[0])
    df["Narrative_Financial_Source"] = narrative_finance.apply(lambda item: item[1])

    explicit_weekly = df["Additional_Cases_Slots_Per_Week"].fillna(0)
    df["Effective_Weekly_Opportunity"] = explicit_weekly.where(
        explicit_weekly > 0,
        df["Narrative_Weekly_Opportunity"],
    )
    df["Opportunity_Driver_Source"] = "No numeric activity driver"
    df.loc[explicit_weekly > 0, "Opportunity_Driver_Source"] = (
        "Column M: Additional Cases / Slots per Week"
    )
    narrative_driver_mask = (explicit_weekly <= 0) & (
        df["Narrative_Weekly_Opportunity"] > 0
    )
    df.loc[narrative_driver_mask, "Opportunity_Driver_Source"] = df.loc[
        narrative_driver_mask,
        "Narrative_Opportunity_Source",
    ]
    df["Is_Modelable"] = (
        (df["Effective_Weekly_Opportunity"] > 0)
        | (df["Narrative_Financial_Opportunity"] > 0)
        | df.apply(is_ptl_validation_intervention, axis=1)
    )
    return df.reset_index(drop=True)


def get_latest_backlog() -> tuple[float, str]:
    try:
        ptl = summarise_ptl_by_month(load_ptl_data())
        latest = ptl.sort_values("PTL_Month").iloc[-1]
        return float(latest["PTL Size"]), latest["PTL_Month"].strftime("%b %Y")
    except Exception:
        return 0.0, "not available"


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

    return {
        "planned_appointments": planned_appointments,
        "attended_appointments": attended_appointments,
        "dna_appointments": dna_appointments,
        "planned_sessions": float(len(planned_session_df)),
        "actual_sessions": float(len(actual_session_df)),
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
        "follow_up": follow_up,
        "first_attendance": first_attendance,
    }


@st.cache_data(show_spinner=False)
def load_available_baselines() -> dict:
    baselines = {
        "theatre": {"available": False, "error": ""},
        "outpatient": {"available": False, "error": ""},
    }

    try:
        theatre_df = load_theatre_activity_data()
        capacity = summarise_theatre_capacity(
            theatre_df,
            start_date=BASELINE_START,
            end_date=BASELINE_END,
            session_type_scope="elective",
            exclude_obstetrics=True,
            actual_sessions_only_for_utilisation=True,
            touch_time_column="Model_Hospital_Touch_Minutes",
        )
        if not capacity.empty:
            baselines["theatre"] = {
                "available": True,
                "utilisation": float(capacity.get("Utilisation", 0.0)),
                "actual_sessions": float(capacity.get("Actual_Sessions", 0.0)),
                "actual_240_sessions": float(
                    capacity.get("Actual_240_Session_Equivalents", 0.0)
                ),
                "scheduled_minutes": float(
                    capacity.get("Actual_Session_Scheduled_Minutes", 0.0)
                ),
                "touch_minutes": float(capacity.get("Touch_Minutes", 0.0)),
                "completed_cases": float(capacity.get("Completed_Cases", 0.0)),
                "avg_case_duration": float(
                    capacity.get("Average_Case_Duration_Minutes", 0.0)
                ),
                "observed_weeks": float(capacity.get("Observed_Weeks", 1.0)),
            }
    except Exception as exc:
        baselines["theatre"]["error"] = str(exc)

    try:
        outpatient_df = load_outpatient_data()
        outpatient_period = outpatient_df[
            (outpatient_df["Contact_Start"] >= BASELINE_START)
            & (outpatient_df["Contact_Start"] <= BASELINE_END)
        ].copy()
        outpatient_baseline = build_outpatient_baseline(outpatient_period)
        baselines["outpatient"] = {
            "available": not outpatient_period.empty,
            **outpatient_baseline,
        }
    except Exception as exc:
        baselines["outpatient"]["error"] = str(exc)

    return baselines


def is_theatre_theme(theme: str) -> bool:
    text = theme.lower()
    return "theatre" in text


def is_outpatient_theme(theme: str) -> bool:
    text = theme.lower()
    return "outpatient" in text


def is_commercial_theme(theme: str) -> bool:
    text = theme.lower()
    return "commercial" in text or "contract" in text


def is_ptl_validation_intervention(row: pd.Series) -> bool:
    text = " ".join(
        str(row.get(col, ""))
        for col in [
            "Intervention",
            "Current_Baseline",
            "Benchmark",
            "PAH_Target",
            "Assumption_Detail",
        ]
    ).lower()
    return "validation" in text and ("ptl" in text or "waiting list" in text)


def extract_percent_assumption(text: str) -> float | None:
    text = str(text)
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*%", text)
    if range_match:
        return float(range_match.group(1)) / 100

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if matches:
        return float(matches[0]) / 100

    return None


def ptl_validation_rate(row: pd.Series) -> float | None:
    for col in ["PAH_Target", "Benchmark", "Current_Baseline", "Assumption_Detail"]:
        rate = extract_percent_assumption(str(row.get(col, "")))
        if rate is not None:
            return rate
    return None


def ptl_validation_reduction(row: pd.Series, opening_backlog: float, share: float) -> float:
    rate = ptl_validation_rate(row)
    if rate is None or opening_backlog <= 0:
        return 0.0
    return min(opening_backlog * rate * share, opening_backlog)


def build_opportunity_states(
    theme: str,
    weekly_full: float,
    baselines: dict,
    finance_full: float = 0.0,
) -> list[dict]:
    if is_theatre_theme(theme) and baselines["theatre"].get("available"):
        current_utilisation = baselines["theatre"]["utilisation"]
        max_target_gap = max(THEATRE_UTILISATION_TARGETS[-1] - current_utilisation, 0)
        states = []
        for target in THEATRE_UTILISATION_TARGETS:
            target_gap = max(target - current_utilisation, 0)
            share = target_gap / max_target_gap if max_target_gap > 0 else 0
            states.append(
                {
                    "label": f"Utilisation to {target:.1%}",
                    "share": min(max(share, 0), 1),
                    "target_value": target,
                    "target_type": "theatre_utilisation",
                }
            )
        return states

    if weekly_full <= 0:
        if is_commercial_theme(theme):
            labels = [
                "Coding/BPT audit",
                "Validated income risk",
                "Confirmed income benefit",
            ]
            shares = DEFAULT_STATE_SHARES if finance_full > 0 else [0.0, 0.0, 0.0]
        else:
            labels = [
                "Baseline to quantify",
                "Validated opportunity",
                "Confirmed benefit",
            ]
            shares = [0.0, 0.0, 0.0]
        return [
            {
                "label": label,
                "share": share,
                "target_value": 0.0,
                "target_type": "not_quantified",
            }
            for label, share in zip(labels, shares)
        ]

    states = []
    for share in DEFAULT_STATE_SHARES:
        weekly_activity = weekly_full * share
        if is_outpatient_theme(theme):
            label = f"+{format_decimal(weekly_activity, 1)} appts/week"
        else:
            label = f"+{format_decimal(weekly_activity, 1)} cases/slots/week"
        states.append(
            {
                "label": label,
                "share": share,
                "target_value": weekly_activity,
                "target_type": "weekly_activity",
            }
        )
    return states


def build_intervention_impact_table(
    theme: str,
    theme_df: pd.DataFrame,
    opportunity_states: list[dict],
    horizon_weeks: float,
    value_per_case_slot: float,
    opening_backlog: float,
) -> pd.DataFrame:
    work = theme_df.copy()
    rows = []
    for _, row in work.iterrows():
        explicit_weekly = float(row.get("Additional_Cases_Slots_Per_Week", 0) or 0)
        weekly_full = float(row.get("Effective_Weekly_Opportunity", explicit_weekly) or 0)
        narrative_finance = float(row.get("Narrative_Financial_Opportunity", 0) or 0)
        is_ptl_validation = is_ptl_validation_intervention(row)
        has_numeric_driver = weekly_full > 0
        has_narrative = any(
            str(row.get(col, "")).strip()
            for col in [
                "Current_Baseline",
                "Assumption_Detail",
                "Financial_Opportunity_Text",
                "Performance_Opportunity_Text",
            ]
        )
        output = {
            "Group / Theme": row.get("Group_Theme", ""),
            "Intervention": row.get("Intervention", ""),
            "Calculation flag": (
                "Calculated from latest PTL backlog and workbook validation assumption; actual validation outcome data not available"
                if is_ptl_validation
                else (
                    "Calculated from workbook weekly opportunity"
                    if explicit_weekly > 0
                    else (
                        "Calculated from Column L narrative-derived opportunity"
                        if has_numeric_driver
                        else (
                            "Calculated from Column L narrative finance opportunity"
                            if narrative_finance > 0
                            else (
                                "Workbook baseline provided - numeric opportunity driver needed"
                                if str(row.get("Current_Baseline", "")).strip()
                                else (
                                    "Workbook narrative only - numeric opportunity driver needed"
                                    if has_narrative
                                    else "Needs source data"
                                )
                            )
                        )
                    )
                )
            ),
            "Baseline source": (
                "Latest PTL backlog + Column D workbook baseline"
                if is_ptl_validation
                else (
                    "Column D: Current Baseline (PAH position)"
                    if str(row.get("Current_Baseline", "")).strip()
                    else (
                        "No Column D baseline populated"
                        if not has_narrative
                        else "Other workbook narrative only"
                    )
                )
            ),
            "Current baseline": row.get("Current_Baseline", ""),
            "Benchmark / target": row.get("Benchmark", ""),
            "PAH target": row.get("PAH_Target", ""),
            "Assumption detail": row.get("Assumption_Detail", ""),
            "Additional cases / slots per week": explicit_weekly,
            "Modelled cases / slots per week": weekly_full,
            "Opportunity driver source": row.get("Opportunity_Driver_Source", ""),
            "Column L narrative finance": narrative_finance,
            "Finance category": row.get("Financial_Category", ""),
            "Finance opportunity narrative": row.get("Financial_Opportunity_Text", ""),
            "Validation status": row.get("Validation_Status", ""),
        }
        for state in opportunity_states:
            opportunity_share = state["share"]
            state_label = state_column_label(state)
            total = (
                ptl_validation_reduction(row, opening_backlog, opportunity_share)
                if is_ptl_validation
                else weekly_full * opportunity_share * horizon_weeks
            )
            output[f"{state_label}: total activity"] = total
            output[f"{state_label}: finance proxy"] = (
                0
                if is_ptl_validation
                else (
                    narrative_finance * opportunity_share
                    if narrative_finance > 0 and weekly_full <= 0
                    else total * value_per_case_slot
                )
            )
        rows.append(output)

    return pd.DataFrame(rows)


def state_column_label(state: dict) -> str:
    return str(state["label"])


def summary_calculation_flag(
    row: pd.Series,
    weekly_full: float,
    annual_benefit: float,
    investment: float,
) -> str:
    group = str(row.get("What-if lens / group", ""))
    metric = str(row.get("Metric", ""))
    row_weekly_driver = pd.to_numeric(row.get("_Weekly_Driver"), errors="coerce")
    row_driver_source = str(row.get("_Driver_Source", ""))
    row_finance_driver = pd.to_numeric(row.get("_Narrative_Finance"), errors="coerce")

    if group == "Data-led baseline":
        return "Calculated from raw Apr 2025-Mar 2026 data"
    if group == "Workbook baseline":
        return "Workbook baseline provided from Column D"
    if group in {
        "What if more throughput",
        "What if same throughput, fewer sessions",
    }:
        if bool(row.get("_PTL_Validation", False)):
            return "Calculated from latest PTL backlog and workbook validation assumption; actual validation outcome data not available"
        if pd.notna(row_weekly_driver) and row_weekly_driver > 0:
            if "Column L" in row_driver_source:
                return "Calculated from Column L narrative-derived opportunity"
            return "Calculated from workbook weekly opportunity"
        if pd.notna(row_finance_driver) and row_finance_driver > 0:
            return "Calculated from Column L narrative finance opportunity"
        if str(row.get("Current baseline", "")).strip():
            return "Workbook baseline provided - numeric opportunity driver needed"
        return "Not calculated - missing baseline and numeric opportunity driver"
    if group == "Workbook opportunity":
        return (
            "Calculated from effective weekly opportunity (Column M plus Column L fallback)"
            if weekly_full > 0
            else "Needs numeric weekly opportunity"
        )
    if group in {"Performance impact", "RTT backlog impact"}:
        if metric == "Opening RTT backlog":
            return "Calculated from latest PTL data"
        if bool(row.get("_PTL_Validation", False)):
            return "Calculated from latest PTL backlog and workbook validation assumption; actual validation outcome data not available"
        return (
            "Calculated from effective weekly opportunity (Column M plus Column L fallback)"
            if weekly_full > 0
            else "Not calculated - missing numeric weekly opportunity"
        )
    if group == "Financial opportunity":
        if metric == "Column L narrative finance opportunity":
            return (
                "Calculated from Column L narrative finance opportunity"
                if pd.notna(row_finance_driver) and row_finance_driver > 0
                else "Not calculated - no material finance value found in Column L"
            )
        if metric == "Workbook annual benefit applied":
            return (
                "Calculated from workbook annual benefit"
                if annual_benefit > 0
                else "Not calculated - annual benefit field blank"
            )
        if metric == "Investment required":
            return (
                "Calculated from workbook investment field"
                if investment > 0
                else "Not calculated - investment field blank"
            )
        return (
            "Calculated from activity x finance proxy"
            if weekly_full > 0
            else "Not calculated - missing activity driver"
        )
    return "Context"


def build_intervention_metric_rows(
    theme_df: pd.DataFrame,
    opportunity_states: list[dict],
    horizon_weeks: float,
    opening_backlog: float,
) -> list[dict]:
    rows = []
    for _, row in theme_df.iterrows():
        explicit_weekly = float(row.get("Additional_Cases_Slots_Per_Week", 0) or 0)
        weekly_driver = float(row.get("Effective_Weekly_Opportunity", explicit_weekly) or 0)
        driver_source = str(row.get("Opportunity_Driver_Source", ""))
        narrative_finance = float(row.get("Narrative_Financial_Opportunity", 0) or 0)
        is_ptl_validation = is_ptl_validation_intervention(row)
        current_baseline = str(row.get("Current_Baseline", "")).strip()
        assumption = str(row.get("Assumption_Detail", "")).strip()

        if is_ptl_validation:
            validation_rate = ptl_validation_rate(row)
            state_values = {
                state_column_label(state): format_number(
                    ptl_validation_reduction(row, opening_backlog, float(state["share"]))
                )
                for state in opportunity_states
            }
            calculation = (
                f"Latest PTL backlog ({format_number(opening_backlog)}) x "
                f"workbook validation assumption ({format_percent(validation_rate or 0)}). "
                "Actual validation outcome data is not available, so this is assumption-led."
            )
            group = "RTT backlog impact"
            action = (
                "Use as a modelled PTL validation opportunity. Replace with actual "
                "validation removals if a validation tracker/export is supplied."
            )
        elif weekly_driver > 0:
            state_values = {
                state_column_label(state): format_number(
                    weekly_driver * float(state["share"]) * horizon_weeks
                )
                for state in opportunity_states
            }
            calculation = (
                f"{format_decimal(weekly_driver, 1)} modelled cases / slots per week "
                f"x state x {format_decimal(horizon_weeks, 0)} weeks. "
                f"Driver source: {driver_source}."
            )
            group = "What if more throughput"
            action = (
                "Intervention-level row from workbook. Numeric rows contribute to theme totals; "
                "blank numeric rows are retained as baseline/evidence."
            )
        elif narrative_finance > 0:
            state_values = {
                state_column_label(state): format_currency(
                    narrative_finance * float(state["share"])
                )
                for state in opportunity_states
            }
            calculation = (
                f"Column L narrative finance opportunity "
                f"({format_currency(narrative_finance)}) x state."
            )
            group = "Financial opportunity"
            action = (
                "Finance-only opportunity derived from Column L. It does not create "
                "additional case/slot activity unless an activity driver is supplied."
            )
        else:
            state_values = {
                state_column_label(state): "Needs numeric weekly opportunity"
                for state in opportunity_states
            }
            calculation = (
                "Column D baseline is shown, but `Additional Cases / Slots per Week` "
                "is blank so impact cannot be calculated yet."
            )
            group = "What if more throughput"
            action = (
                "Intervention-level row from workbook. Numeric rows contribute to theme totals; "
                "blank numeric rows are retained as baseline/evidence."
            )

        if assumption and not is_ptl_validation:
            calculation = f"{calculation} Assumption: {assumption}"

        rows.append(
            {
                "What-if lens / group": group,
                "Metric": row.get("Intervention", ""),
                "_Weekly_Driver": weekly_driver,
                "_Driver_Source": driver_source,
                "_Narrative_Finance": narrative_finance,
                "_PTL_Validation": is_ptl_validation,
                "Current baseline": current_baseline or "Not populated in Column D",
                **state_values,
                "Calculation / evidence": calculation,
                "Action / caveat": action,
            }
        )

    return rows


def build_same_throughput_rows(
    theme: str,
    theme_df: pd.DataFrame,
    opportunity_states: list[dict],
    baselines: dict,
    horizon_weeks: float,
) -> list[dict]:
    rows = []

    if is_theatre_theme(theme) and baselines["theatre"].get("available"):
        actual_240_sessions = baselines["theatre"].get("actual_240_sessions", 0)
        completed_cases = baselines["theatre"].get("completed_cases", 0)
        productivity = (
            completed_cases / actual_240_sessions if actual_240_sessions > 0 else 0
        )
        unit_label = "240-min elective sessions freed"
        calculation_suffix = (
            "Additional case volume over the horizon / baseline cases per "
            "240-min actual elective session."
        )
    elif is_outpatient_theme(theme) and baselines["outpatient"].get("available"):
        actual_sessions = baselines["outpatient"].get("actual_sessions", 0)
        attended = baselines["outpatient"].get("attended_appointments", 0)
        productivity = attended / actual_sessions if actual_sessions > 0 else 0
        unit_label = "clinic-session proxies freed"
        calculation_suffix = (
            "Additional appointment volume over the horizon / baseline attended "
            "appointments per actual clinic-session proxy."
        )
    else:
        return rows

    if productivity <= 0:
        return rows

    for _, row in theme_df.iterrows():
        weekly_driver = float(row.get("Effective_Weekly_Opportunity", 0) or 0)
        if weekly_driver <= 0:
            continue

        state_values = {
            state_column_label(state): format_number(
                (weekly_driver * float(state["share"]) * horizon_weeks)
                / productivity
            )
            for state in opportunity_states
        }
        rows.append(
            {
                "What-if lens / group": "What if same throughput, fewer sessions",
                "Metric": f"{unit_label}: {row.get('Intervention', '')}",
                "_Weekly_Driver": weekly_driver,
                "_Driver_Source": row.get("Opportunity_Driver_Source", ""),
                "Current baseline": "0",
                **state_values,
                "Calculation / evidence": calculation_suffix,
                "Action / caveat": (
                    "This is the capacity-release lens. It holds current activity constant "
                    "and translates the intervention's extra capacity into session equivalents."
                ),
            }
        )

    return rows


def build_theme_impact_table(
    theme: str,
    theme_df: pd.DataFrame,
    opportunity_states: list[dict],
    baselines: dict,
    horizon_weeks: float,
    opening_backlog: float,
    backlog_month: str,
    rtt_conversion: float,
    value_per_case_slot: float,
) -> tuple[pd.DataFrame, list[str]]:
    modelable = theme_df[theme_df["Is_Modelable"]].copy()
    weekly_full = float(modelable["Effective_Weekly_Opportunity"].fillna(0).sum())
    annual_benefit = float(modelable["Annual_Financial_Benefit"].fillna(0).sum())
    narrative_financial_full = float(
        modelable["Narrative_Financial_Opportunity"].fillna(0).sum()
    )
    investment = float(modelable["Investment_Required"].fillna(0).sum())
    has_ptl_validation = any(
        is_ptl_validation_intervention(row) for _, row in theme_df.iterrows()
    )

    outputs = {}
    for state in opportunity_states:
        label = state_column_label(state)
        share = float(state["share"])
        weekly_activity = weekly_full * share
        total_activity = weekly_activity * horizon_weeks
        validation_reduction = sum(
            ptl_validation_reduction(row, opening_backlog, share)
            for _, row in theme_df.iterrows()
            if is_ptl_validation_intervention(row)
        )
        throughput_backlog_reduction = total_activity * rtt_conversion
        backlog_reduction = min(
            throughput_backlog_reduction + validation_reduction,
            opening_backlog,
        )
        closing_backlog = max(opening_backlog - backlog_reduction, 0)
        finance_proxy = total_activity * value_per_case_slot
        narrative_finance = narrative_financial_full * share
        workbook_benefit = annual_benefit * share
        net = finance_proxy + narrative_finance + workbook_benefit - investment
        outputs[label] = {
            "share": share,
            "weekly_activity": weekly_activity,
            "total_activity": total_activity,
            "validation_reduction": validation_reduction,
            "backlog_reduction": backlog_reduction,
            "closing_backlog": closing_backlog,
            "backlog_reduction_pct": (
                backlog_reduction / opening_backlog if opening_backlog > 0 else 0
            ),
            "finance_proxy": finance_proxy,
            "narrative_finance": narrative_finance,
            "workbook_benefit": workbook_benefit,
            "investment": investment,
            "net": net,
        }

    def state_value(label: str, key: str, formatter=format_number) -> str:
        return formatter(outputs[label][key])

    state_labels = [state_column_label(state) for state in opportunity_states]
    intervention_rows = build_intervention_metric_rows(
        theme_df,
        opportunity_states,
        horizon_weeks,
        opening_backlog,
    )
    same_throughput_rows = build_same_throughput_rows(
        theme,
        theme_df,
        opportunity_states,
        baselines,
        horizon_weeks,
    )

    rows = [
        {
            "What-if lens / group": "Baseline context",
            "Metric": "Baseline period",
            "Current baseline": BASELINE_LABEL,
            **{label: BASELINE_LABEL for label in state_labels},
            "Calculation / evidence": (
                "Where raw data exists, baselines are calculated from records dated "
                f"{BASELINE_LABEL}."
            ),
            "Action / caveat": "For data gaps, the workbook baseline text is retained.",
        },
        {
            "What-if lens / group": "Baseline context",
            "Metric": "Selected theme group",
            "Current baseline": theme,
            **{label: theme for label in state_labels},
            "Calculation / evidence": "Selected from workbook `Group / Theme`.",
            "Action / caveat": "Use the dropdown to switch theme group.",
        },
        {
            "What-if lens / group": "Baseline context",
            "Metric": "Interventions in theme",
            "Current baseline": format_number(len(theme_df)),
            **{label: format_number(len(theme_df)) for label in state_labels},
            "Calculation / evidence": "Count of workbook intervention rows under the selected theme.",
            "Action / caveat": "Includes quantified and non-quantified interventions.",
        },
        {
            "What-if lens / group": "Baseline context",
            "Metric": "Modelled interventions",
            "Current baseline": format_number(len(modelable)),
            **{label: format_number(len(modelable)) for label in state_labels},
            "Calculation / evidence": (
                "Rows with a numeric Column M weekly driver, a usable Column L "
                "activity/finance assumption, or a modelled PTL validation assumption."
            ),
            "Action / caveat": (
                "Rows without a calculable driver remain as baseline/evidence only."
            ),
        },
        {
            "What-if lens / group": "Workbook opportunity",
            "Metric": "Stated full weekly opportunity",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_decimal(weekly_full, 1),
            **{label: format_decimal(weekly_full, 1) for label in state_labels},
            "Calculation / evidence": (
                "Sum of the selected theme's effective weekly driver. Column M is "
                "used first; if Column M is blank, a usable Column L activity "
                "assumption is converted into a weekly value."
            ),
            "Action / caveat": (
                "This is the full weekly opportunity before the state logic is applied."
            ),
        },
        {
            "What-if lens / group": "Performance impact",
            "Metric": f"Additional cases / slots over {format_decimal(horizon_weeks, 0)} weeks",
            "_Weekly_Driver": weekly_full,
            "Current baseline": "0",
            **{
                label: state_value(label, "total_activity") for label in state_labels
            },
            "Calculation / evidence": "Additional cases / slots per week x impact period weeks.",
            "Action / caveat": "Impact period is controlled in the sidebar.",
        },
        {
            "What-if lens / group": "RTT backlog impact",
            "Metric": "Opening RTT backlog",
            "Current baseline": format_number(opening_backlog),
            **{label: format_number(opening_backlog) for label in state_labels},
            "Calculation / evidence": f"Latest PTL backlog month: {backlog_month}.",
            "Action / caveat": "This is the starting backlog before applying opportunity impact.",
        },
        {
            "What-if lens / group": "RTT backlog impact",
            "Metric": "RTT backlog reduction",
            "_Weekly_Driver": weekly_full,
            "Current baseline": "0",
            **{
                label: state_value(label, "backlog_reduction")
                for label in state_labels
            },
            "Calculation / evidence": (
                (
                    f"Additional cases / slots x RTT conversion ({format_percent(rtt_conversion)}) "
                    "+ PTL validation opportunity from latest backlog x workbook assumption, "
                    "capped at opening backlog."
                )
                if has_ptl_validation
                else (
                    f"Additional cases / slots x RTT conversion ({format_percent(rtt_conversion)}), "
                    "capped at opening backlog."
                )
            ),
            "Action / caveat": "Validate whether each intervention genuinely removes RTT pathways.",
        },
        {
            "What-if lens / group": "RTT backlog impact",
            "Metric": "Closing RTT backlog",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_number(opening_backlog),
            **{
                label: state_value(label, "closing_backlog")
                for label in state_labels
            },
            "Calculation / evidence": "Opening RTT backlog - RTT backlog reduction.",
            "Action / caveat": "This is the modelled closing backlog position.",
        },
        {
            "What-if lens / group": "RTT backlog impact",
            "Metric": "RTT backlog reduction %",
            "_Weekly_Driver": weekly_full,
            "Current baseline": "0%",
            **{
                label: state_value(
                    label,
                    "backlog_reduction_pct",
                    format_percent,
                )
                for label in state_labels
            },
            "Calculation / evidence": "RTT backlog reduction / opening RTT backlog.",
            "Action / caveat": "Read alongside the closing backlog row.",
        },
        {
            "What-if lens / group": "Financial opportunity",
            "Metric": "Indicative finance proxy",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_currency(0),
            **{
                label: state_value(label, "finance_proxy", format_currency)
                for label in state_labels
            },
            "Calculation / evidence": f"Additional cases / slots x {format_currency(value_per_case_slot)} per case / slot.",
            "Action / caveat": "Proxy only unless Finance confirms value and cashability.",
        },
        {
            "What-if lens / group": "Financial opportunity",
            "Metric": "Column L narrative finance opportunity",
            "_Narrative_Finance": narrative_financial_full,
            "Current baseline": format_currency(0),
            **{
                label: state_value(label, "narrative_finance", format_currency)
                for label in state_labels
            },
            "Calculation / evidence": (
                "Finance value parsed from Column L narrative where the text contains "
                "a material income, saving, benefit or opportunity value."
            ),
            "Action / caveat": "Narrative finance is indicative; Finance should validate value, timing and cashability.",
        },
        {
            "What-if lens / group": "Financial opportunity",
            "Metric": "Workbook annual benefit applied",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_currency(0),
            **{
                label: state_value(label, "workbook_benefit", format_currency)
                for label in state_labels
            },
            "Calculation / evidence": "Numeric `Annual Financial Benefit (£)` x opportunity level.",
            "Action / caveat": "Most workbook finance fields are narrative, so this may be zero.",
        },
        {
            "What-if lens / group": "Financial opportunity",
            "Metric": "Investment required",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_currency(investment),
            **{
                label: state_value(label, "investment", format_currency)
                for label in state_labels
            },
            "Calculation / evidence": "Sum of numeric `Investment Required (£)` for selected theme.",
            "Action / caveat": "Currently zero if the workbook does not hold numeric investment values.",
        },
        {
            "What-if lens / group": "Financial opportunity",
            "Metric": "Indicative net position",
            "_Weekly_Driver": weekly_full,
            "Current baseline": format_currency(0),
            **{
                label: state_value(label, "net", format_currency)
                for label in state_labels
            },
            "Calculation / evidence": (
                "Finance proxy + Column L narrative finance opportunity + workbook "
                "annual benefit applied - investment required."
            ),
            "Action / caveat": "Do not treat as cashable until Finance validates overlap and benefit type.",
        },
    ]

    if is_theatre_theme(theme):
        theatre = baselines["theatre"]
        if theatre.get("available"):
            cases_per_session = (
                theatre["completed_cases"] / theatre["actual_240_sessions"]
                if theatre["actual_240_sessions"] > 0
                else 0
            )
            theatre_rows = [
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Current elective theatre utilisation",
                    "Current baseline": format_percent(theatre["utilisation"]),
                    **{
                        label: format_percent(state["target_value"])
                        for label, state in zip(state_labels, opportunity_states)
                    },
                    "Calculation / evidence": (
                        "Model Hospital style basis: elective, non-obstetric, "
                        "actual sessions only; touch minutes / scheduled minutes."
                    ),
                    "Action / caveat": "Validate elective/emergency and obstetric exclusions with PAH.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Actual elective sessions, 240-min equivalents",
                    "Current baseline": format_number(theatre["actual_240_sessions"]),
                    **{label: format_number(theatre["actual_240_sessions"]) for label in state_labels},
                    "Calculation / evidence": (
                        "Actual elective session scheduled minutes / 240 from "
                        f"{BASELINE_LABEL}."
                    ),
                    "Action / caveat": "Held constant in the throughput view.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Completed elective cases",
                    "Current baseline": format_number(theatre["completed_cases"]),
                    **{label: format_number(theatre["completed_cases"]) for label in state_labels},
                    "Calculation / evidence": f"Completed elective cases in {BASELINE_LABEL}.",
                    "Action / caveat": "Scenario rows show additional volume above this baseline.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Cases per 240-min actual session",
                    "Current baseline": format_decimal(cases_per_session, 2),
                    **{label: format_decimal(cases_per_session, 2) for label in state_labels},
                    "Calculation / evidence": "Completed cases / actual 240-min session equivalents.",
                    "Action / caveat": "Used as context; workbook weekly opportunity drives the state impact.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Average case duration",
                    "Current baseline": f"{format_decimal(theatre['avg_case_duration'], 1)} mins",
                    **{
                        label: f"{format_decimal(theatre['avg_case_duration'], 1)} mins"
                        for label in state_labels
                    },
                    "Calculation / evidence": "Touch minutes / cases with valid touch time.",
                    "Action / caveat": "Use to sense-check additional case volumes.",
                },
            ]
            rows = rows[:4] + theatre_rows + rows[4:]
        else:
            rows.insert(
                4,
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Theatre baseline data",
                    "Current baseline": "Not available",
                    **{label: "Not available" for label in state_labels},
                    "Calculation / evidence": theatre.get("error", "No theatre data loaded."),
                    "Action / caveat": "Use workbook baseline text until the raw data issue is resolved.",
                },
            )

    elif is_outpatient_theme(theme):
        outpatient = baselines["outpatient"]
        if outpatient.get("available"):
            attended_per_week = outpatient["attended_appointments"] / outpatient["observed_weeks"]
            planned_per_week = outpatient["planned_appointments"] / outpatient["observed_weeks"]
            actual_sessions_per_week = outpatient["actual_sessions"] / outpatient["observed_weeks"]
            outpatient_rows = [
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Planned appointment records",
                    "Current baseline": format_number(outpatient["planned_appointments"]),
                    **{label: format_number(outpatient["planned_appointments"]) for label in state_labels},
                    "Calculation / evidence": f"Unique Contact_ID records in {BASELINE_LABEL}.",
                    "Action / caveat": "This is booked/planned contact records, not full template capacity.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Actual attended appointments",
                    "Current baseline": format_number(outpatient["attended_appointments"]),
                    **{
                        label: format_number(
                            outpatient["attended_appointments"]
                            + outputs[label]["total_activity"]
                        )
                        for label in state_labels
                    },
                    "Calculation / evidence": (
                        "Baseline = unique Contact_ID records with Checked In or "
                        "Checked Out status. State values add calculated extra slots."
                    ),
                    "Action / caveat": "This is the operational activity state, not a percentage label.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Average attended appointments / week",
                    "Current baseline": format_number(attended_per_week),
                    **{
                        label: format_number(attended_per_week + outputs[label]["weekly_activity"])
                        for label in state_labels
                    },
                    "Calculation / evidence": (
                        "Attended appointments / observed weeks, plus each state "
                        "additional appointment volume per week."
                    ),
                    "Action / caveat": "Use this row to explain what the state means in real patients.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Booked appointment fill / attendance proxy",
                    "Current baseline": format_percent(outpatient["fill_rate"]),
                    **{label: format_percent(outpatient["fill_rate"]) for label in state_labels},
                    "Calculation / evidence": (
                        "Attended Contact_ID records / planned Contact_ID records. "
                        "True template fill cannot be measured without empty slot/template data."
                    ),
                    "Action / caveat": "Do not present this as true template fill.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "DNA / no-show rate",
                    "Current baseline": format_percent(outpatient["dna_rate"]),
                    **{label: format_percent(outpatient["dna_rate"]) for label in state_labels},
                    "Calculation / evidence": "No Show Contact_ID records / planned Contact_ID records.",
                    "Action / caveat": "Use with DNA-specific intervention rows in the detail table.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Actual clinic-session proxy / week",
                    "Current baseline": format_number(actual_sessions_per_week),
                    **{label: format_number(actual_sessions_per_week) for label in state_labels},
                    "Calculation / evidence": (
                        "Clinic/performance unit + date + AM/PM with at least one "
                        "attended appointment."
                    ),
                    "Action / caveat": "Held constant where the state is more throughput.",
                },
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Planned appointment records / week",
                    "Current baseline": format_number(planned_per_week),
                    **{label: format_number(planned_per_week) for label in state_labels},
                    "Calculation / evidence": "Planned appointment records / observed weeks.",
                    "Action / caveat": "Context only; state impact uses workbook additional slots/week.",
                },
            ]
            rows = rows[:4] + outpatient_rows + rows[4:]
        else:
            rows.insert(
                4,
                {
                    "What-if lens / group": "Data-led baseline",
                    "Metric": "Outpatient baseline data",
                    "Current baseline": "Not available",
                    **{label: "Not available" for label in state_labels},
                    "Calculation / evidence": outpatient.get("error", "No outpatient data loaded."),
                    "Action / caveat": "Use workbook baseline text until the raw data issue is resolved.",
                },
            )

    else:
        workbook_baseline_text = "; ".join(
            sorted(
                {
                    value
                    for value in theme_df["Current_Baseline"].fillna("").astype(str)
                    if value.strip()
                }
            )[:3]
        )
        rows.insert(
            4,
            {
                "What-if lens / group": "Workbook baseline",
                "Metric": "Current baseline evidence",
                "Current baseline": workbook_baseline_text or "Not quantified in workbook",
                **{label: workbook_baseline_text or "Not quantified in workbook" for label in state_labels},
                "Calculation / evidence": (
                    "Uses workbook Column D: `Current Baseline (PAH position)`. "
                    "No matching raw Apr 2025-Mar 2026 baseline loader exists yet for this theme."
                ),
                "Action / caveat": "Use Column D as baseline evidence; add raw-data calculation when the relevant extract is available.",
            },
        )

    summary_df = pd.DataFrame(rows)
    insert_at = summary_df.index[
        summary_df["What-if lens / group"] == "Workbook opportunity"
    ].min()
    if pd.isna(insert_at):
        insert_at = len(summary_df)
    intervention_df = pd.DataFrame([*intervention_rows, *same_throughput_rows])
    if not intervention_df.empty:
        summary_df = pd.concat(
            [
                summary_df.iloc[: int(insert_at)],
                intervention_df,
                summary_df.iloc[int(insert_at) :],
            ],
            ignore_index=True,
            sort=False,
        )

    summary_df.insert(
        2,
        "Calculation flag",
        summary_df.apply(
            lambda row: summary_calculation_flag(
                row,
                weekly_full,
                annual_benefit,
                investment,
            ),
            axis=1,
        ),
    )
    hidden_cols = [col for col in summary_df.columns if str(col).startswith("_")]
    if hidden_cols:
        summary_df = summary_df.drop(columns=hidden_cols)

    return summary_df, state_labels


default_path = find_default_workbook()

with st.sidebar:
    st.header("Opportunity Setup")
    source_path = st.text_input("Opportunity model file", value=str(default_path))
    horizon_weeks = st.number_input(
        "Impact period weeks",
        min_value=1.0,
        max_value=104.0,
        value=43.0,
        step=1.0,
    )
    rtt_conversion = st.slider(
        "RTT conversion",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
        help="Share of additional cases/slots assumed to remove an RTT pathway.",
    ) / 100
    value_per_case_slot = st.number_input(
        "Finance proxy per case / slot",
        min_value=0.0,
        max_value=25_000.0,
        value=250.0,
        step=50.0,
    )

try:
    opportunity_df = load_opportunity_model(source_path)
except Exception as exc:
    st.error(f"Could not load opportunity model: {exc}")
    st.stop()

opening_backlog, backlog_month = get_latest_backlog()
available_baselines = load_available_baselines()

themes = sorted(
    [
        theme
        for theme in opportunity_df["Group_Theme"].dropna().unique().tolist()
        if str(theme).strip()
    ]
)

if not themes:
    st.warning("No theme groups found in the opportunity model.")
    st.stop()

selected_theme = st.selectbox("Select group / theme", themes)
theme_df = opportunity_df[opportunity_df["Group_Theme"] == selected_theme].copy()

modelable_theme_df = theme_df[theme_df["Is_Modelable"]].copy()
weekly_opportunity = float(
    modelable_theme_df["Effective_Weekly_Opportunity"].fillna(0).sum()
)
finance_opportunity = float(
    modelable_theme_df["Narrative_Financial_Opportunity"].fillna(0).sum()
)
opportunity_states = build_opportunity_states(
    selected_theme,
    weekly_opportunity,
    available_baselines,
    finance_opportunity,
)

theme_impact_df, state_columns = build_theme_impact_table(
    theme=selected_theme,
    theme_df=theme_df,
    opportunity_states=opportunity_states,
    baselines=available_baselines,
    horizon_weeks=horizon_weeks,
    opening_backlog=opening_backlog,
    backlog_month=backlog_month,
    rtt_conversion=rtt_conversion,
    value_per_case_slot=value_per_case_slot,
)
intervention_impact_df = build_intervention_impact_table(
    theme=selected_theme,
    theme_df=theme_df,
    opportunity_states=opportunity_states,
    horizon_weeks=horizon_weeks,
    value_per_case_slot=value_per_case_slot,
    opening_backlog=opening_backlog,
)

st.caption(
    f"RTT backlog source: latest PTL month {backlog_month}. Finance is an "
    "indicative proxy unless a numeric benefit is provided in the workbook."
)

st.subheader("Selected Theme Impact Table")
theme_table_config = {
    "What-if lens / group": st.column_config.TextColumn(
        "What-if lens / group",
        width="medium",
    ),
    "Metric": st.column_config.TextColumn("Metric", width="medium"),
    "Calculation flag": st.column_config.TextColumn(
        "Calculation flag",
        width="medium",
    ),
    "Current baseline": st.column_config.TextColumn(
        "Current baseline",
        width="medium",
    ),
    "Calculation / evidence": st.column_config.TextColumn(
        "Calculation / evidence",
        width="large",
    ),
    "Action / caveat": st.column_config.TextColumn(
        "Action / caveat",
        width="large",
    ),
}

theme_sections = [
    (
        "Baseline And Evidence",
        ["Baseline context", "Data-led baseline", "Workbook baseline"],
    ),
    (
        "Intervention And Activity Impact",
        [
            "What if more throughput",
            "What if same throughput, fewer sessions",
            "Workbook opportunity",
            "Performance impact",
        ],
    ),
    (
        "RTT And Financial Impact",
        ["RTT backlog impact", "Financial opportunity"],
    ),
]

for section_label, section_groups in theme_sections:
    section_df = theme_impact_df[
        theme_impact_df["What-if lens / group"].isin(section_groups)
    ].copy()
    if section_df.empty:
        continue

    st.markdown(f"**{section_label}**")
    st.dataframe(
        section_df,
        use_container_width=True,
        hide_index=True,
        column_config=theme_table_config,
    )

st.download_button(
    "Download selected theme summary table",
    data=theme_impact_df.to_csv(index=False).encode("utf-8"),
    file_name=f"opportunity_summary_{selected_theme.replace(' ', '_')}.csv",
    mime="text/csv",
)

st.subheader("Intervention Detail Table")
st.caption(
    "This table shows all interventions in the selected theme. Rows without a "
    "numeric weekly opportunity are retained as workbook baseline/narrative rows."
)
impact_display = intervention_impact_df.copy()
numeric_cols = [
    "Additional cases / slots per week",
    "Modelled cases / slots per week",
    "Column L narrative finance",
]
for state_col in state_columns:
    numeric_cols.extend(
        [
            f"{state_col}: total activity",
            f"{state_col}: finance proxy",
        ]
    )

for col in numeric_cols:
    if col in impact_display.columns and "finance" in col.lower():
        impact_display[col] = impact_display[col].map(format_currency)
    elif col in impact_display.columns:
        impact_display[col] = impact_display[col].map(format_number)

st.dataframe(
    impact_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Intervention": st.column_config.TextColumn("Intervention", width="large"),
        "Calculation flag": st.column_config.TextColumn(
            "Calculation flag",
            width="medium",
        ),
        "Baseline source": st.column_config.TextColumn(
            "Baseline source",
            width="medium",
        ),
        "Opportunity driver source": st.column_config.TextColumn(
            "Opportunity driver source",
            width="medium",
        ),
        "Assumption detail": st.column_config.TextColumn(
            "Assumption detail",
            width="large",
        ),
        "Current baseline": st.column_config.TextColumn(
            "Current baseline",
            width="medium",
        ),
        "Benchmark / target": st.column_config.TextColumn(
            "Benchmark / target",
            width="medium",
        ),
        "PAH target": st.column_config.TextColumn("PAH target", width="medium"),
        "Finance category": st.column_config.TextColumn(
            "Finance category",
            width="medium",
        ),
        "Finance opportunity narrative": st.column_config.TextColumn(
            "Finance opportunity narrative",
            width="large",
        ),
    },
)

st.download_button(
    "Download selected theme intervention table",
    data=impact_display.to_csv(index=False).encode("utf-8"),
    file_name=f"opportunity_interventions_{selected_theme.replace(' ', '_')}.csv",
    mime="text/csv",
)

with st.expander("Calculation Notes", expanded=True):
    st.markdown(
        f"""
- Theme group comes from the workbook `Group / Theme` column and the dropdown controls which theme is shown.
- Baseline period is fixed to {BASELINE_LABEL}. Theatre and outpatient baseline rows are calculated from the raw data where available.
- State columns are operational labels rather than delivery percentages: {", ".join(state_columns)}.
- For Theatre Productivity, the states are utilisation targets and the current baseline is the measured elective theatre utilisation.
- For Outpatients, the states are additional appointment volumes per week, with actual attended appointments, DNA rate and attendance/fill proxy calculated from raw outpatient records.
- For themes without a numeric weekly case/slot value, the state columns become evidence/action columns until the opportunity is quantified.
- Column D from the workbook is used as the workbook-provided baseline where a raw-data calculation is not available.
- The `Calculation flag` column shows whether the row has been calculated from raw data, calculated from the workbook driver, or needs more source data.
- The main quantified driver is `Additional Cases / Slots per Week`; if that is blank, Column L is parsed for usable numeric assumptions such as cases/week, slots/year, patients/year, ten-month activity, or material finance values.
- Column L-derived annual activity is converted to a weekly driver using {format_decimal(ANNUAL_WEEKS, 0)} weeks; ten-month activity is converted using {format_decimal(TEN_MONTH_WEEKS, 0)} weeks.
- Additional cases / slots over {format_decimal(horizon_weeks, 0)} weeks = additional cases / slots per week x impact period weeks.
- RTT backlog reduction = additional cases / slots over {format_decimal(horizon_weeks, 0)} weeks x RTT conversion ({format_percent(rtt_conversion)}), capped at the opening RTT backlog.
- Closing RTT backlog = opening RTT backlog - modelled RTT backlog reduction.
- Indicative finance proxy = additional cases / slots x {format_currency(value_per_case_slot)} per case / slot.
- Workbook finance benefit is only used where a numeric `Annual Financial Benefit (£)` is populated; most rows currently hold finance as narrative rather than a numeric value.
        """
    )
