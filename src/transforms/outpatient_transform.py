import pandas as pd


CONTACT_VISIT_TYPE_FIELD = "ContactVisitType_Group"


def filter_outpatients(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
    specialties=None,
    clinics=None,
    clinic_types=None,
    contact_types=None,
    visit_types=None,
    statuses=None,
) -> pd.DataFrame:
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[filtered["Contact_Start"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[filtered["Contact_Start"] <= pd.to_datetime(end_date)]

    if specialties:
        filtered = filtered[filtered["Standardised_Specialty"].isin(specialties)]

    if clinics and "ContactClinicPerfUnit" in filtered.columns:
        filtered = filtered[filtered["ContactClinicPerfUnit"].isin(clinics)]

    if clinic_types and "ContactClinicPerfUnit_Type" in filtered.columns:
        filtered = filtered[filtered["ContactClinicPerfUnit_Type"].isin(clinic_types)]

    if contact_types and "Type" in filtered.columns:
        filtered = filtered[filtered["Type"].isin(contact_types)]

    visit_type_col = (
        CONTACT_VISIT_TYPE_FIELD
        if CONTACT_VISIT_TYPE_FIELD in filtered.columns
        else "ContactVisitType"
    )

    if visit_types and visit_type_col in filtered.columns:
        filtered = filtered[filtered[visit_type_col].isin(visit_types)]

    if statuses and "Status" in filtered.columns:
        filtered = filtered[filtered["Status"].isin(statuses)]

    return filtered


def summarise_outpatients_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Contact_Month", "Contacts"])

    return (
        df.groupby("Contact_Month")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contact_Month")
    )


def summarise_outpatients_by_specialty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Standardised_Specialty", "Contacts"])

    return (
        df.groupby("Standardised_Specialty")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contacts", ascending=False)
    )


def summarise_outpatients_by_clinic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ContactClinicPerfUnit" not in df.columns:
        return pd.DataFrame(columns=["ContactClinicPerfUnit", "Contacts"])

    return (
        df.groupby("ContactClinicPerfUnit")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contacts", ascending=False)
    )


def summarise_outpatients_by_clinic_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ContactClinicPerfUnit_Type" not in df.columns:
        return pd.DataFrame(columns=["ContactClinicPerfUnit_Type", "Contacts"])

    return (
        df.groupby("ContactClinicPerfUnit_Type")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contacts", ascending=False)
    )


def summarise_outpatients_by_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Type" not in df.columns:
        return pd.DataFrame(columns=["Type", "Contacts"])

    return (
        df.groupby("Type")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contacts", ascending=False)
    )


def summarise_outpatients_by_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Status" not in df.columns:
        return pd.DataFrame(columns=["Status", "Contacts"])

    return (
        df.groupby("Status")
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .sort_values("Contacts", ascending=False)
    )


def summarise_outpatients_by_visit_type(df: pd.DataFrame) -> pd.DataFrame:
    visit_type_col = (
        CONTACT_VISIT_TYPE_FIELD
        if CONTACT_VISIT_TYPE_FIELD in df.columns
        else "ContactVisitType"
    )

    if df.empty or visit_type_col not in df.columns:
        return pd.DataFrame(columns=["ContactVisitType", "Contacts"])

    return (
        df.groupby(visit_type_col)
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
        .rename(columns={visit_type_col: "ContactVisitType"})
        .sort_values("Contacts", ascending=False)
    )


def outpatient_heatmap_matrix(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["Standardised_Specialty", "Contact_Month"])
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
    )

    top_specialties = (
        summary.groupby("Standardised_Specialty")["Contacts"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    summary = summary[summary["Standardised_Specialty"].isin(top_specialties)]

    heatmap_df = summary.pivot(
        index="Standardised_Specialty",
        columns="Contact_Month",
        values="Contacts",
    ).fillna(0)

    heatmap_df = heatmap_df.loc[
        heatmap_df.sum(axis=1).sort_values(ascending=False).index
    ]

    return heatmap_df


def add_monthly_contact_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()

    if df.empty:
        return df

    df["Previous_Month_Contacts"] = df["Contacts"].shift(1)
    df["Monthly_Change"] = df["Contacts"] - df["Previous_Month_Contacts"]
    df["Monthly_Growth_%"] = (
        df["Monthly_Change"] / df["Previous_Month_Contacts"] * 100
    ).round(1)

    return df


def outpatient_growth_signal(
    monthly_df: pd.DataFrame,
    baseline_months: int = 6,
) -> dict:
    df = monthly_df.sort_values("Contact_Month").copy()

    if df.empty:
        return {
            "latest": 0,
            "baseline": 0,
            "change": 0,
            "change_pct": 0,
            "signal": "No data",
        }

    latest = df.iloc[-1]["Contacts"]

    if len(df) >= baseline_months:
        baseline = df.tail(baseline_months)["Contacts"].mean()
    else:
        baseline = df["Contacts"].mean()

    change = latest - baseline
    change_pct = (change / baseline * 100) if baseline else 0

    if change_pct >= 10:
        signal = "High activity growth"
    elif change_pct >= 5:
        signal = "Moderate activity growth"
    elif change_pct <= -10:
        signal = "Activity falling"
    else:
        signal = "Stable activity"

    return {
        "latest": round(latest, 0),
        "baseline": round(baseline, 1),
        "change": round(change, 1),
        "change_pct": round(change_pct, 1),
        "signal": signal,
    }


def summarise_checked_flow_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise checked-in and checked-out outpatient activity by month.
    """

    if df.empty or "Status" not in df.columns:
        return pd.DataFrame(
            columns=[
                "Contact_Month",
                "Checked In",
                "Checked Out",
                "Flow Gap",
            ]
        )

    work = df.copy()

    work["Status_Clean"] = (
        work["Status"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def classify_status(value: str) -> str:
        if "checked in" in value or "check in" in value:
            return "Checked In"
        if "checked out" in value or "check out" in value:
            return "Checked Out"
        return "Other"

    work["Flow_Status"] = work["Status_Clean"].apply(classify_status)

    work = work[work["Flow_Status"].isin(["Checked In", "Checked Out"])]

    if work.empty:
        return pd.DataFrame(
            columns=[
                "Contact_Month",
                "Checked In",
                "Checked Out",
                "Flow Gap",
            ]
        )

    summary = (
        work.groupby(["Contact_Month", "Flow_Status"])
        .agg(Contacts=("Contact_ID", "nunique"))
        .reset_index()
    )

    pivot = summary.pivot(
        index="Contact_Month",
        columns="Flow_Status",
        values="Contacts",
    ).fillna(0)

    if "Checked In" not in pivot.columns:
        pivot["Checked In"] = 0

    if "Checked Out" not in pivot.columns:
        pivot["Checked Out"] = 0

    pivot = pivot.reset_index()
    pivot["Flow Gap"] = pivot["Checked In"] - pivot["Checked Out"]

    return pivot.sort_values("Contact_Month")


def summarise_outpatient_attendances_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Outpatient attendances = Checked In + Checked Out.
    """

    if df.empty or "Status" not in df.columns:
        return pd.DataFrame(columns=["Contact_Month", "Outpatient Attendances"])

    work = df.copy()

    status_clean = (
        work["Status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    attendance_mask = (
        status_clean.str.contains("checked in")
        | status_clean.str.contains("check in")
        | status_clean.str.contains("checked out")
        | status_clean.str.contains("check out")
    )

    work = work[attendance_mask]

    if work.empty:
        return pd.DataFrame(columns=["Contact_Month", "Outpatient Attendances"])

    return (
        work.groupby("Contact_Month")
        .agg(**{"Outpatient Attendances": ("Contact_ID", "nunique")})
        .reset_index()
        .sort_values("Contact_Month")
    )


def summarise_outpatient_attendances_by_month_visit_type(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise attended outpatient contacts by month and grouped ContactVisitType.
    """

    visit_type_col = (
        CONTACT_VISIT_TYPE_FIELD
        if CONTACT_VISIT_TYPE_FIELD in df.columns
        else "ContactVisitType"
    )

    required_cols = {"Status", visit_type_col}

    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "Contact_Month",
                "ContactVisitType",
                "Outpatient Attendances",
            ]
        )

    work = df.copy()

    status_clean = (
        work["Status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    attendance_mask = (
        status_clean.str.contains("checked in")
        | status_clean.str.contains("check in")
        | status_clean.str.contains("checked out")
        | status_clean.str.contains("check out")
    )

    work = work[attendance_mask]

    if work.empty:
        return pd.DataFrame(
            columns=[
                "Contact_Month",
                "ContactVisitType",
                "Outpatient Attendances",
            ]
        )

    work["ContactVisitType"] = (
        work[visit_type_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    return (
        work.groupby(["Contact_Month", "ContactVisitType"])
        .agg(**{"Outpatient Attendances": ("Contact_ID", "nunique")})
        .reset_index()
        .sort_values(["Contact_Month", "ContactVisitType"])
    )
