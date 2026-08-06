import pandas as pd


def filter_referrals(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
    specialties=None,
    sources=None,
    priorities=None,
    ccgs=None,
) -> pd.DataFrame:
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[
            filtered["Referral_Received_Date"] >= pd.to_datetime(start_date)
        ]

    if end_date is not None:
        filtered = filtered[
            filtered["Referral_Received_Date"] <= pd.to_datetime(end_date)
        ]

    if specialties:
        filtered = filtered[filtered["Standardised_Specialty"].isin(specialties)]

    if sources and "ReferralSource" in filtered.columns:
        filtered = filtered[filtered["ReferralSource"].isin(sources)]

    if priorities and "Medical_Priority_Desc" in filtered.columns:
        filtered = filtered[filtered["Medical_Priority_Desc"].isin(priorities)]

    if ccgs and "CCG" in filtered.columns:
        filtered = filtered[filtered["CCG"].isin(ccgs)]

    return filtered


def summarise_referrals_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Referral_Month", "Referrals"])

    return (
        df.groupby("Referral_Month")
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .sort_values("Referral_Month")
    )


def summarise_referrals_by_specialty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Standardised_Specialty", "Referrals"])

    return (
        df.groupby("Standardised_Specialty")
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .sort_values("Referrals", ascending=False)
    )


def summarise_referrals_by_ccg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["CCG", "Referrals"])

    if "CCG" in df.columns:
        ccg_col = "CCG"
    elif "CCGShortCode" in df.columns:
        ccg_col = "CCGShortCode"
    else:
        return pd.DataFrame(columns=["CCG", "Referrals"])

    return (
        df.groupby(ccg_col)
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .rename(columns={ccg_col: "CCG"})
        .sort_values("Referrals", ascending=False)
    )


def summarise_referrals_by_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ReferralSource" not in df.columns:
        return pd.DataFrame(columns=["ReferralSource", "Referrals"])

    return (
        df.groupby("ReferralSource")
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .sort_values("Referrals", ascending=False)
    )


def summarise_referrals_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Priority", "Referrals"])

    if "Graded_Med_Priority_Desc" in df.columns:
        priority_col = "Graded_Med_Priority_Desc"
    elif "Medical_Priority_Desc" in df.columns:
        priority_col = "Medical_Priority_Desc"
    else:
        return pd.DataFrame(columns=["Priority", "Referrals"])

    return (
        df.groupby(priority_col)
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .rename(columns={priority_col: "Priority"})
        .sort_values("Referrals", ascending=False)
    )


def summarise_referrals_by_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ReferralType" not in df.columns:
        return pd.DataFrame(columns=["ReferralType", "Referrals"])

    return (
        df.groupby("ReferralType")
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
        .sort_values("Referrals", ascending=False)
    )


def add_monthly_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()

    if df.empty:
        return df

    df["Previous_Month_Referrals"] = df["Referrals"].shift(1)
    df["Monthly_Change"] = df["Referrals"] - df["Previous_Month_Referrals"]

    df["Monthly_Growth_%"] = (
        df["Monthly_Change"] / df["Previous_Month_Referrals"] * 100
    ).round(1)

    return df


def referral_growth_signal(
    monthly_df: pd.DataFrame,
    baseline_months: int = 6,
) -> dict:
    df = monthly_df.sort_values("Referral_Month").copy()

    if df.empty:
        return {
            "latest": 0,
            "baseline": 0,
            "change": 0,
            "change_pct": 0,
            "signal": "No data",
        }

    latest = df.iloc[-1]["Referrals"]

    if len(df) >= baseline_months:
        baseline = df.tail(baseline_months)["Referrals"].mean()
    else:
        baseline = df["Referrals"].mean()

    change = latest - baseline
    change_pct = (change / baseline * 100) if baseline else 0

    if change_pct >= 10:
        signal = "High demand growth"
    elif change_pct >= 5:
        signal = "Moderate demand growth"
    elif change_pct <= -10:
        signal = "Demand falling"
    else:
        signal = "Stable demand"

    return {
        "latest": round(latest, 0),
        "baseline": round(baseline, 1),
        "change": round(change, 1),
        "change_pct": round(change_pct, 1),
        "signal": signal,
    }


def top_specialty_growth(
    df: pd.DataFrame,
    recent_months: int = 3,
    baseline_months: int = 6,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Standardised_Specialty",
                "Recent_Referrals",
                "Baseline_Referrals",
                "Change",
                "Growth_%",
            ]
        )

    work = df.copy()
    max_month = work["Referral_Month"].max()

    recent_start = max_month - pd.DateOffset(months=recent_months - 1)
    baseline_start = recent_start - pd.DateOffset(months=baseline_months)

    recent = work[work["Referral_Month"] >= recent_start]

    baseline = work[
        (work["Referral_Month"] >= baseline_start)
        & (work["Referral_Month"] < recent_start)
    ]

    recent_summary = (
        recent.groupby("Standardised_Specialty")
        .agg(Recent_Referrals=("Referral_ID", "nunique"))
        .reset_index()
    )

    baseline_summary = (
        baseline.groupby("Standardised_Specialty")
        .agg(Baseline_Referrals=("Referral_ID", "nunique"))
        .reset_index()
    )

    result = recent_summary.merge(
        baseline_summary,
        on="Standardised_Specialty",
        how="left",
    )

    result["Baseline_Referrals"] = result["Baseline_Referrals"].fillna(0)
    result["Change"] = result["Recent_Referrals"] - result["Baseline_Referrals"]

    result["Growth_%"] = result.apply(
        lambda row: round(row["Change"] / row["Baseline_Referrals"] * 100, 1)
        if row["Baseline_Referrals"] > 0
        else None,
        axis=1,
    )

    return result.sort_values("Change", ascending=False)

def referral_heatmap_matrix(df: pd.DataFrame, top_n: int = 20):
    """
    Create specialty x month referral matrix for heatmap visualisation.
    """

    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["Standardised_Specialty", "Referral_Month"])
        .agg(Referrals=("Referral_ID", "nunique"))
        .reset_index()
    )

    top_specialties = (
        summary.groupby("Standardised_Specialty")["Referrals"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    summary = summary[summary["Standardised_Specialty"].isin(top_specialties)]

    heatmap_df = summary.pivot(
        index="Standardised_Specialty",
        columns="Referral_Month",
        values="Referrals",
    )

    heatmap_df = heatmap_df.fillna(0)

    return heatmap_df

