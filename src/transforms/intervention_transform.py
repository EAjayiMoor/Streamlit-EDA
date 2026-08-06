import pandas as pd


def filter_interventions(
    df: pd.DataFrame,
    groups=None,
    modelable_only: bool = False,
) -> pd.DataFrame:
    filtered = df.copy()

    if groups:
        filtered = filtered[filtered["Group_Theme"].isin(groups)]

    if modelable_only and "Is_Modelable" in filtered.columns:
        filtered = filtered[filtered["Is_Modelable"]]

    return filtered


def enrich_intervention_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds narrative modelling fields if they do not already exist.
    """

    work = df.copy()

    if "Impact_Type" not in work.columns:
        work["Impact_Type"] = work["Is_Modelable"].apply(
            lambda x: "Direct quantified" if x else "Enabling / indirect"
        )

    if "Assumption_Source" not in work.columns:
        work["Assumption_Source"] = "Opportunity model / service assumption"

    if "Logic_Chain" not in work.columns:
        work["Logic_Chain"] = work.apply(
            lambda row: (
                f"If PAH implements '{row['Intervention']}', then the operating model is expected "
                f"to change in line with the stated assumption. This creates an estimated "
                f"{row['Additional_Cases_Per_Week']:,.1f} additional cases or slots per week "
                f"over {row['Weeks_To_Recover']:,.0f} weeks, contributing to the PAH target: "
                f"{row['PAH_Target']}."
                if row.get("Is_Modelable", False)
                else (
                    f"If PAH implements '{row['Intervention']}', then this is expected to enable "
                    f"improved operational control, pathway management or decision-making. The impact "
                    f"is currently indirect or enabling unless a quantified activity assumption is added."
                )
            ),
            axis=1,
        )

    return work


def summarise_intervention_library(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_interventions": 0,
            "modelable_interventions": 0,
            "total_cases_per_week": 0,
            "total_annualised_activity": 0,
            "total_recovery_cases": 0,
            "total_investment": 0,
        }

    modelable_df = df[df["Is_Modelable"]] if "Is_Modelable" in df.columns else df

    total_investment = (
        df["Investment_Required"].sum()
        if "Investment_Required" in df.columns
        else 0
    )

    return {
        "total_interventions": len(df),
        "modelable_interventions": len(modelable_df),
        "total_cases_per_week": modelable_df["Additional_Cases_Per_Week"].sum(),
        "total_annualised_activity": modelable_df[
            "Annualised_Additional_Activity"
        ].sum(),
        "total_recovery_cases": modelable_df["Total_Additional_Cases"].sum(),
        "total_investment": total_investment,
    }


def calculate_selected_intervention_impact(
    intervention_df: pd.DataFrame,
    selected_interventions: list[str],
) -> dict:
    if intervention_df.empty or not selected_interventions:
        return {
            "selected_count": 0,
            "selected_cases_per_week": 0,
            "selected_monthly_activity": 0,
            "selected_annualised_activity": 0,
            "selected_total_recovery_cases": 0,
            "selected_max_recovery_weeks": 0,
            "selected_investment": 0,
            "selected_df": pd.DataFrame(),
        }

    selected_df = intervention_df[
        intervention_df["Intervention"].isin(selected_interventions)
    ].copy()

    selected_df = enrich_intervention_logic(selected_df)

    direct_df = selected_df[selected_df["Is_Modelable"]].copy()

    selected_cases_per_week = direct_df["Additional_Cases_Per_Week"].sum()
    selected_monthly_activity = selected_cases_per_week * 4.33
    selected_annualised_activity = selected_cases_per_week * 52
    selected_total_recovery_cases = direct_df["Total_Additional_Cases"].sum()

    if not direct_df.empty:
        selected_max_recovery_weeks = direct_df["Weeks_To_Recover"].max()
    else:
        selected_max_recovery_weeks = 0

    if "Investment_Required" in direct_df.columns:
        selected_investment = direct_df["Investment_Required"].sum()
    else:
        selected_investment = 0

    return {
        "selected_count": len(selected_df),
        "selected_cases_per_week": selected_cases_per_week,
        "selected_monthly_activity": selected_monthly_activity,
        "selected_annualised_activity": selected_annualised_activity,
        "selected_total_recovery_cases": selected_total_recovery_cases,
        "selected_max_recovery_weeks": selected_max_recovery_weeks,
        "selected_investment": selected_investment,
        "selected_df": selected_df,
    }


def prepare_intervention_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = enrich_intervention_logic(df)

    display_cols = [
        "Group_Theme",
        "Intervention",
        "Impact_Type",
        "Current_Baseline",
        "Benchmark",
        "PAH_Target",
        "Assumption_Detail",
        "Assumption_Source",
        "Additional_Cases_Per_Week",
        "Weeks_To_Recover",
        "Total_Additional_Cases",
        "Annualised_Additional_Activity",
        "Investment_Required",
        "Is_Modelable",
    ]

    available_cols = [col for col in display_cols if col in work.columns]

    display_df = work[available_cols].copy()

    rename_map = {
        "Group_Theme": "Group / Theme",
        "Impact_Type": "Impact Type",
        "Current_Baseline": "Current Baseline",
        "PAH_Target": "PAH Target",
        "Assumption_Detail": "Assumption Detail",
        "Assumption_Source": "Assumption Source",
        "Additional_Cases_Per_Week": "Additional Cases / Week",
        "Weeks_To_Recover": "Weeks to Recover",
        "Total_Additional_Cases": "Total Additional Cases",
        "Annualised_Additional_Activity": "Annualised Additional Activity",
        "Investment_Required": "Investment Required (£)",
        "Is_Modelable": "Modelable",
    }

    return display_df.rename(columns=rename_map)


def prepare_logic_chain_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = enrich_intervention_logic(df)

    cols = [
        "Intervention",
        "Impact_Type",
        "Logic_Chain",
        "Assumption_Detail",
        "Assumption_Source",
        "Additional_Cases_Per_Week",
        "Weeks_To_Recover",
        "PAH_Target",
    ]

    available_cols = [col for col in cols if col in work.columns]

    return work[available_cols].rename(
        columns={
            "Impact_Type": "Impact Type",
            "Logic_Chain": "Logic Chain",
            "Assumption_Detail": "Assumption Detail",
            "Assumption_Source": "Assumption Source",
            "Additional_Cases_Per_Week": "Additional Cases / Week",
            "Weeks_To_Recover": "Weeks to Recover",
            "PAH_Target": "PAH Target",
        }
    )