import pandas as pd


def filter_inpatients(
    df: pd.DataFrame,
    start_date=None,
    end_date=None,
    specialties=None,
    elective_emergency=None,
    patient_classifications=None,
    statuses=None,
) -> pd.DataFrame:
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[
            filtered["Admission datetime"] >= pd.to_datetime(start_date)
        ]

    if end_date is not None:
        filtered = filtered[
            filtered["Admission datetime"] <= pd.to_datetime(end_date)
        ]

    if specialties:
        filtered = filtered[
            filtered["Standardised_Specialty"].isin(specialties)
        ]

    if elective_emergency and "Elective/emergency" in filtered.columns:
        filtered = filtered[
            filtered["Elective/emergency"].isin(elective_emergency)
        ]

    if patient_classifications and "Patient classification" in filtered.columns:
        filtered = filtered[
            filtered["Patient classification"].isin(patient_classifications)
        ]

    if statuses and "Status" in filtered.columns:
        filtered = filtered[filtered["Status"].isin(statuses)]

    return filtered


def summarise_inpatients_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Admission_Month", "Inpatient Activity"])

    return (
        df.groupby("Admission_Month")
        .agg(**{"Inpatient Activity": ("Spell ID", "nunique")})
        .reset_index()
        .sort_values("Admission_Month")
    )