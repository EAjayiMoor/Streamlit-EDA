import pandas as pd


def remove_aggregate_rows(
    df: pd.DataFrame,
    specialty_column: str = "Treatment Function Name",
) -> pd.DataFrame:
    """
    Remove aggregate/non-specialty rows from specialty-based RTT outputs.

    Why:
    - RTT extracts often include summary rows such as 'Total'
    - these are not real specialties and distort charts, rankings, and heatmaps
    - cleaning them centrally avoids repeating the same filter across pages/functions
    """

    cleaned = df.copy()

    aggregate_values = {
        "Total",
        "TOTAL",
        "total",
    }

    cleaned = cleaned[
        ~cleaned[specialty_column].astype(str).str.strip().isin(aggregate_values)
    ]

    return cleaned