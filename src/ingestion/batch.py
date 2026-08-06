from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import re
from typing import Iterable

import pandas as pd

from src.utils.data_cleaning import remove_aggregate_rows
from src.data.outpatient_loader import _clean_outpatient_data
from src.data.referral_loader import _clean_referral_data
from src.data.inpatient_loader import _clean_inpatient_data
from src.workflows.config import Workflow


@dataclass
class FileValidation:
    filename: str
    rows: int = 0
    columns: int = 0
    organisation: str = "Not detected"
    period: str = "Not detected"
    status: str = "Not checked"
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _detect_period(filename: str) -> str:
    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)[-_ ](20\d{2})",
        filename,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).title()} {match.group(2)}"
    return "Not detected"


def _detect_organisation(frame: pd.DataFrame) -> str:
    candidates = (
        "Provider Org Name",
        "Provider Organisation Name",
        "Organisation",
        "Organisation Name",
        "Provider",
    )
    for column in candidates:
        if column in frame.columns:
            values = frame[column].dropna().astype(str).str.strip()
            if not values.empty:
                unique = values.unique()
                return unique[0] if len(unique) == 1 else f"{len(unique)} organisations"
    return "Not detected"


def validate_csv(file, workflow: Workflow) -> tuple[pd.DataFrame | None, FileValidation]:
    result = FileValidation(filename=file.name, period=_detect_period(file.name))
    try:
        frame = pd.read_csv(BytesIO(file.getvalue()), low_memory=False)
    except Exception as exc:
        result.status = "Error"
        result.message = f"Could not read CSV: {exc}"
        return None, result

    frame.columns = [str(column).strip() for column in frame.columns]
    try:
        if workflow.key == "rtt":
            frame = _normalise_rtt_frame(frame, file.name)
        elif workflow.key == "referrals":
            frame = _clean_referral_data(frame)
        elif workflow.key == "outpatient":
            frame = _clean_outpatient_data(frame)
        elif workflow.key == "inpatient":
            frame = _clean_inpatient_data(frame)
    except Exception as exc:
        result.status = "Error"
        result.message = str(exc)
        return None, result
    result.rows, result.columns = frame.shape
    result.organisation = _detect_organisation(frame)
    missing = [column for column in workflow.expected_columns if column not in frame.columns]

    if missing:
        result.status = "Warning"
        result.message = "Missing expected columns: " + ", ".join(missing)
    else:
        result.status = "Ready"
        result.message = "Schema checks passed."

    frame["_source_file"] = file.name
    frame["_source_period"] = result.period
    frame["_source_organisation"] = result.organisation
    return frame, result


def _normalise_rtt_frame(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Match uploaded RTT files to the existing monthly RTT loader contract."""
    frame = frame.copy()
    if "Month" not in frame.columns:
        frame["Month"] = _detect_period(filename)
    if "source_file" not in frame.columns:
        frame["source_file"] = filename
    return remove_aggregate_rows(frame)


def validate_batch(files: Iterable, workflow: Workflow) -> tuple[list[pd.DataFrame], list[FileValidation]]:
    frames: list[pd.DataFrame] = []
    results: list[FileValidation] = []
    for file in files:
        frame, result = validate_csv(file, workflow)
        if frame is not None:
            frames.append(frame)
        results.append(result)
    return frames, results


def manifest_frame(results: Iterable[FileValidation]) -> pd.DataFrame:
    return pd.DataFrame([result.as_dict() for result in results])
