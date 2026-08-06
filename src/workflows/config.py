from dataclasses import dataclass


@dataclass(frozen=True)
class Workflow:
    key: str
    label: str
    description: str
    data_source: str
    source_note: str
    expected_columns: tuple[str, ...] = ()


WORKFLOWS = (
    Workflow(
        key="rtt",
        label="RTT",
        description="Waiting-list, backlog, flow, and pathway analysis.",
        data_source="NHS England CSV",
        source_note="Suitable for historical NHS England RTT extracts.",
        expected_columns=("Provider Org Name", "RTT Part Description", "Period"),
    ),
    Workflow(
        key="referrals",
        label="Referrals",
        description="Referral demand, source, priority, and specialty analysis.",
        data_source="NHS England CSV",
        source_note="Suitable for historical NHS England referral extracts.",
        expected_columns=("Referral_ID", "Referral_Received_Date", "TFC_Name"),
    ),
    Workflow(
        key="theatre",
        label="Theatre",
        description="Activity, utilisation, cancellations, and productivity analysis.",
        data_source="User-provided local CSV",
        source_note="Usually not available as a directly comparable NHS England extract and may contain sensitive operational data.",
    ),
    Workflow(
        key="outpatient",
        label="Outpatient activity",
        description="Activity, contact type, clinic, and specialty analysis.",
        data_source="NHS England CSV",
        source_note="Suitable for historical NHS England outpatient extracts.",
        expected_columns=("Contact_ID", "Contact_Start", "TreatmentFunctionDesc"),
    ),
    Workflow(
        key="inpatient",
        label="Inpatient activity",
        description="Admissions, spells, classifications, and specialty analysis.",
        data_source="NHS England CSV",
        source_note="Suitable for historical NHS England inpatient extracts.",
        expected_columns=("Spell ID", "Admission datetime", "Specialty"),
    ),
    Workflow(
        key="workforce",
        label="Workforce",
        description="Workforce, capacity, and cost analysis.",
        data_source="User-provided local CSV",
        source_note="Workforce and capacity data may be sensitive and is not treated as an NHS England upload assumption.",
    ),
    Workflow(
        key="finance",
        label="Finance",
        description="Financial position, opportunity, and variance analysis.",
        data_source="User-provided local CSV",
        source_note="Financial data is sensitive and is not treated as an NHS England upload assumption.",
    ),
)


def get_workflow(key: str) -> Workflow:
    for workflow in WORKFLOWS:
        if workflow.key == key:
            return workflow
    raise KeyError(f"Unknown workflow: {key}")
