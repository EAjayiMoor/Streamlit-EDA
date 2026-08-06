import streamlit as st
import pandas as pd
import plotly.express as px

from src.data.rtt_loader import load_all_rtt_files
from src.transforms.rtt_transform import (
    summarise_rtt_additions_by_month,
    summarise_rtt_completions_by_month,
    build_rtt_flow_summary,
)

st.set_page_config(page_title="Flow", layout="wide")

st.title("Flow")
st.write(
    """
    This page focuses on the drivers of waiting list movement by comparing demand entering
    the system with patients being completed. It shows whether monthly flow is adding to
    or relieving pressure on the backlog.
    """
)

# ---------------------------------------------------------
# Load and transform flow data
# ---------------------------------------------------------
try:
    raw_df = load_all_rtt_files()

    demand_df = summarise_rtt_additions_by_month(raw_df)
    completion_df = summarise_rtt_completions_by_month(raw_df)

    # Minimal monthly flow summary
    backlog_stub = demand_df[["Month"]].copy()
    backlog_stub["Total"] = 0  # placeholder so build function can still be reused

    flow_df = build_rtt_flow_summary(backlog_stub, demand_df, completion_df)

except Exception as e:
    st.error(f"Error loading or transforming flow data: {e}")
    st.stop()

if flow_df.empty:
    st.warning("No flow data available.")
    st.stop()

# ---------------------------------------------------------
# Sort months properly
# ---------------------------------------------------------
flow_df = flow_df.copy()
flow_df["Month_Date"] = pd.to_datetime(flow_df["Month"], format="%B %Y")
flow_df = flow_df.sort_values("Month_Date").reset_index(drop=True)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Flow Filters")

available_months = flow_df["Month"].tolist()

selected_months = st.sidebar.multiselect(
    "Select months",
    options=available_months,
    default=available_months,
)

filtered_df = flow_df[flow_df["Month"].isin(selected_months)].copy()
filtered_df = filtered_df.sort_values("Month_Date").reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data available for the selected months.")
    st.stop()

# ---------------------------------------------------------
# Latest position
# ---------------------------------------------------------
latest = filtered_df.iloc[-1]

latest_demand = int(latest["additions_total"])
latest_throughput = int(latest["completed_total"])
latest_net_flow = int(latest["net_flow"])
latest_admitted = int(latest["admitted_completed_total"])
latest_nonadmitted = int(latest["nonadmitted_completed_total"])

if len(filtered_df) > 1:
    previous = filtered_df.iloc[-2]
    delta_demand = latest_demand - int(previous["additions_total"])
    delta_throughput = latest_throughput - int(previous["completed_total"])
    delta_net_flow = latest_net_flow - int(previous["net_flow"])
else:
    delta_demand = None
    delta_throughput = None
    delta_net_flow = None

st.subheader("Latest Position")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Demand (Additions)",
        f"{latest_demand:,}",
        delta=f"{delta_demand:+,}" if delta_demand is not None else None,
    )

with col2:
    st.metric(
        "Throughput (Completed)",
        f"{latest_throughput:,}",
        delta=f"{delta_throughput:+,}" if delta_throughput is not None else None,
    )

with col3:
    st.metric(
        "Net Flow",
        f"{latest_net_flow:+,}",
        delta=f"{delta_net_flow:+,}" if delta_net_flow is not None else None,
    )

st.markdown(
    f"""
    **Interpretation:** In the latest month, **{latest_demand:,}** new RTT periods entered the
    waiting list, while **{latest_throughput:,}** pathways were completed
    (**{latest_admitted:,}** admitted and **{latest_nonadmitted:,}** non-admitted),
    resulting in a net flow of **{latest_net_flow:+,}**.
    """
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Demand vs Throughput", "Net Flow", "Summary Table"]
)

# ---------------------------------------------------------
# Tab 1: Demand vs Throughput
# ---------------------------------------------------------
with tab1:
    st.subheader("Demand vs Throughput")

    st.write(
        """
        This view compares patients entering the waiting list with patients being completed.
        It shows whether monthly system flow is adding pressure or supporting recovery.
        """
    )

    fig_demand_throughput = px.line(
        filtered_df,
        x="Month",
        y=["additions_total", "completed_total"],
        markers=True,
        title="Demand and Throughput Over Time",
    )

    fig_demand_throughput.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Patients",
        legend_title_text="Metric",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig_demand_throughput.update_traces(
        hovertemplate="%{fullData.name}: %{y:,}<extra></extra>"
    )

    st.plotly_chart(fig_demand_throughput, use_container_width=True)

    st.markdown(
        """
        **How to read this:**  
        When additions sit above completions, demand is exceeding throughput and pressure is building.  
        When completions sit above additions, the system is reducing pressure.
        """
    )

# ---------------------------------------------------------
# Tab 2: Net Flow
# ---------------------------------------------------------
with tab2:
    st.subheader("Net Flow")

    st.write(
        """
        Net flow is the difference between demand and throughput.
        It shows whether each month is adding to backlog pressure or helping reduce it.
        """
    )

    fig_net = px.bar(
        filtered_df,
        x="Month",
        y="net_flow",
        title="Net Flow Over Time (Demand - Throughput)",
    )

    fig_net.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Patients",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig_net.update_traces(
        hovertemplate="Net Flow: %{y:+,}<extra></extra>"
    )

    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown(
        """
        **How to read this:**  
        Positive values mean additions exceeded completions, increasing pressure.  
        Negative values mean completions exceeded additions, supporting recovery.
        """
    )

# ---------------------------------------------------------
# Tab 3: Summary Table
# ---------------------------------------------------------
with tab3:
    st.subheader("Monthly Flow Summary")

    table_df = filtered_df[
        [
            "Month",
            "additions_total",
            "admitted_completed_total",
            "nonadmitted_completed_total",
            "completed_total",
            "net_flow",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "additions_total": "Demand (Additions)",
            "admitted_completed_total": "Admitted Completed",
            "nonadmitted_completed_total": "Non-Admitted Completed",
            "completed_total": "Throughput (Completed)",
            "net_flow": "Net Flow",
        }
    )

    st.dataframe(table_df, use_container_width=True)

with st.expander("Definitions"):
    st.markdown(
        """
**Demand (Additions)**  
New RTT periods entering the waiting list.

**Throughput (Completed)**  
Completed pathways, including admitted and non-admitted patients.

**Net Flow**  
Demand minus throughput. Positive values increase pressure; negative values reduce it.
        """
    )