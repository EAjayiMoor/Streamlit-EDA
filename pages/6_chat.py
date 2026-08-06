import os
import streamlit as st
import pandas as pd

from src.data.rtt_loader import load_all_rtt_files
from src.transforms.rtt_transform import (
    filter_pah_incomplete,
    add_wait_band_metrics,
    summarise_rtt_by_month,
    summarise_rtt_by_month_specialty,
    summarise_rtt_additions_by_month,
    summarise_rtt_completions_by_month,
    build_rtt_flow_summary,
)
from src.utils.openai_chat import ask_openai_about_data


st.set_page_config(page_title="Chat", layout="wide")

st.title("Chat with the data")

st.write(
    """
    Ask questions about RTT backlog, demand, throughput, flow, specialty performance,
    waiting time distribution, and heatmap risk. Responses are grounded in the data
    loaded into the app.
    """
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
try:
    raw_df = load_all_rtt_files()

    pah_df = filter_pah_incomplete(raw_df)
    metric_df = add_wait_band_metrics(pah_df)

    backlog_df = summarise_rtt_by_month(metric_df)
    specialty_df = summarise_rtt_by_month_specialty(metric_df)
    demand_df = summarise_rtt_additions_by_month(raw_df)
    completion_df = summarise_rtt_completions_by_month(raw_df)

    flow_df = build_rtt_flow_summary(backlog_df, demand_df, completion_df)

except Exception as e:
    st.error(f"Error loading chat context data: {e}")
    st.stop()


# ---------------------------------------------------------
# Prepare dates
# ---------------------------------------------------------
def add_month_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Month_Date" not in df.columns:
        df["Month_Date"] = pd.to_datetime(df["Month"], format="%B %Y", errors="coerce")
    return df.sort_values("Month_Date")


backlog_df = add_month_date(backlog_df)
specialty_df = add_month_date(specialty_df)
demand_df = add_month_date(demand_df)
completion_df = add_month_date(completion_df)
flow_df = add_month_date(flow_df)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Chat Context")

context_scope = st.sidebar.selectbox(
    "Choose data context",
    [
        "Executive summary",
        "Backlog",
        "Flow",
        "Specialty",
        "All core data",
    ],
)

months_to_include = st.sidebar.slider(
    "Recent months to include",
    min_value=3,
    max_value=18,
    value=6,
    step=1,
)

show_context = st.sidebar.checkbox("Preview context sent to AI", value=False)


# ---------------------------------------------------------
# Build concise data context
# ---------------------------------------------------------
latest_backlog = backlog_df.iloc[-1]
latest_flow = flow_df.iloc[-1]

latest_backlog_total = int(latest_backlog["Total"])
latest_pct_18 = float(latest_backlog.get("pct_0_18", 0))
latest_52_plus = int(latest_backlog.get("waiting_52_plus_total", 0))

latest_demand = int(latest_flow.get("additions_total", 0))
latest_throughput = int(latest_flow.get("completed_total", 0))
latest_net_flow = int(latest_flow.get("net_flow", 0))


top_specialties = (
    specialty_df[specialty_df["Month"] == specialty_df.iloc[-1]["Month"]]
    .sort_values("Total", ascending=False)
    .head(10)
)

top_specialty_table = top_specialties[
    [
        "Month",
        "Treatment Function Name",
        "Total",
        "waiting_0_18_total",
        "waiting_52_plus_total",
        "pct_0_18",
        "pct_52_plus",
    ]
].to_string(index=False)


backlog_context = f"""
Latest backlog position:
- Month: {latest_backlog["Month"]}
- Total backlog / incomplete pathways: {latest_backlog_total:,}
- % within 18 weeks: {latest_pct_18:.1%}
- 52+ week waits: {latest_52_plus:,}

Recent backlog trend:
{backlog_df.tail(months_to_include).to_string(index=False)}
"""


flow_context = f"""
Latest flow position:
- Month: {latest_flow["Month"]}
- Demand / additions: {latest_demand:,}
- Throughput / completed pathways: {latest_throughput:,}
- Net flow: {latest_net_flow:+,}

Recent flow trend:
{flow_df.tail(months_to_include)[[
    "Month",
    "additions_total",
    "admitted_completed_total",
    "nonadmitted_completed_total",
    "completed_total",
    "net_flow",
]].to_string(index=False)}
"""


specialty_context = f"""
Latest top specialty position:
{top_specialty_table}
"""


executive_context = f"""
Executive context:
- Latest backlog is {latest_backlog_total:,} incomplete RTT pathways.
- Latest % within 18 weeks is {latest_pct_18:.1%}.
- Latest 52+ waits are {latest_52_plus:,}.
- Latest demand is {latest_demand:,}.
- Latest throughput is {latest_throughput:,}.
- Latest net flow is {latest_net_flow:+,}.

Top specialties by backlog:
{top_specialty_table}
"""


if context_scope == "Executive summary":
    data_context = executive_context
elif context_scope == "Backlog":
    data_context = backlog_context
elif context_scope == "Flow":
    data_context = flow_context
elif context_scope == "Specialty":
    data_context = specialty_context
else:
    data_context = f"""
{executive_context}

{backlog_context}

{flow_context}

{specialty_context}
"""


if show_context:
    with st.expander("Context being sent to AI", expanded=True):
        st.text(data_context[:15000])


# ---------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------
st.subheader("Suggested questions")

q1, q2, q3 = st.columns(3)

with q1:
    if st.button("Explain latest RTT position"):
        st.session_state["pending_question"] = "Explain the latest RTT backlog position using the data."

with q2:
    if st.button("What is driving pressure?"):
        st.session_state["pending_question"] = "What is driving pressure based on demand, throughput and net flow?"

with q3:
    if st.button("Which specialties need focus?"):
        st.session_state["pending_question"] = "Which specialties appear to need the most focus based on backlog and long waits?"


# ---------------------------------------------------------
# Chat state
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

default_question = st.session_state.pop("pending_question", "")

question = st.text_input(
    "Ask a question about the data",
    value=default_question,
)


if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analysing data context..."):
            try:
                answer = ask_openai_about_data(question, data_context)

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "context_scope": context_scope,
                    }
                )

            except Exception as e:
                st.error("Error calling AI assistant")
                st.write(e)


# ---------------------------------------------------------
# Display chat history
# ---------------------------------------------------------
st.subheader("Conversation")

if not st.session_state.chat_history:
    st.info("Ask a question to start the conversation.")

for item in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {item['question']}")
    st.markdown(f"**AI Assistant:** {item['answer']}")
    st.caption(f"Context used: {item['context_scope']}")
    st.markdown("---")


# ---------------------------------------------------------
# Definitions
# ---------------------------------------------------------
with st.expander("Definitions used by the assistant"):
    st.markdown(
        """
**Backlog**  
Incomplete RTT pathways — patients currently waiting.

**Demand / Additions**  
New RTT periods entering the waiting list.

**Throughput / Completed pathways**  
Completed admitted and non-admitted RTT pathways.

**Net Flow**  
Demand minus throughput. Positive values add pressure; negative values support backlog reduction.

**0–18 Weeks**  
Patients waiting within the RTT standard window.

**18–52 Weeks**  
Patients waiting beyond 18 weeks but less than one year.

**52+ Weeks**  
Patients waiting over one year. This is a key backlog severity indicator.

**Heatmap score**  
Relative score from 0 to 1, where higher values indicate greater comparative pressure.
        """
    )