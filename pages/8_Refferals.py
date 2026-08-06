import streamlit as st
import plotly.express as px

from src.data.referral_loader import (
    load_referral_data,
    validate_referral_data,
)

from src.transforms.referral_transform import (
    filter_referrals,
    summarise_referrals_by_month,
    summarise_referrals_by_specialty,
    summarise_referrals_by_ccg,
    summarise_referrals_by_source,
    summarise_referrals_by_priority,
    summarise_referrals_by_type,
    add_monthly_growth,
    referral_growth_signal,
    top_specialty_growth,
    referral_heatmap_matrix,
    
)


st.set_page_config(
    page_title="Referral & Demand Intelligence",
    page_icon="📨",
    layout="wide",
)

st.title("📨 Referral & Demand Intelligence")

st.caption(
    "Upstream referral demand intelligence showing where future RTT pressure may be emerging."
)


DEFAULT_REFERRAL_PATH = "data/raw/Refferals"


# -------------------------------------------------------------------
# Load data automatically from project folder
# -------------------------------------------------------------------

try:
    referral_df = load_referral_data(DEFAULT_REFERRAL_PATH)

except Exception as e:
    st.error(f"Could not load referral data: {e}")
    st.info(
        "Check that your referral files are saved in: "
        "`data/raw/Refferals` and are CSV or Excel files."
    )
    st.stop()


warnings = validate_referral_data(referral_df)

for warning in warnings:
    st.warning(warning)


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Referral Filters")

    min_date = referral_df["Referral_Received_Date"].min().date()
    max_date = referral_df["Referral_Received_Date"].max().date()

    selected_dates = st.date_input(
        "Referral received date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    specialties = sorted(referral_df["Standardised_Specialty"].dropna().unique())

    selected_specialties = st.multiselect(
        "Specialty",
        specialties,
        default=[],
    )

    sources = (
        sorted(referral_df["ReferralSource"].dropna().unique())
        if "ReferralSource" in referral_df.columns
        else []
    )

    selected_sources = st.multiselect(
        "Referral source",
        sources,
        default=[],
    )

    priorities = (
        sorted(referral_df["Medical_Priority_Desc"].dropna().unique())
        if "Medical_Priority_Desc" in referral_df.columns
        else []
    )

    selected_priorities = st.multiselect(
        "Medical priority",
        priorities,
        default=[],
    )

    ccgs = (
        sorted(referral_df["CCG"].dropna().unique())
        if "CCG" in referral_df.columns
        else []
    )

    selected_ccgs = st.multiselect(
        "CCG / commissioner",
        ccgs,
        default=[],
    )

    st.markdown("---")
    st.caption(f"Loaded from: `{DEFAULT_REFERRAL_PATH}`")

    if "Source_File" in referral_df.columns:
        st.caption(f"Files loaded: {referral_df['Source_File'].nunique()}")


# -------------------------------------------------------------------
# Apply filters
# -------------------------------------------------------------------

filtered_df = filter_referrals(
    referral_df,
    start_date=start_date,
    end_date=end_date,
    specialties=selected_specialties,
    sources=selected_sources,
    priorities=selected_priorities,
    ccgs=selected_ccgs,
)


# -------------------------------------------------------------------
# Core summaries
# -------------------------------------------------------------------

monthly_df = summarise_referrals_by_month(filtered_df)
monthly_growth_df = add_monthly_growth(monthly_df)
growth_signal = referral_growth_signal(monthly_df, baseline_months=6)

total_referrals = filtered_df["Referral_ID"].nunique()
specialties_count = filtered_df["Standardised_Specialty"].nunique()
months_count = filtered_df["Referral_Month"].nunique()


# -------------------------------------------------------------------
# KPI row
# -------------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total referrals", f"{total_referrals:,.0f}")
kpi2.metric("Specialties", f"{specialties_count:,.0f}")
kpi3.metric("Months in view", f"{months_count:,.0f}")

kpi4.metric(
    "Latest month referrals",
    f"{growth_signal['latest']:,.0f}",
    delta=f"{growth_signal['change_pct']}% vs baseline",
)


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Referral Overview",
        "Specialty Demand",
        "Geographic Demand",
        "Referral Source",
        "Priority Profile",
        "Referral Growth Signals",
    ]
)


# -------------------------------------------------------------------
# Tab 1: Referral Overview
# -------------------------------------------------------------------

with tab1:
    st.subheader("Monthly referral demand")

    if monthly_df.empty:
        st.info("No referral data available for the selected filters.")
    else:
        fig = px.line(
            monthly_df,
            x="Referral_Month",
            y="Referrals",
            markers=True,
            title="Monthly referral volume",
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Referrals",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly referral table")
        st.dataframe(monthly_growth_df, use_container_width=True)


# -------------------------------------------------------------------
# Tab 2: Specialty Demand
# -------------------------------------------------------------------

with tab2:
    st.subheader("Referral demand by specialty")

    specialty_df = summarise_referrals_by_specialty(filtered_df)

    if specialty_df.empty:
        st.info("No specialty referral data available.")
    else:
        top_n = st.slider(
            "Number of specialties to show",
            min_value=5,
            max_value=30,
            value=15,
        )

        fig = px.bar(
            specialty_df.head(top_n),
            x="Referrals",
            y="Standardised_Specialty",
            orientation="h",
            title="Top specialties by referral volume",
        )

        fig.update_layout(
            xaxis_title="Referrals",
            yaxis_title="Specialty",
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Specialty referral table")
        st.dataframe(specialty_df, use_container_width=True)


# -------------------------------------------------------------------
# Tab 3: Geographic Demand
# -------------------------------------------------------------------

with tab3:
    st.subheader("Referral demand by commissioner / geography")

    ccg_df = summarise_referrals_by_ccg(filtered_df)

    if ccg_df.empty:
        st.info("No CCG / commissioner data available.")
    else:
        fig = px.bar(
            ccg_df.head(20),
            x="Referrals",
            y="CCG",
            orientation="h",
            title="Top commissioners by referral volume",
        )

        fig.update_layout(
            xaxis_title="Referrals",
            yaxis_title="CCG / Commissioner",
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Commissioner referral table")
        st.dataframe(ccg_df, use_container_width=True)


# -------------------------------------------------------------------
# Tab 4: Referral Source
# -------------------------------------------------------------------

with tab4:
    st.subheader("Referral source mix")

    source_df = summarise_referrals_by_source(filtered_df)
    type_df = summarise_referrals_by_type(filtered_df)

    source_col, type_col = st.columns(2)

    with source_col:
        st.markdown("### Referral source")

        if source_df.empty:
            st.info("No referral source data available.")
        else:
            fig = px.pie(
                source_df,
                names="ReferralSource",
                values="Referrals",
                title="Referral source mix",
                hole=0.35,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(source_df, use_container_width=True)

    with type_col:
        st.markdown("### Referral type")

        if type_df.empty:
            st.info("No referral type data available.")
        else:
            fig = px.bar(
                type_df,
                x="ReferralType",
                y="Referrals",
                title="Referral type volume",
            )

            fig.update_layout(
                xaxis_title="Referral type",
                yaxis_title="Referrals",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(type_df, use_container_width=True)


# -------------------------------------------------------------------
# Tab 5: Priority Profile
# -------------------------------------------------------------------

with tab5:
    st.subheader("Referral priority profile")

    priority_df = summarise_referrals_by_priority(filtered_df)

    if priority_df.empty:
        st.info("No priority data available.")
    else:
        fig = px.bar(
            priority_df,
            x="Priority",
            y="Referrals",
            title="Referral urgency / priority mix",
        )

        fig.update_layout(
            xaxis_title="Priority",
            yaxis_title="Referrals",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Priority referral table")
        st.dataframe(priority_df, use_container_width=True)


# -------------------------------------------------------------------
# Tab 6: Referral Growth Signals
# -------------------------------------------------------------------

with tab6:
    st.subheader("Specialty referral demand heatmap")

    st.markdown(
        """
        This view shows referral demand intensity by specialty over time.

        Each row represents a specialty and each column represents a month. 
        Red indicates higher referral volume and blue indicates lower referral volume.

        The specialties are ordered so that the highest-volume specialties appear at the top.
        """
    )

    heatmap_top_n = st.slider(
        "Number of specialties to include in heatmap",
        min_value=5,
        max_value=30,
        value=15,
        key="referral_heatmap_top_n",
    )

    heatmap_matrix = referral_heatmap_matrix(
        filtered_df,
        top_n=heatmap_top_n,
    )

    if heatmap_matrix.empty:
        st.info("Not enough data to create the referral demand heatmap.")

    else:
        # Order specialties by total referral volume, highest at the top
        heatmap_matrix = heatmap_matrix.loc[
            heatmap_matrix.sum(axis=1)
            .sort_values(ascending=False)
            .index
        ]

        fig = px.imshow(
            heatmap_matrix,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(
                x="Month",
                y="Specialty",
                color="Referrals",
            ),
            title="Specialty referral demand heatmap",
        )

        fig.update_layout(
            xaxis_title="Referral month",
            yaxis_title="Specialty",
            height=700,
            coloraxis_colorbar=dict(
                title="Referral Volume"
            ),
        )

        fig.update_yaxes(
            autorange="reversed"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
            Interpretation: this heatmap shows where referral demand is concentrated over time.

            Persistent red cells suggest sustained referral pressure. 
            A specialty becoming redder over recent months may indicate an emerging demand hotspot.
            Blue areas indicate comparatively lower referral volume.
            """
        )

        st.subheader("Referral growth signal summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest month",
            f"{growth_signal['latest']:,.0f}",
        )

        c2.metric(
            "Baseline average",
            f"{growth_signal['baseline']:,.0f}",
        )

        c3.metric(
            "Change vs baseline",
            f"{growth_signal['change_pct']}%",
            delta=f"{growth_signal['change']:,.0f} referrals",
        )

        c4.metric(
            "Demand signal",
            growth_signal["signal"],
        )

        st.subheader("Heatmap data table")
        st.dataframe(heatmap_matrix, use_container_width=True)

        st.subheader("Specialty recent change table")

        growth_df = top_specialty_growth(
            filtered_df,
            recent_months=3,
            baseline_months=6,
        )

        if growth_df.empty:
            st.info("Not enough data to calculate specialty growth.")
        else:
            st.dataframe(growth_df, use_container_width=True)