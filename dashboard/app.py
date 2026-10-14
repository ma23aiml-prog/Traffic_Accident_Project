"""
dashboard/app.py
-----------------
Streamlit interactive analytics dashboard for the Traffic Accident
Severity Analytics project.

Run with:
    streamlit run dashboard/app.py

Reads data from:
    data/cleaned/accident_severity_dataset.csv
    (falls back gracefully if database is preferred — just swap the loader)
"""

import sys
from pathlib import Path

# ── Add project root to sys.path so relative imports work ─────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Accident Severity Analytics — UK 2025",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; color: #38bdf8; }
    .metric-card p  { margin: 0; font-size: 0.85rem; color: #94a3b8; }
    .section-header { color: #38bdf8; border-bottom: 2px solid #1e40af; padding-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────
DATA_FILE = ROOT / "data" / "cleaned" / "accident_severity_dataset.csv"

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_COLOURS = {
    "Fatal":   "#ef4444",
    "Serious": "#f97316",
    "Slight":  "#22c55e",
}


@st.cache_data(show_spinner="Loading accident data …")
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(
            f"Dataset not found: {DATA_FILE}\n\n"
            "Please run the data pipeline first:\n"
            "```\npython scripts/process_weather.py\n"
            "python scripts/collect_road_data.py\n"
            "python scripts/integrate_data.py\n```"
        )
        st.stop()

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["severity_label"] = df["collision_severity"].map(SEVERITY_LABELS).fillna("Unknown")
    df["month_name"] = df["date"].dt.strftime("%b")
    df["month_num"]  = df["date"].dt.month

    return df


df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")

# Severity filter
severity_options = sorted(df["severity_label"].dropna().unique())
selected_severity = st.sidebar.multiselect(
    "Severity",
    options=severity_options,
    default=severity_options,
)

# Month filter
all_months = list(range(1, 13))
month_labels = {m: pd.Timestamp(f"2025-{m:02d}-01").strftime("%B") for m in all_months}
selected_months = st.sidebar.multiselect(
    "Month",
    options=all_months,
    default=all_months,
    format_func=lambda m: month_labels[m],
)

# Road type filter
if "road_type_label" in df.columns:
    rt_options = sorted(df["road_type_label"].dropna().unique())
    selected_rt = st.sidebar.multiselect(
        "Road Type",
        options=rt_options,
        default=rt_options,
    )
else:
    selected_rt = None

# Hour of day filter
hour_min, hour_max = st.sidebar.slider(
    "Hour of day", min_value=0, max_value=23, value=(0, 23)
)

# ── Apply filters ─────────────────────────────────────────────────────────
mask = (
    df["severity_label"].isin(selected_severity)
    & df["month_num"].isin(selected_months)
    & df["hour_of_day"].between(hour_min, hour_max, inclusive="both")
)
if selected_rt is not None and "road_type_label" in df.columns:
    mask &= df["road_type_label"].isin(selected_rt)

filtered = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────
st.title("🚗 UK Traffic Accident Severity Analytics — 2025")
st.markdown(
    "_Data sources: UK DfT STATS19 · Open-Meteo Historical Weather · DfT STATS19 Lookup Tables_"
)

# ── KPI cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
total    = len(filtered)
fatal    = (filtered["collision_severity"] == 1).sum()
serious  = (filtered["collision_severity"] == 2).sum()
slight   = (filtered["collision_severity"] == 3).sum()
wx_match = filtered["wx_temp_mean_c"].notna().sum()

metrics = [
    (col1, "Total Accidents", f"{total:,}"),
    (col2, "Fatal",     f"{fatal:,}"),
    (col3, "Serious",   f"{serious:,}"),
    (col4, "Slight",    f"{slight:,}"),
    (col5, "Weather-matched", f"{wx_match:,}"),
]
for col, label, value in metrics:
    with col:
        st.markdown(
            f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Chart 1: Severity distribution ───────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 Accident Severity Distribution", unsafe_allow_html=True)
    sev_counts = (
        filtered.groupby("severity_label", sort=False)
        .size()
        .reset_index(name="count")
    )
    fig1 = px.bar(
        sev_counts,
        x="severity_label",
        y="count",
        color="severity_label",
        color_discrete_map=SEVERITY_COLOURS,
        text="count",
        labels={"severity_label": "Severity", "count": "Accidents"},
        template="plotly_dark",
    )
    fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig1.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Accidents by month ───────────────────────────────────────────
with col_b:
    st.markdown("### 📅 Accidents by Month", unsafe_allow_html=True)
    monthly = (
        filtered.groupby(["month_num", "month_name", "severity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("month_num")
    )
    fig2 = px.line(
        monthly,
        x="month_name",
        y="count",
        color="severity_label",
        color_discrete_map=SEVERITY_COLOURS,
        markers=True,
        labels={"month_name": "Month", "count": "Accidents", "severity_label": "Severity"},
        template="plotly_dark",
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Accidents by hour of day ────────────────────────────────────
st.markdown("### 🕐 Accidents by Hour of Day", unsafe_allow_html=True)
if "hour_of_day" in filtered.columns:
    hour_data = filtered[filtered["hour_of_day"].notna()].copy()
    hour_data["hour_int"] = hour_data["hour_of_day"].astype(int)
    hourly = (
        hour_data.groupby(["hour_int", "severity_label"])
        .size()
        .reset_index(name="count")
    )
    fig3 = px.bar(
        hourly,
        x="hour_int",
        y="count",
        color="severity_label",
        color_discrete_map=SEVERITY_COLOURS,
        labels={"hour_int": "Hour of Day (24h)", "count": "Accidents", "severity_label": "Severity"},
        template="plotly_dark",
        barmode="stack",
    )
    fig3.update_layout(height=380, xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Hour-of-day data not available in this dataset.")

col_c, col_d = st.columns(2)

# ── Chart 4: Weather conditions vs. severity ──────────────────────────────
with col_c:
    st.markdown("### 🌦️ Accidents by Weather Conditions", unsafe_allow_html=True)
    if "weather_conditions_label" in filtered.columns:
        wx_sev = (
            filtered[filtered["weather_conditions_label"].notna()]
            .groupby(["weather_conditions_label", "severity_label"])
            .size()
            .reset_index(name="count")
        )
        fig4 = px.bar(
            wx_sev,
            x="weather_conditions_label",
            y="count",
            color="severity_label",
            color_discrete_map=SEVERITY_COLOURS,
            barmode="group",
            labels={"weather_conditions_label": "Weather", "count": "Accidents"},
            template="plotly_dark",
        )
        fig4.update_layout(height=420, xaxis_tickangle=-35, showlegend=True)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        # Fall back to Open-Meteo precipitation categories
        if "wx_precip_mm" in filtered.columns:
            precip = filtered["wx_precip_mm"].fillna(0)
            filtered["precip_cat"] = pd.cut(
                precip,
                bins=[-0.01, 0.1, 5, 20, 1000],
                labels=["Dry (0mm)", "Light rain (<5mm)", "Moderate rain (<20mm)", "Heavy rain (20mm+)"],
            )
            wx_cat = (
                filtered.groupby(["precip_cat", "severity_label"])
                .size()
                .reset_index(name="count")
            )
            fig4 = px.bar(
                wx_cat,
                x="precip_cat",
                y="count",
                color="severity_label",
                color_discrete_map=SEVERITY_COLOURS,
                barmode="group",
                template="plotly_dark",
                labels={"precip_cat": "Precipitation Category", "count": "Accidents"},
            )
            fig4.update_layout(height=420)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Weather condition data not available in filtered dataset.")

# ── Chart 5: Road type vs. severity ──────────────────────────────────────
with col_d:
    st.markdown("### 🛣️ Accidents by Road Type", unsafe_allow_html=True)
    rt_col = "road_type_label" if "road_type_label" in filtered.columns else "road_type"
    if rt_col in filtered.columns:
        rt_data = (
            filtered[filtered[rt_col].notna()]
            .groupby([rt_col, "severity_label"])
            .size()
            .reset_index(name="count")
        )
        fig5 = px.bar(
            rt_data,
            x=rt_col,
            y="count",
            color="severity_label",
            color_discrete_map=SEVERITY_COLOURS,
            barmode="stack",
            labels={rt_col: "Road Type", "count": "Accidents"},
            template="plotly_dark",
        )
        fig5.update_layout(height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Road type data not available.")

# ── Chart 6 (bonus): Geographic map ──────────────────────────────────────
st.markdown("### 🗺️ Accident Locations", unsafe_allow_html=True)
st.caption(
    "Showing a random sample of up to 5,000 accidents. "
    "Each point is colour-coded by severity."
)
map_sample = filtered.dropna(subset=["latitude", "longitude"]).sample(
    min(5000, len(filtered)), random_state=42
)

if not map_sample.empty:
    fig6 = px.scatter_mapbox(
        map_sample,
        lat="latitude",
        lon="longitude",
        color="severity_label",
        color_discrete_map=SEVERITY_COLOURS,
        hover_data={"latitude": False, "longitude": False},
        zoom=5,
        center={"lat": 52.5, "lon": -1.5},
        mapbox_style="carto-darkmatter",
        opacity=0.5,
        height=500,
        template="plotly_dark",
    )
    fig6.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("No coordinates available for map display.")

# ── Summary statistics ────────────────────────────────────────────────────
with st.expander("📋 Summary Statistics"):
    num_cols = [
        c for c in ["wx_temp_mean_c", "wx_precip_mm", "wx_wind_max_kmh",
                     "number_of_vehicles", "number_of_casualties", "speed_limit"]
        if c in filtered.columns
    ]
    if num_cols:
        st.dataframe(filtered[num_cols].describe().round(2), use_container_width=True)
    else:
        st.write("No numeric summary columns available.")

# ── Data table preview ────────────────────────────────────────────────────
with st.expander("🔎 Raw Data Preview (first 100 rows)"):
    display_cols = [
        c for c in [
            "collision_index", "date", "severity_label", "road_type_label",
            "speed_limit", "light_conditions_label", "weather_conditions_label",
            "wx_temp_mean_c", "wx_precip_mm", "number_of_vehicles",
        ]
        if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols].head(100), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: UK Department for Transport (OGL v3) · "
    "Weather: Open-Meteo (CC BY 4.0) · "
    "Road context: DfT STATS19 Data Guide (OGL v3). "
    "This dashboard is for educational purposes only. "
    "Observed patterns do not imply causation."
)
