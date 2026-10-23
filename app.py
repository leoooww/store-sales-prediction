from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import urllib.request
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "sales_model.joblib"
GITHUB_MODEL_RAW = "https://raw.githubusercontent.com/leoooww/store-sales-prediction/main/models/sales_model.joblib"
GITHUB_SAMPLE_RAW = "https://raw.githubusercontent.com/leoooww/store-sales-prediction/main/data/historical_sample.csv.gz"

st.set_page_config(
    page_title="Retail Store Sales Forecasting & Demand Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,slnt,wdth,wght,GRAD,ROND@8..144,-10..0,25..150,100..1000,-400..400,0..100&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* 1. Base font-family on root container and typography elements */
    html, body, .stApp,
    h1, h2, h3, h4, h5, h6,
    p, label, input, textarea, select,
    .stMarkdown, .stText, .stCaption,
    .stButton > button, .stDownloadButton > button,
    .stSelectbox, .stMultiSelect, .stRadio, .stCheckbox,
    .stSlider, .stNumberInput, .stTextInput,
    .stMetric, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    .metric-card, .metric-label, .metric-value, .metric-sub,
    [data-testid="stTable"], [data-testid="stDataFrame"],
    [data-testid="stSidebar"] {
        font-family: 'Google Sans Flex', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* 2. Code and monospace blocks */
    code, pre, pre *, code *, kbd {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
    }

    /* 3. MATERIAL SYMBOLS & ICONS: Strictly target icon spans to preserve glyphs without breaking toolbar flex containers */
    [data-testid="stIconMaterial"],
    [data-testid*="Icon"]:not([data-testid="stChatMessageAvatarImage"]),
    [data-testid*="icon"],
    span[translate="no"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    span[class*="e1vmumty"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="collapsedControl"] span,
    [data-baseweb="popover"] span[translate="no"],
    [data-baseweb="menu"] span[translate="no"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Metric cards styling */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #212529;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_chart(fig: go.Figure) -> None:
    """Render Plotly chart with fallback for Streamlit version differences."""
    fig.update_layout(font_family="Google Sans Flex, sans-serif")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with st.spinner("Downloading trained model artifact from repository..."):
                urllib.request.urlretrieve(GITHUB_MODEL_RAW, path)
        except Exception as dl_err:
            st.error(f"Failed to fetch model artifact from remote: {dl_err}")
            return None

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Failed to load model artifact ({path}): {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_historical_sample(sales_path: Optional[str | Path] = None, max_rows: int = 150000) -> pd.DataFrame:
    path = Path(sales_path) if sales_path is not None else (BASE_DIR / "data" / "train.csv")
    if not path.is_absolute():
        path = BASE_DIR / path

    fallback_path = BASE_DIR / "data" / "historical_sample.csv.gz"

    cols = ["Store", "Date", "Sales", "Customers", "Open", "Promo", "DayOfWeek"]
    if path.exists():
        df = pd.read_csv(path, usecols=cols, low_memory=False, nrows=max_rows)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path, usecols=cols, low_memory=False)
    else:
        try:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(GITHUB_SAMPLE_RAW, fallback_path)
            if fallback_path.exists():
                df = pd.read_csv(fallback_path, usecols=cols, low_memory=False)
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Open"] == 1].sort_values("Date")
    return df


@st.cache_data(show_spinner=False)
def get_store_coordinates() -> pd.DataFrame:
    """Deterministically assign stores to German metropolitan hubs for geospatial analysis."""
    hubs = [
        ("Berlin", 52.5200, 13.4050),
        ("Munich", 48.1351, 11.5820),
        ("Hamburg", 53.5511, 9.9937),
        ("Frankfurt", 50.1109, 8.6821),
        ("Cologne / Rhine-Ruhr", 50.9375, 6.9603),
        ("Stuttgart", 48.7758, 9.1829),
        ("Hanover", 52.3759, 9.7320),
        ("Leipzig / Dresden", 51.3397, 12.3731),
        ("Nuremberg", 49.4521, 11.0767),
    ]

    records = []
    for s in range(1, 1116):
        rng = np.random.RandomState(s * 42)
        hub_name, hub_lat, hub_lon = hubs[s % len(hubs)]
        lat = hub_lat + rng.normal(0, 0.075)
        lon = hub_lon + rng.normal(0, 0.095)
        bearing = (s * 53) % 360
        records.append({
            "Store": s,
            "Hub": hub_name,
            "lat": lat,
            "lon": lon,
            "bearing": bearing,
        })
    return pd.DataFrame(records)


def predict_sales(
    model_bundle: Dict[str, Any],
    store_id: int,
    target_date: date,
    promo: int,
    state_holiday: str,
    school_holiday: int,
    store_type: str,
    assortment: str,
    competition_dist: float,
    competition_active: int,
) -> float:
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    store_type_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    assortment_map = {"Basic (a)": 1, "Extra (b)": 2, "Extended (c)": 3, "a": 1, "b": 2, "c": 3}
    holiday_map = {"None": 0, "Public Holiday": 1, "Easter": 2, "Christmas": 3, "0": 0, "a": 1, "b": 2, "c": 3}

    ts = pd.Timestamp(target_date)

    input_dict = {
        "Store": store_id,
        "DayOfWeek": ts.day_of_week,
        "Promo": int(promo),
        "SchoolHoliday": int(school_holiday),
        "StateHoliday": holiday_map.get(state_holiday, 0),
        "StoreType": store_type_map.get(store_type.lower(), 1),
        "Assortment": assortment_map.get(assortment, 1),
        "CompetitionDistance": float(competition_dist),
        "CompetitionDistanceLog": float(np.log1p(competition_dist)),
        "CompetitionActive": int(competition_active),
        "Year": ts.year,
        "Month": ts.month,
        "Day": ts.day,
        "WeekOfYear": int(ts.isocalendar().week),
        "IsMonthStart": int(ts.is_month_start),
        "IsMonthEnd": int(ts.is_month_end),
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]
    prediction = float(model.predict(input_df)[0])
    return max(0.0, prediction)


def render_sidebar(store_profiles: Dict[int, Dict[str, Any]]) -> Tuple[int, date, int, str, int, str, str, float, int]:
    st.sidebar.header("Store & Date Configuration")

    store_id = st.sidebar.number_input(
        "Store ID",
        min_value=1,
        max_value=1115,
        value=1,
        step=1,
        help="Select any store in the network (1 to 1115)",
    )

    profile = store_profiles.get(store_id, {})
    default_dist = profile.get("CompetitionDistance", 1270.0)
    type_inv = {1: "a", 2: "b", 3: "c", 4: "d"}
    assort_inv = {1: "Basic (a)", 2: "Extra (b)", 3: "Extended (c)"}

    default_type = type_inv.get(profile.get("StoreType", 1), "a")
    default_assort = assort_inv.get(profile.get("Assortment", 1), "Basic (a)")

    target_date = st.sidebar.date_input(
        "Forecast Date",
        value=date(2015, 8, 1),
        min_value=date(2013, 1, 1),
        max_value=date(2016, 12, 31),
    )

    st.sidebar.subheader("Promotions & Holidays")
    promo = 1 if st.sidebar.toggle("Active Promotion (Promo)", value=True) else 0
    school_holiday = 1 if st.sidebar.toggle("School Holiday", value=False) else 0
    state_holiday = st.sidebar.selectbox("State Holiday", ["None", "Public Holiday", "Easter", "Christmas"], index=0)

    st.sidebar.subheader("Competition Radius (Interactive)")
    competition_dist = st.sidebar.slider(
        "Competition Distance (meters)",
        min_value=50.0,
        max_value=25000.0,
        value=float(default_dist),
        step=50.0,
        help="Slide to move competitor closer or farther on the map",
    )
    comp_active = 1 if st.sidebar.checkbox("Competitor Currently Open", value=bool(profile.get("CompetitionActive", 1))) else 0

    with st.sidebar.expander("Store Profile Overrides", expanded=False):
        store_type = st.selectbox("Store Type", ["a", "b", "c", "d"], index=["a", "b", "c", "d"].index(default_type))
        assortment = st.selectbox(
            "Assortment Level",
            ["Basic (a)", "Extra (b)", "Extended (c)"],
            index=["Basic (a)", "Extra (b)", "Extended (c)"].index(default_assort),
        )

    return (
        store_id,
        target_date,
        promo,
        state_holiday,
        school_holiday,
        store_type,
        assortment,
        competition_dist,
        comp_active,
    )


def build_geospatial_map(
    store_coords: pd.DataFrame,
    store_id: int,
    competition_dist: float,
    comp_active: int,
    store_type: str,
    assortment: str,
    pred_sales: float,
    show_network: bool = True,
) -> go.Figure:
    """Generate dynamic geospatial map with store pin, moving competitor, and catchment buffer."""
    store_row = store_coords[store_coords["Store"] == store_id].iloc[0]
    store_lat = store_row["lat"]
    store_lon = store_row["lon"]
    bearing = store_row["bearing"]
    hub = store_row["Hub"]

    delta_lat = (competition_dist * np.cos(np.radians(bearing))) / 111320.0
    delta_lon = (competition_dist * np.sin(np.radians(bearing))) / (111320.0 * np.cos(np.radians(store_lat)))
    comp_lat = store_lat + delta_lat
    comp_lon = store_lon + delta_lon

    circle_angles = np.linspace(0, 360, 60)
    circle_lats = store_lat + (competition_dist * np.cos(np.radians(circle_angles))) / 111320.0
    circle_lons = store_lon + (competition_dist * np.sin(np.radians(circle_angles))) / (
        111320.0 * np.cos(np.radians(store_lat))
    )

    fig = go.Figure()

    if show_network:
        nearby = store_coords[(store_coords["Hub"] == hub) & (store_coords["Store"] != store_id)]
        if not nearby.empty:
            fig.add_trace(
                go.Scattermap(
                    lat=nearby["lat"],
                    lon=nearby["lon"],
                    mode="markers",
                    marker=dict(size=7, color="#6c757d", opacity=0.55),
                    name=f"Network ({hub})",
                    text=[f"Store {s}" for s in nearby["Store"]],
                    hovertemplate="<b>%{text}</b><br>Hub: " + hub + "<extra></extra>",
                )
            )

    fig.add_trace(
        go.Scattermap(
            lat=circle_lats,
            lon=circle_lons,
            mode="lines",
            line=dict(color="rgba(230, 57, 70, 0.7)", width=2),
            fill="toself",
            fillcolor="rgba(230, 57, 70, 0.08)",
            name="Catchment Buffer",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=[store_lat, comp_lat],
            lon=[store_lon, comp_lon],
            mode="lines",
            line=dict(color="#e63946", width=2),
            name=f"Vector ({competition_dist:,.0f}m)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=[store_lat],
            lon=[store_lon],
            mode="markers+text",
            marker=dict(size=16, color="#1d3557"),
            text=[f"Store {store_id}"],
            textposition="top right",
            name=f"Store {store_id} (Focus)",
            hovertemplate=(
                f"<b>Store {store_id}</b><br>"
                f"Region: {hub}<br>"
                f"Type: {store_type} | Assortment: {assortment}<br>"
                f"Predicted Sales: €{pred_sales:,.0f}<extra></extra>"
            ),
        )
    )

    status_str = "Open & Operating" if comp_active else "Closed"
    fig.add_trace(
        go.Scattermap(
            lat=[comp_lat],
            lon=[comp_lon],
            mode="markers+text",
            marker=dict(size=14, color="#e63946"),
            text=["Competitor"],
            textposition="bottom right",
            name=f"Competitor ({competition_dist:,.0f}m)",
            hovertemplate=(
                f"<b>Primary Competitor</b><br>"
                f"Distance: {competition_dist:,.0f} m ({competition_dist / 1000.0:.2f} km)<br>"
                f"Status: {status_str}<extra></extra>"
            ),
        )
    )

    center_lat = (store_lat + comp_lat) / 2
    center_lon = (store_lon + comp_lon) / 2
    if competition_dist < 600:
        zoom = 15
    elif competition_dist < 2000:
        zoom = 14
    elif competition_dist < 6000:
        zoom = 12
    elif competition_dist < 15000:
        zoom = 10
    else:
        zoom = 9

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


def main():
    st.title("Retail Store Sales Forecasting & Demand Analytics")

    model_bundle = load_model_bundle()
    if model_bundle is None:
        st.error(
            f"Model artifact could not be loaded from `{DEFAULT_MODEL_PATH}`.\n\n"
            f"- Working directory: `{Path.cwd()}`\n"
            f"- Models directory exists: `{DEFAULT_MODEL_PATH.parent.exists()}`\n"
            f"- Model file exists: `{DEFAULT_MODEL_PATH.exists()}`\n\n"
            "Please reboot the app on Streamlit Cloud (**Manage app** in bottom-right -> `⋮` -> **Reboot app**) to fetch the latest commits and clear cache."
        )
        st.stop()

    store_profiles = model_bundle.get("store_profiles", {})
    metrics = model_bundle.get("metrics", {})
    store_coords = get_store_coordinates()

    (
        store_id,
        target_date,
        promo,
        state_holiday,
        school_holiday,
        store_type,
        assortment,
        competition_dist,
        comp_active,
    ) = render_sidebar(store_profiles)

    pred_sales = predict_sales(
        model_bundle,
        store_id=store_id,
        target_date=target_date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
        store_type=store_type,
        assortment=assortment,
        competition_dist=competition_dist,
        competition_active=comp_active,
    )

    hist_df = load_historical_sample()
    store_hist = hist_df[hist_df["Store"] == store_id] if not hist_df.empty else pd.DataFrame()

    avg_sales = store_hist["Sales"].mean() if not store_hist.empty else pred_sales
    delta_pct = ((pred_sales - avg_sales) / avg_sales * 100) if avg_sales > 0 else 0.0

    sales_per_cust = (store_hist["Sales"] / store_hist["Customers"]).median() if not store_hist.empty else 9.5
    est_customers = int(pred_sales / max(sales_per_cust, 1.0))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Daily Revenue</div>
                <div class="metric-value">€{pred_sales:,.0f}</div>
                <div class="metric-sub" style="color: {'#28a745' if delta_pct >= 0 else '#dc3545'};">
                    {'+' if delta_pct >= 0 else ''}{delta_pct:.1f}% vs Store Mean
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Footfall</div>
                <div class="metric-value">{est_customers:,}</div>
                <div class="metric-sub" style="color: #6c757d;">
                    ~€{sales_per_cust:.2f} avg spend
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        rmspe_val = metrics.get("rmspe", 19.13)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model RMSPE Error</div>
                <div class="metric-value">{rmspe_val:.2f}%</div>
                <div class="metric-sub" style="color: #6c757d;">
                    Holdout validation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Competitor Proximity</div>
                <div class="metric-value">{competition_dist / 1000.0:.2f} km</div>
                <div class="metric-sub" style="color: #6c757d;">
                    {competition_dist:,.0f} meters away
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    tab_map, tab_timeline, tab_weekly, tab_features = st.tabs([
        "Geospatial & Catchment Map",
        "Sales Timeline",
        "Weekly Distribution",
        "Feature Importance",
    ])

    with tab_map:
        col_ctrl, col_info = st.columns([3, 1])
        with col_ctrl:
            show_network = st.checkbox("Display Regional Store Network", value=True)
        with col_info:
            catchment_area_km2 = np.pi * ((competition_dist / 1000.0) ** 2)
            st.caption(f"Catchment Area: **{catchment_area_km2:,.1f} km²** | Radius: **{competition_dist / 1000.0:.2f} km**")

        map_fig = build_geospatial_map(
            store_coords=store_coords,
            store_id=store_id,
            competition_dist=competition_dist,
            comp_active=comp_active,
            store_type=store_type,
            assortment=assortment,
            pred_sales=pred_sales,
            show_network=show_network,
        )
        render_chart(map_fig)
        st.info("Tip: Adjust the **Competition Distance (meters)** slider in the sidebar. The red competitor pin and catchment circle will dynamically move closer or further in real time.")

    with tab_timeline:
        if not store_hist.empty:
            recent_hist = store_hist.tail(90).copy()
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=recent_hist["Date"],
                    y=recent_hist["Sales"],
                    mode="lines+markers",
                    name="Historical Sales",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=4),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(target_date)],
                    y=[pred_sales],
                    mode="markers",
                    name="Forecast",
                    marker=dict(color="#d62728", size=12, symbol="diamond", line=dict(width=2, color="white")),
                )
            )

            fig.update_layout(
                title=f"Store {store_id} Sales Trajectory (Trailing 90 Days + Forecast Point)",
                xaxis_title="Date",
                yaxis_title="Daily Sales (€)",
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            render_chart(fig)
        else:
            st.info("Historical transaction records unavailable for selected store timeline.")

    with tab_weekly:
        if not store_hist.empty:
            day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            store_hist_dow = store_hist.copy()
            store_hist_dow["DayName"] = store_hist_dow["DayOfWeek"].map(day_map)

            fig_box = px.box(
                store_hist_dow,
                x="DayName",
                y="Sales",
                category_orders={"DayName": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
                title=f"Store {store_id} Historical Sales by Day of Week",
                color_discrete_sequence=["#2b5c8f"],
                template="plotly_white",
            )
            fig_box.update_layout(height=400, margin=dict(l=40, r=40, t=50, b=40))
            render_chart(fig_box)
        else:
            st.info("Weekday variance benchmark requires historical data.")

    with tab_features:
        model = model_bundle.get("model")
        feature_cols = model_bundle.get("feature_cols", [])
        if hasattr(model, "feature_importances_"):
            imp_df = (
                pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
                .sort_values("Importance", ascending=True)
                .tail(12)
            )

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Predictive Drivers (Normalized Gain)",
                color="Importance",
                color_continuous_scale="Viridis",
                template="plotly_white",
            )
            fig_imp.update_layout(
                height=420,
                coloraxis_showscale=False,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            render_chart(fig_imp)


if __name__ == "__main__":
    main()
