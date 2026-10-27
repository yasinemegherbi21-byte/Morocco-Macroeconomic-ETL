import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(page_title="Rain and the Economy – Morocco", layout="wide")

st.markdown(
    """
    <style>
        h1 { font-weight: 700; color: #0F172A; letter-spacing: -0.03em; margin-bottom: 0.25rem; }
        .section-header { font-size: 1.3rem; font-weight: 600; color: #1E293B; margin-top: 2rem;
                          margin-bottom: 0.75rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.5rem; }
        .body-text { font-size: 1rem; color: #334155; line-height: 1.6; margin-bottom: 1.25rem; }
        .insight-card { background-color: #F8FAFC; border-left: 4px solid #475569; padding: 1.25rem;
                        margin: 1.5rem 0; border-radius: 0 6px 6px 0; }
        .insight-title { font-weight: 600; color: #0F172A; margin-bottom: 0.5rem; font-size: 1.05rem; }
        div[data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 1.25rem;
                                      border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>Rain and the Economy: droughts, GDP and jobs in Morocco</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 1.1rem; color: #64748B; margin-bottom: 2rem;'>"
    "World Bank data from 1990 to 2024, cleaned and stored in PostgreSQL, "
    "to see how bad harvest years show up in unemployment.</p>",
    unsafe_allow_html=True,
)


# --- Database ---
# Credentials come from the environment (.env file), see .env.example.
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "world_economics")

if not DB_PASSWORD:
    st.error("DB_PASSWORD is not set. Copy .env.example to .env and fill it in.")
    st.stop()


@st.cache_resource
def get_db_engine():
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


try:
    engine = get_db_engine()
    df = pd.read_sql('SELECT * FROM morocco_economic_trends ORDER BY "Year" ASC;', con=engine)
    st.sidebar.success("Connected to PostgreSQL")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error("Could not connect to PostgreSQL")
    st.sidebar.caption(str(exc))
    st.stop()


# --- Filters ---
st.sidebar.markdown("### Years")
year_range = st.sidebar.slider(
    "Analysis window",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)
df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]


# --- 1. Context ---
st.markdown("<div class='section-header'>1. What this is about</div>", unsafe_allow_html=True)
st.markdown(
    """
    <p class='body-text'>
    A large part of Morocco's economy depends on farming, and most crops depend on seasonal rain.
    A dry year does not only hurt farmers: crop output falls, GDP growth drops, and unemployment goes up.
    This page shows that chain with real numbers. The data behind it is pulled from the World Bank API,
    cleaned with Pandas and stored in a PostgreSQL table (<code>morocco_economic_trends</code>).
    </p>
    """,
    unsafe_allow_html=True,
)


# --- 2. Key figures ---
st.markdown("<div class='section-header'>2. Averages over the selected years</div>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Crop production index", f"{df_filtered['Crop_Index'].mean():.1f}")
m2.metric("Unemployment rate", f"{df_filtered['Unemployment_Rate'].mean():.2f}%")
m3.metric("Agriculture share of GDP", f"{df_filtered['Agriculture_GDP_Share'].mean():.2f}%")


# --- 3. Charts ---
st.markdown("<div class='section-header'>3. Trends and what drives unemployment</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Crops vs. unemployment over time", "Which variable matters most"])

with tab1:
    st.markdown(
        """
        <p class='body-text'>
        Crop production index and national unemployment rate, year by year.
        Look at the drought years (1995, 2020): the crop line falls first, the unemployment line rises after.
        </p>
        """,
        unsafe_allow_html=True,
    )

    chart_data = df_filtered.set_index("Year")[["Crop_Index", "Unemployment_Rate"]]
    st.line_chart(chart_data, color=["#475569", "#991B1B"])

    st.markdown(
        """
        <div class='insight-card'>
            <div class='insight-title'>How to read it</div>
            <div class='body-text' style='margin-bottom:0;'>
            The two lines tend to move in opposite directions: a drop in the crop index is usually followed by a
            rise in unemployment the same year or the year after. The relationship is clear visually, but with only
            35 yearly points it should be read as a pattern, not a precise law.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown(
        """
        <p class='body-text'>
        A Random Forest is fitted on the full table with unemployment as the target and three inputs:
        crop index, GDP growth and the share of agriculture in GDP. The bars show how much each input
        contributes to the model's splits.
        </p>
        """,
        unsafe_allow_html=True,
    )

    X = df[["Crop_Index", "GDP_Growth", "Agriculture_GDP_Share"]]
    y = df["Unemployment_Rate"]
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    df_imp = pd.DataFrame({"Importance": rf.feature_importances_}, index=X.columns)

    st.bar_chart(df_imp, color="#334155")

    st.markdown(
        """
        <div class='insight-card'>
            <div class='insight-title'>A word of caution</div>
            <div class='body-text' style='margin-bottom:0;'>
            Feature importance tells you which variable the model leans on, not how well it predicts.
            In the notebook, the same kind of model gets a negative R² on held-out years, so this is a
            descriptive tool, not a forecast.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- 4. Footer ---
st.markdown("<div class='section-header'>4. How the data gets here</div>", unsafe_allow_html=True)
st.markdown(
    """
    <p class='body-text' style='font-size: 0.9rem; color: #64748B;'>
    World Bank API &rarr; Pandas (linear interpolation for the few missing years) &rarr; PostgreSQL
    &rarr; this Streamlit page. A PySpark version of the loader (<code>etl_pipeline.py</code>) reads the
    World Bank CSV exports instead of the API.
    </p>
    """,
    unsafe_allow_html=True,
)
