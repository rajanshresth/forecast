"""
Streamlit UI — Admissions Forecasting
------------------------------------------------
Talks to the FastAPI backend (backend/main.py) to fetch data from
Google Sheets and run Poisson GLM / Holt-Winters / OLS forecasts.

Run with:  streamlit run app.py
(Make sure the backend is running first: uvicorn main:app --port 8000)
"""

import os
import re
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Admissions Forecasting", layout="wide")
st.title("🏫🎓 Admissions Forecasting")
st.caption(
    "Poisson GLM · Holt-Winters (damped, seasonal) · Classical OLS with seasonality"
)


# =========================================================
# Helpers
# =========================================================
def parse_sheet_url(url_or_id: str, default_gid: str = "0"):
    """Accepts either a raw Sheet ID or a full Google Sheets URL and returns (sheet_id, gid)."""
    url_or_id = url_or_id.strip()
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url_or_id)
    sheet_id = m.group(1) if m else url_or_id
    gid_match = re.search(r"[?#&]gid=(\d+)", url_or_id)
    gid = gid_match.group(1) if gid_match else default_gid
    return sheet_id, gid


def fetch_sheet(sheet_id: str, gid: str) -> pd.DataFrame:
    resp = requests.post(
        f"{BACKEND_URL}/data/google-sheet",
        json={"sheet_id": sheet_id, "gid": gid},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    payload = resp.json()
    return pd.DataFrame(payload["rows"], columns=payload["columns"])


def run_forecast(data_points, seasonal_period, horizon, method):
    resp = requests.post(
        f"{BACKEND_URL}/forecast",
        json={
            "data": data_points,
            "seasonal_period": seasonal_period,
            "horizon": horizon,
            "method": method,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


def build_chart(
    hist_labels,
    hist_values,
    fitted,
    future_labels,
    forecast_vals,
    ci_lower,
    ci_upper,
    title,
    color,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_labels,
            y=hist_values,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_labels,
            y=fitted,
            mode="lines",
            name="Fitted (in-sample)",
            line=dict(color="#1f77b4", width=1.5, dash="dash"),
            opacity=0.6,
        )
    )

    join_x = [hist_labels[-1]] + future_labels
    join_y = [hist_values[-1]] + forecast_vals
    fig.add_trace(
        go.Scatter(
            x=join_x,
            y=join_y,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        )
    )

    if (
        ci_lower is not None
        and ci_upper is not None
        and all(v is not None for v in ci_lower + ci_upper)
    ):
        join_lo = [hist_values[-1]] + ci_lower
        join_hi = [hist_values[-1]] + ci_upper
        fig.add_trace(
            go.Scatter(
                x=join_x + join_x[::-1],
                y=join_hi + join_lo[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Interval",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Admissions",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


METHOD_LABELS = {
    "poisson_glm": "Poisson GLM",
    "holt_winters": "Holt-Winters (Damped, Seasonal)",
    "ols": "Classical OLS with Seasonality",
}
METHOD_COLORS = {
    "poisson_glm": "#ff7f0e",
    "holt_winters": "#d62728",
    "ols": "#2ca02c",
}


# =========================================================
# Sidebar — data source configuration
# =========================================================
st.sidebar.header("⚙️ Data Source")

dataset_choice = st.sidebar.radio(
    "Choose dataset",
    ["Seasonal (Year + Season)", "Daily"],
    help="Seasonal: Sheet1 (Year, Season, Admissions). Daily: Sheet2 (Day, Total admissions).",
)

st.sidebar.markdown("**Google Sheet**")
sheet_url_input = st.sidebar.text_input(
    "Sheet URL or Sheet ID",
    help="Paste the full Google Sheets URL (with gid) or just the Sheet ID. "
    "The sheet/tab must be shared as 'Anyone with the link can view'.",
)
manual_gid = st.sidebar.text_input("gid (tab id, optional override)", value="")

use_sample_data = st.sidebar.checkbox("Use built-in sample data instead", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🔮 Forecast Settings")
horizon = st.sidebar.slider("Periods to forecast", min_value=1, max_value=12, value=4)
selected_methods = st.sidebar.multiselect(
    "Model(s)",
    list(METHOD_LABELS.keys()),
    default=list(METHOD_LABELS.keys()),
    format_func=lambda m: METHOD_LABELS[m],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend: `{BACKEND_URL}`")


# =========================================================
# Sample data (fallback / demo)
# =========================================================
SAMPLE_SEASONAL = pd.DataFrame(
    {
        "Year": [2022, 2023, 2023, 2024, 2024, 2025],
        "Season": [
            "Autumn",
            "Spring",
            "Autumn",
            "Spring",
        ],
        "Admissions": [664, 413, 742, 335],
    }
)

SAMPLE_DAILY = pd.DataFrame(
    {
        "Day": list(range(1, 25)),
        "Total number of admission": [
            7,
            8,
            4,
            8,
            3,
            6,
            7,
            8,
            9,
            4,
            4,
            0,
            5,
            9,
            6,
            5,
            14,
            9,
            1,
            5,
            15,
            4,
            8,
            11,
        ],
    }
)


# =========================================================
# Load data
# =========================================================
df = None
load_error = None

if not use_sample_data and sheet_url_input.strip():
    default_gid = manual_gid.strip() if manual_gid.strip() else "0"
    sheet_id, gid = parse_sheet_url(sheet_url_input, default_gid)
    if manual_gid.strip():
        gid = manual_gid.strip()
    try:
        with st.spinner("Fetching data from Google Sheets..."):
            df = fetch_sheet(sheet_id, gid)
    except Exception as e:
        load_error = str(e)

if df is None:
    df = (
        SAMPLE_SEASONAL
        if dataset_choice == "Seasonal (Year + Season)"
        else SAMPLE_DAILY
    )
    if load_error:
        st.sidebar.error(
            f"Sheet fetch failed, showing sample data instead:\n\n{load_error}"
        )

st.subheader("📄 Data Preview")
st.dataframe(df, use_container_width=True, height=220)


# =========================================================
# Build (t, season, value, label) records for the API
# =========================================================
data_points = None
seasonal_period = None

if dataset_choice == "Seasonal (Year + Season)":
    required_cols = {"Year", "Season", "Admissions"}
    if not required_cols.issubset(df.columns):
        st.error(f"Expected columns {required_cols}, got {list(df.columns)}.")
        st.stop()

    seasons_present = list(df["Season"].unique())
    seasonal_period = st.sidebar.number_input(
        "Seasonal period (# of seasons/cycle)",
        min_value=2,
        max_value=12,
        value=min(len(seasons_present), 2),
        step=1,
    )

    work = df.reset_index(drop=True).copy()
    work["t"] = np.arange(1, len(work) + 1)
    work["label"] = work["Year"].astype(str) + "-" + work["Season"].astype(str)
    data_points = [
        {
            "t": int(r.t),
            "season": str(r.Season),
            "value": float(r.Admissions),
            "label": r.label,
        }
        for r in work.itertuples()
    ]

else:  # Daily
    day_col = "Day" if "Day" in df.columns else df.columns[0]
    value_col = (
        "Total number of admission"
        if "Total number of admission" in df.columns
        else df.columns[1]
    )

    seasonal_period = st.sidebar.number_input(
        "Seasonal period (e.g. 7 = day-of-week cycle)",
        min_value=2,
        max_value=31,
        value=7,
        step=1,
    )

    work = df.reset_index(drop=True).copy()
    work["t"] = np.arange(1, len(work) + 1)
    work["season"] = [f"CYCLE_{(i) % int(seasonal_period)}" for i in range(len(work))]
    work["label"] = "Day " + work[day_col].astype(str)
    data_points = [
        {
            "t": int(row.t),
            "season": str(row.season),
            "value": float(row[value_col]),
            "label": row["label"],
        }
        for _, row in work.iterrows()
    ]


# =========================================================
# Run forecasts
# =========================================================
if st.sidebar.button("Run Forecast", type="primary"):
    if not selected_methods:
        st.warning("Select at least one model in the sidebar.")
        st.stop()

    if len(data_points) < seasonal_period * 2:
        st.error(
            f"Need at least {seasonal_period * 2} observations "
            f"(2 full cycles of period {seasonal_period}); got {len(data_points)}."
        )
        st.stop()

    hist_labels = [d["label"] for d in data_points]
    hist_values = [d["value"] for d in data_points]

    results = {}
    errors = {}
    for method in selected_methods:
        try:
            with st.spinner(f"Fitting {METHOD_LABELS[method]}..."):
                results[method] = run_forecast(
                    data_points, int(seasonal_period), int(horizon), method
                )
        except Exception as e:
            errors[method] = str(e)

    for method, err in errors.items():
        st.error(f"{METHOD_LABELS[method]} failed: {err}")

    if results:
        # Relabel future periods nicely if seasonal dataset
        for method, res in results.items():
            n_future = len(res["future_labels"])
            future_season = None
            # derive continuing season pattern for nicer labels
            pattern = [d["season"] for d in data_points[:seasonal_period]]
            n = len(data_points)
            nice_future_labels = []
            for i in range(n_future):
                s = pattern[(n + i) % seasonal_period]
                nice_future_labels.append(f"+{i + 1} ({s})")
            res["future_labels"] = nice_future_labels

        tabs = st.tabs([METHOD_LABELS[m] for m in results.keys()])
        comparison_rows = []

        for tab, (method, res) in zip(tabs, results.items()):
            with tab:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = build_chart(
                        hist_labels,
                        hist_values,
                        res["fitted"],
                        res["future_labels"],
                        res["forecast"],
                        res.get("ci_lower"),
                        res.get("ci_upper"),
                        METHOD_LABELS[method],
                        METHOD_COLORS[method],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**Forecast values**")
                    fc_df = pd.DataFrame(
                        {
                            "Period": res["future_labels"],
                            "Forecast": [
                                round(v, 1) if v is not None else None
                                for v in res["forecast"]
                            ],
                        }
                    )
                    if res.get("ci_lower") and res.get("ci_upper"):
                        fc_df["CI Lower"] = [
                            round(v, 1) if v is not None else None
                            for v in res["ci_lower"]
                        ]
                        fc_df["CI Upper"] = [
                            round(v, 1) if v is not None else None
                            for v in res["ci_upper"]
                        ]
                    st.dataframe(fc_df, use_container_width=True, hide_index=True)

                    st.markdown("**Diagnostics**")
                    st.json(res["diagnostics"])

                with st.expander("Full model summary"):
                    st.code(res["summary_text"])

                for i, label in enumerate(res["future_labels"]):
                    comparison_rows.append(
                        {
                            "Model": METHOD_LABELS[method],
                            "Period": label,
                            "Forecast": round(res["forecast"][i], 1)
                            if res["forecast"][i] is not None
                            else None,
                        }
                    )

        if len(results) > 1:
            st.markdown("---")
            st.subheader("📊 Side-by-side comparison")
            comp_df = pd.DataFrame(comparison_rows)
            pivot = comp_df.pivot(index="Period", columns="Model", values="Forecast")
            st.dataframe(pivot, use_container_width=True)
else:
    st.info(
        "Configure your data source and model(s) in the sidebar, then click **Run Forecast**."
    )
