"""
Streamlit UI — Generic Time-Series Forecasting
------------------------------------------------
Talks to the FastAPI backend (backend/main.py) to load data (Google Sheet,
uploaded file, or pasted CSV) and run Poisson GLM / Holt-Winters / OLS
forecasts on ANY dataset — you choose which column is the value, which
column defines order, and which (if any) column is the seasonal category.

Run with:  streamlit run app.py
(Make sure the backend is running first: uvicorn main:app --port 8000)
"""

import io
import os
import re
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Forecasting Studio", layout="wide")
st.title("🔮 Forecasting Studio")
st.caption(
    "Poisson GLM · Holt-Winters (damped, seasonal) · Classical OLS with seasonality — on any dataset"
)


# =========================================================
# Data-loading helpers
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


# =========================================================
# Generic column-agnostic data shaping
# =========================================================
def smart_sort_series(series: pd.Series):
    """Try numeric, then datetime, else leave as-is (original row order).
    Returns (sort_key_or_None, kind_str)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric, "numeric"
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().all():
        return dt, "datetime"
    return None, "unsortable"


def build_data_points(df: pd.DataFrame, value_col: str, order_col, season_col):
    """Build the generic {t, season, value, label} records the API expects,
    regardless of the source dataframe's actual column names/shape."""
    work = df.copy()
    sort_note = None

    if order_col:
        key, kind = smart_sort_series(work[order_col])
        if key is not None:
            work = (
                work.assign(_sort_key=key)
                .sort_values("_sort_key")
                .drop(columns="_sort_key")
            )
        else:
            sort_note = (
                f"Column '{order_col}' isn't numeric or a recognizable date, so rows are "
                f"kept in their original order instead of being sorted by it."
            )

    work = work.reset_index(drop=True)
    work["_t"] = np.arange(1, len(work) + 1)

    if season_col:
        work["_season"] = work[season_col].astype(str)
    else:
        work["_season"] = "ALL"

    if order_col:
        work["_label"] = work[order_col].astype(str)
        if season_col:
            work["_label"] = work["_label"] + " (" + work["_season"] + ")"
    else:
        work["_label"] = "t=" + work["_t"].astype(str)
        if season_col:
            work["_label"] = work["_label"] + " (" + work["_season"] + ")"

    data_points = [
        {
            "t": int(work["_t"].iloc[i]),
            "season": str(work["_season"].iloc[i]),
            "value": float(work[value_col].iloc[i]),
            "label": str(work["_label"].iloc[i]),
        }
        for i in range(len(work))
    ]
    return data_points, sort_note


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
    y_label,
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
        yaxis_title=y_label,
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
# Sidebar — Step 1: Data source
# =========================================================
st.sidebar.header("① Data Source")

source_type = st.sidebar.radio(
    "Where's your data?",
    ["Google Sheet", "Upload file", "Paste CSV"],
)

df = None
load_error = None

if source_type == "Google Sheet":
    sheet_url_input = st.sidebar.text_input(
        "Sheet URL or Sheet ID",
        help="Paste the full Google Sheets URL (with gid) or just the Sheet ID. "
        "The sheet/tab must be shared as 'Anyone with the link can view'.",
    )
    manual_gid = st.sidebar.text_input("gid (tab id, optional override)", value="")
    if sheet_url_input.strip():
        default_gid = manual_gid.strip() if manual_gid.strip() else "0"
        sheet_id, gid = parse_sheet_url(sheet_url_input, default_gid)
        try:
            with st.spinner("Fetching data from Google Sheets..."):
                df = fetch_sheet(sheet_id, gid)
        except Exception as e:
            load_error = str(e)

elif source_type == "Upload file":
    uploaded = st.sidebar.file_uploader(
        "CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as e:
            load_error = str(e)

else:  # Paste CSV
    pasted = st.sidebar.text_area(
        "Paste CSV text",
        height=150,
        placeholder="col1,col2,col3\nval1,val2,val3\n...",
    )
    if pasted.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted))
        except Exception as e:
            load_error = str(e)

if load_error:
    st.sidebar.error(f"Couldn't load data:\n\n{load_error}")

if df is None:
    st.info(
        "👈 Load your data in the sidebar to get started (Google Sheet, file upload, or pasted CSV)."
    )
    st.stop()

df = df.dropna(how="all").reset_index(drop=True)

st.subheader("📄 Data Preview")
st.dataframe(df, use_container_width=True, height=220)

if df.empty or len(df.columns) < 1:
    st.error("The loaded data has no usable rows/columns.")
    st.stop()


# =========================================================
# Sidebar — Step 2: Column mapping (fully dynamic)
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("② Map Your Columns")

numeric_cols = [
    c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()
]
all_cols = list(df.columns)

if not numeric_cols:
    st.error(
        "No column in this data is fully numeric — at least one column must contain "
        "the values you want to forecast."
    )
    st.stop()

value_col = st.sidebar.selectbox(
    "Value column (what to forecast)",
    numeric_cols,
    index=len(numeric_cols) - 1,
    help="The numeric column containing the values you want to predict.",
)

order_options = ["(row order)"] + [c for c in all_cols if c != value_col]
order_col_choice = st.sidebar.selectbox(
    "Order / time column",
    order_options,
    help="Defines the sequence of your data (e.g. Year, Date, Day). "
    "Choose '(row order)' if your rows are already in the right sequence.",
)
order_col = None if order_col_choice == "(row order)" else order_col_choice

season_options = ["(none)"] + [c for c in all_cols if c not in (value_col, order_col)]
season_col_choice = st.sidebar.selectbox(
    "Seasonal / category column (optional)",
    season_options,
    help="A categorical column that repeats in a cycle (e.g. Season, day-of-week). "
    "Choose '(none)' if there's no seasonality.",
)
season_col = None if season_col_choice == "(none)" else season_col_choice

data_points, sort_note = build_data_points(df, value_col, order_col, season_col)
if sort_note:
    st.sidebar.caption(f"ℹ️ {sort_note}")

if season_col:
    n_categories = len(set(d["season"] for d in data_points))
    default_period = max(2, n_categories)
    max_period = max(2, len(data_points) // 2)
    seasonal_period = st.sidebar.number_input(
        "Seasonal period (# categories per cycle)",
        min_value=1,
        max_value=max(2, max_period),
        value=min(default_period, max(2, max_period)),
        step=1,
        help=f"Detected {n_categories} distinct value(s) in '{season_col}'.",
    )
else:
    seasonal_period = 1
    st.sidebar.caption(
        "No seasonal column selected — seasonal period fixed at 1 (trend-only)."
    )


# =========================================================
# Sidebar — Step 3: Forecast settings
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("③ Forecast Settings")
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
        # Relabel future periods nicely, continuing the detected seasonal pattern
        pattern = [d["season"] for d in data_points[:seasonal_period]]
        n = len(data_points)
        for method, res in results.items():
            n_future = len(res["future_labels"])
            nice_future_labels = []
            for i in range(n_future):
                s = pattern[(n + i) % seasonal_period]
                nice_future_labels.append(
                    f"+{i + 1} ({s})" if season_col else f"+{i + 1}"
                )
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
                        value_col,
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
        "Map your columns and model(s) in the sidebar, then click **Run Forecast**."
    )
