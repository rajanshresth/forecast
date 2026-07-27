"""
Streamlit frontend for the admissions forecasting app.

Run with (after starting the FastAPI backend separately):
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import requests
import altair as alt

st.set_page_config(page_title="Admissions Forecast", layout="wide")
st.title("Admissions Forecast — Bayesian vs. Moving Average")
st.caption(
    "Trend + seasonal Bayesian regression, compared against a seasonal moving-average baseline, fit live from your Google Sheet."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI backend URL", "http://localhost:8000")
    sheet_id = st.text_input(
        "Google Sheet ID", help="The long ID from your sheet's URL"
    )
    sheet_name = st.text_input("Sheet / tab name", "Sheet1")
    periods_ahead = st.slider("Periods to forecast", 1, 10, 4)

    with st.expander("Moving average settings"):
        use_all_history = st.checkbox("Use all past same-season values", value=True)
        ma_window = (
            None
            if use_all_history
            else st.slider("Same-season lookback window", 1, 10, 3)
        )
        run_backtest = st.checkbox(
            "Run backtest comparison",
            value=True,
            help="Walk-forward validation on your historical data — with only a few points this will be based on very few test cases, but it's still informative.",
        )

    with st.expander("Advanced (MCMC settings)"):
        likelihood = st.selectbox(
            "Likelihood",
            options=["poisson", "negative_binomial"],
            help=(
                "Poisson (default): fewer parameters, more stable with small N, "
                "assumes variance ≈ mean. Negative Binomial: allows overdispersion, "
                "but needs more data (8-10+ points) to estimate reliably — with few "
                "points it produces much wider, right-skewed intervals."
            ),
        )
        draws = st.select_slider(
            "Posterior draws", options=[500, 1000, 2000, 4000], value=1000
        )
        chains = st.select_slider("Chains", options=[2, 4, 6], value=4)
    st.markdown("---")
    st.caption(
        "Sheet must have columns **Year, Season, Admissions** and be shared "
        "as 'Anyone with the link can view'."
    )
    run = st.button("Run forecast", type="primary", use_container_width=True)

if not run:
    st.info("Enter your Google Sheet ID in the sidebar, then click **Run forecast**.")
    st.stop()

if not sheet_id:
    st.error("Enter a Google Sheet ID first.")
    st.stop()

with st.spinner(
    "Fitting Bayesian model + running backtest... this can take 10-60 seconds"
):
    try:
        resp = requests.post(
            f"{api_url}/forecast",
            json={
                "sheet_id": sheet_id,
                "sheet_name": sheet_name,
                "periods_ahead": periods_ahead,
                "draws": draws,
                "tune": 1500,
                "chains": chains,
                "likelihood": likelihood,
                "ma_window": ma_window,
                "run_backtest": run_backtest,
            },
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the FastAPI backend at {api_url}. Is it running?")
        st.stop()

if resp.status_code != 200:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"Error: {detail}")
    st.stop()

data = resp.json()
hist_df = pd.DataFrame(data["historical"])
fc_df = pd.DataFrame(data["forecasts"])
ma_df = pd.DataFrame(data["moving_average_forecasts"])
params = data["params"]
season_cycle = data[
    "season_cycle"
]  # e.g. ["spring", "autumn"] -- detected from your sheet
season_a, season_b = season_cycle[0].capitalize(), season_cycle[1].capitalize()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical + Forecast")

    hist_plot = hist_df.rename(columns={"admissions": "value"})[
        ["period_label", "value"]
    ].copy()
    hist_plot["type"] = "Historical"
    fc_plot = fc_df.rename(columns={"mean": "value"})[["period_label", "value"]].copy()
    fc_plot["type"] = "Bayesian forecast"
    ma_plot = ma_df.rename(columns={"forecast": "value"})[
        ["period_label", "value"]
    ].copy()
    ma_plot["type"] = "Moving average forecast"
    combined = pd.concat([hist_plot, fc_plot, ma_plot], ignore_index=True)
    period_order = list(hist_plot["period_label"]) + list(fc_plot["period_label"])

    line = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("period_label:N", sort=period_order, title="Period"),
            y=alt.Y("value:Q", title="Admissions"),
            color=alt.Color("type:N", title=None),
            strokeDash=alt.StrokeDash("type:N", legend=None),
        )
    )
    band = (
        alt.Chart(fc_df)
        .mark_area(opacity=0.15, color="orange")
        .encode(
            x=alt.X("period_label:N", sort=period_order),
            y=alt.Y("lower_95:Q", title="Admissions"),
            y2="upper_95:Q",
        )
    )
    st.altair_chart((band + line).properties(height=420), use_container_width=True)

with col2:
    st.subheader("What the model learned")
    trend = params["trend_pct_per_period"]
    season = params["seasonal_effect_pct"]

    st.metric(
        "Trend per period",
        f"{trend['mean']:+.1f}%",
        help=f"95% credible interval: [{trend['lower_95']:+.1f}%, {trend['upper_95']:+.1f}%]",
    )
    st.metric(
        f"{season_b} vs {season_a}",
        f"{season['mean']:+.1f}%",
        help=f"95% credible interval: [{season['lower_95']:+.1f}%, {season['upper_95']:+.1f}%]",
    )

    if season["lower_95"] > 0:
        st.success(
            f"{season_b} is credibly higher than {season_a} (interval excludes 0%)."
        )
    elif season["upper_95"] < 0:
        st.success(
            f"{season_a} is credibly higher than {season_b} (interval excludes 0%)."
        )
    else:
        st.warning(
            f"The seasonal effect's 95% interval includes 0% — with this little data, we can't yet be confident {season_a} and {season_b} actually differ."
        )

    st.caption(
        "Model: log-link "
        + params["likelihood"].replace("_", " ")
        + " regression — forecasts are always non-negative integers (∈ ℝ⁺), "
        "never negative admissions, by construction."
    )

st.subheader("Forecast: Bayesian vs. Moving Average")
compare_df = fc_df[["period_label", "mean", "lower_95", "upper_95"]].merge(
    ma_df[["period_label", "forecast"]], on="period_label"
)
st.dataframe(
    compare_df.rename(
        columns={
            "period_label": "Period",
            "mean": "Bayesian forecast",
            "lower_95": "Bayesian lower 95%",
            "upper_95": "Bayesian upper 95%",
            "forecast": "Moving avg forecast",
        }
    ).round(1),
    use_container_width=True,
    hide_index=True,
)

if "backtest" in data:
    st.subheader("Backtest: which one actually predicted better?")
    bt = data["backtest"]
    bt_ma, bt_bayes = bt["moving_average"], bt["bayesian"]

    if bt_bayes["n_test"] == 0:
        st.info(
            "Not enough historical points yet for a backtest — add a few more seasons and re-run."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Backtest points", bt_bayes["n_test"])
        c2.metric(
            "MAE — Bayesian vs Moving Avg",
            f"{bt_bayes['mae']:.1f}",
            delta=f"{bt_bayes['mae'] - bt_ma['mae']:+.1f} vs MA",
            delta_color="inverse",
        )
        c3.metric(
            "RMSE — Bayesian vs Moving Avg",
            f"{bt_bayes['rmse']:.1f}",
            delta=f"{bt_bayes['rmse'] - bt_ma['rmse']:+.1f} vs MA",
            delta_color="inverse",
        )

        if bt_bayes["n_test"] < 3:
            st.caption(
                f"Only {bt_bayes['n_test']} backtest point(s) available with this much history — "
                "treat this comparison as indicative, not conclusive. It'll get more reliable as "
                "you log more seasons."
            )

        detail_rows = []
        for d_ma, d_bayes in zip(bt_ma["details"], bt_bayes["details"]):
            detail_rows.append(
                {
                    "Period": d_ma["period_label"],
                    "Actual": d_ma["actual"],
                    "Moving avg predicted": round(d_ma["predicted"], 1),
                    "Moving avg error": round(d_ma["error"], 1),
                    "Bayesian predicted": round(d_bayes["predicted"], 1),
                    "Bayesian error": round(d_bayes["error"], 1),
                }
            )
        st.dataframe(
            pd.DataFrame(detail_rows), use_container_width=True, hide_index=True
        )

        st.caption(
            "Lower MAE/RMSE = better. The moving average has no way to express "
            "uncertainty or a trend — the Bayesian model's edge (when it has one) "
            "is the credible interval, not necessarily a lower point-forecast error "
            "on this little data."
        )

with st.expander("Historical data used"):
    st.dataframe(
        hist_df[["period_label", "admissions"]].rename(
            columns={"period_label": "Period", "admissions": "Admissions"}
        ),
        use_container_width=True,
        hide_index=True,
    )
