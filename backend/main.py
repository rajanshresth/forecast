"""
FastAPI backend for Admissions Forecasting
--------------------------------------------
Provides:
  - GET  /health
  - POST /data/google-sheet   -> fetch a public Google Sheet as records
  - POST /forecast            -> run Poisson GLM / Holt-Winters / OLS forecast

Run with:  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import io
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = FastAPI(title="Admissions Forecasting API", version="1.0")


# =========================================================
# Schemas
# =========================================================
class SheetRequest(BaseModel):
    sheet_id: str = Field(..., description="The Google Sheet ID (from the URL)")
    gid: str = Field("0", description="The gid (tab id) of the specific sheet/tab")


class SheetResponse(BaseModel):
    columns: List[str]
    rows: List[dict]


class DataPoint(BaseModel):
    t: int  # sequential period index, 1..n
    season: str  # categorical seasonal bucket label (e.g. "Autumn", "DOW_3")
    value: float
    label: str  # display label, e.g. "2022-Autumn" or "Day 1"


class ForecastRequest(BaseModel):
    data: List[DataPoint]
    seasonal_period: int = Field(
        ..., gt=0, description="e.g. 2 for Spring/Autumn, 7 for day-of-week"
    )
    horizon: int = Field(4, gt=0, le=52)
    method: Literal["poisson_glm", "holt_winters", "ols"]


class ForecastResponse(BaseModel):
    labels: List[str]  # history labels
    fitted: List[float]  # in-sample fitted values
    future_labels: List[str]
    forecast: List[float]
    ci_lower: Optional[List[float]] = None
    ci_upper: Optional[List[float]] = None
    params: dict
    diagnostics: dict
    summary_text: str


# =========================================================
# Helpers
# =========================================================
def _future_seasons(
    season_history: List[str], seasonal_period: int, horizon: int
) -> List[str]:
    """Continue the repeating seasonal pattern forward. Assumes the first
    `seasonal_period` entries define one full, perfectly repeating cycle."""
    pattern = season_history[:seasonal_period]
    n = len(season_history)
    return [pattern[(n + i) % seasonal_period] for i in range(horizon)]


def _clean(values) -> List[float]:
    """Replace NaN/inf with None-safe floats for JSON serialization."""
    out = []
    for v in np.asarray(values, dtype=float):
        out.append(None if (np.isnan(v) or np.isinf(v)) else float(v))
    return out


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/google-sheet", response_model=SheetResponse)
def fetch_google_sheet(req: SheetRequest):
    """
    Fetches a Google Sheet via its public CSV export URL.
    The sheet (or specific tab) MUST be shared as
    'Anyone with the link can view' for this to work.
    """
    gid_value = req.gid.strip()
    if gid_value.isdigit():
        url = f"https://docs.google.com/spreadsheets/d/{req.sheet_id}/export?format=csv&gid={gid_value}"
    else:
        # Not a numeric gid — treat it as a sheet/tab NAME instead (e.g. "Sheet1"),
        # using Google's gviz endpoint which accepts names directly.
        from urllib.parse import quote

        url = (
            f"https://docs.google.com/spreadsheets/d/{req.sheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={quote(gid_value)}"
        )

    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Network error reaching Google Sheets: {e}"
        )

    content_type = r.headers.get("Content-Type", "")

    if r.status_code == 400:
        if gid_value.isdigit():
            reason = (
                f"the numeric gid '{gid_value}' does not exist on this spreadsheet, "
                f"or the sheet_id is wrong"
            )
        else:
            reason = (
                f"there's no tab named '{gid_value}' on this spreadsheet (sheet/tab "
                f"names are case-sensitive), or the sheet_id is wrong"
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Google rejected the request (HTTP 400) for URL: {url} — "
                f"this almost always means {reason}. Open that exact URL in an "
                f"incognito browser tab to see Google's own error page for confirmation."
            ),
        )

    if r.status_code in (401, 403) or "accounts.google.com" in r.url:
        raise HTTPException(
            status_code=403,
            detail=(
                "Google requires sign-in to view this sheet. Set sharing to "
                "'Anyone with the link' → Viewer, then try again."
            ),
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected response from Google (HTTP {r.status_code}) for URL: {url}",
        )

    if "text/html" in content_type:
        raise HTTPException(
            status_code=403,
            detail=(
                "Got an HTML page instead of CSV data — this sheet is not publicly "
                "viewable yet. Set sharing to 'Anyone with the link' → Viewer, then try again."
            ),
        )

    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Downloaded the sheet but couldn't parse it as CSV: {e}",
        )

    df = df.dropna(how="all")
    return SheetResponse(columns=list(df.columns), rows=df.to_dict(orient="records"))


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if len(req.data) < req.seasonal_period * 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {req.seasonal_period * 2} observations "
                f"(2 full seasonal cycles of period {req.seasonal_period}); "
                f"got {len(req.data)}."
            ),
        )

    t = [d.t for d in req.data]
    season = [d.season for d in req.data]
    value = [d.value for d in req.data]
    labels = [d.label for d in req.data]

    n = len(req.data)
    future_t = list(range(n + 1, n + 1 + req.horizon))
    future_season = _future_seasons(season, req.seasonal_period, req.horizon)
    future_labels = [f"t={ft}" for ft in future_t]  # frontend may relabel these

    if req.method == "ols":
        df = pd.DataFrame({"t": t, "season": season, "value": value})
        fit = smf.ols("value ~ t + C(season)", data=df).fit()
        fdf = pd.DataFrame({"t": future_t, "season": future_season})
        pred = fit.get_prediction(fdf).summary_frame(alpha=0.05)

        return ForecastResponse(
            labels=labels,
            fitted=_clean(fit.fittedvalues.values),
            future_labels=future_labels,
            forecast=_clean(pred["mean"].values),
            ci_lower=_clean(pred["obs_ci_lower"].values),
            ci_upper=_clean(pred["obs_ci_upper"].values),
            params={k: float(v) for k, v in fit.params.items()},
            diagnostics={
                "r_squared": float(fit.rsquared),
                "adj_r_squared": float(fit.rsquared_adj),
                "aic": float(fit.aic),
            },
            summary_text=fit.summary().as_text(),
        )

    elif req.method == "poisson_glm":
        df = pd.DataFrame({"t": t, "season": season, "value": value})
        fit = smf.glm(
            "value ~ t + C(season)", data=df, family=sm.families.Poisson()
        ).fit()
        fdf = pd.DataFrame({"t": future_t, "season": future_season})
        pred = fit.get_prediction(fdf).summary_frame(alpha=0.05)

        dispersion = (
            float(fit.pearson_chi2 / fit.df_resid) if fit.df_resid > 0 else float("nan")
        )

        return ForecastResponse(
            labels=labels,
            fitted=_clean(fit.fittedvalues.values),
            future_labels=future_labels,
            forecast=_clean(pred["mean"].values),
            ci_lower=_clean(pred["mean_ci_lower"].values),
            ci_upper=_clean(pred["mean_ci_upper"].values),
            params={k: float(v) for k, v in fit.params.items()},
            diagnostics={
                "deviance": float(fit.deviance),
                "pearson_chi2": float(fit.pearson_chi2),
                "dispersion": dispersion,
                "overdispersed": bool(dispersion > 1.5)
                if not np.isnan(dispersion)
                else False,
                "aic": float(fit.aic),
            },
            summary_text=fit.summary().as_text(),
        )

    elif req.method == "holt_winters":
        y = pd.Series(value)
        use_seasonal = req.seasonal_period > 1
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=req.seasonal_period if use_seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit()
        fc = fit.forecast(req.horizon)

        ci_lower, ci_upper = None, None
        try:
            sims = fit.simulate(req.horizon, repetitions=500, error="add")
            lo = np.nanpercentile(sims, 2.5, axis=1)
            hi = np.nanpercentile(sims, 97.5, axis=1)
            if not np.any(np.isnan(lo)) and not np.any(np.isnan(hi)):
                ci_lower, ci_upper = _clean(lo), _clean(hi)
        except Exception:
            pass  # CI simulation can fail on degenerate/very short series; forecast still returned

        params = {
            k: (float(v) if np.isscalar(v) else str(v)) for k, v in fit.params.items()
        }

        return ForecastResponse(
            labels=labels,
            fitted=_clean(fit.fittedvalues),
            future_labels=future_labels,
            forecast=_clean(fc.values),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            params=params,
            diagnostics={"sse": float(fit.sse)},
            summary_text=str(fit.summary()),
        )

    raise HTTPException(status_code=400, detail="Unknown method")
