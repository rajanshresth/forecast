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
import numpy as np
import pandas as pd
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


def _clean(values, apply_ceil: bool = False) -> List[float]:
    """Replace NaN/inf with None-safe floats for JSON serialization.
    Optionally applies ceiling to round up to the nearest whole number."""
    out = []
    for v in np.asarray(values, dtype=float):
        if np.isnan(v) or np.isinf(v):
            out.append(None)
        else:
            val = np.ceil(v) if apply_ceil else v
            out.append(float(val))
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
    url = f"https://docs.google.com/spreadsheets/d/{req.sheet_id}/export?format=csv&gid={req.gid}"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read the Google Sheet. Make sure it is shared as "
                f"'Anyone with the link can view' and that the sheet_id/gid are correct. "
                f"Underlying error: {e}"
            ),
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
            fitted=_clean(fit.fittedvalues.values, apply_ceil=True),
            future_labels=future_labels,
            forecast=_clean(pred["mean"].values, apply_ceil=True),
            ci_lower=_clean(pred["obs_ci_lower"].values, apply_ceil=True),
            ci_upper=_clean(pred["obs_ci_upper"].values, apply_ceil=True),
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
            fitted=_clean(fit.fittedvalues.values, apply_ceil=True),
            future_labels=future_labels,
            forecast=_clean(pred["mean"].values, apply_ceil=True),
            ci_lower=_clean(pred["mean_ci_lower"].values, apply_ceil=True),
            ci_upper=_clean(pred["mean_ci_upper"].values, apply_ceil=True),
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
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=req.seasonal_period,
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
                ci_lower = _clean(lo, apply_ceil=True)
                ci_upper = _clean(hi, apply_ceil=True)
        except Exception:
            pass  # CI simulation can fail on degenerate/very short series; forecast still returned

        params = {
            k: (float(v) if np.isscalar(v) else str(v)) for k, v in fit.params.items()
        }

        return ForecastResponse(
            labels=labels,
            fitted=_clean(fit.fittedvalues, apply_ceil=True),
            future_labels=future_labels,
            forecast=_clean(fc.values, apply_ceil=True),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            params=params,
            diagnostics={"sse": float(fit.sse)},
            summary_text=str(fit.summary()),
        )

    raise HTTPException(status_code=400, detail="Unknown method")
