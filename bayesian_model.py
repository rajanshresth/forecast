"""
Bayesian regression for admissions forecasting.

Model (log-link count regression):
    log(mu_i) = beta0 + beta1 * t_i + beta2 * season_idx_i
    admissions_i ~ Poisson(mu_i)                      [default]
                or NegativeBinomial(mu_i, alpha)       [likelihood="negative_binomial"]

Why a log-link count model, and why Poisson by default:
    Admissions are counts of people -- never negative. A plain Normal
    likelihood doesn't know that, and with wide credible intervals from
    small N it can predict negative admissions. Poisson/NegativeBinomial
    both have support {0, 1, 2, ...}, so every forecast and every credible
    interval bound is guaranteed to be a realistic non-negative count.

    Negative Binomial additionally estimates a dispersion parameter to
    allow variance > mean (real overdispersion). That's one more
    parameter to fit -- fine with 8-10+ data points, but with only 3-4
    points it's nearly unconstrained and its own prior uncertainty ends
    up dominating the forecast, producing absurdly wide, right-skewed
    intervals. Poisson (mean = variance, no separate dispersion term) is
    the more stable choice until you have enough seasons to actually
    estimate overdispersion. Switch to "negative_binomial" once you do
    (see README) -- especially if your real variance across repeated
    same-season admissions looks clearly larger than the mean.

    Because of the log link, beta1 and beta2 are on a *multiplicative*
    (percent) scale: beta1 is the trend in percent change per period,
    beta2 is the percent difference between the two seasons -- reported
    that way in `params` below.

Priors are weakly informative, centered on the observed data's own
log-mean, and deliberately tightened for small-N stability (see comments
inline) -- with only a handful of points, vague priors on a log scale
blow up once exponentiated.
"""

from typing import List, Dict, Any, Literal
import numpy as np
import pymc as pm
import pandas as pd


def fit_and_forecast(
    df: pd.DataFrame,
    future_periods: List[Dict[str, Any]],
    draws: int = 1000,
    tune: int = 1500,
    chains: int = 4,
    seed: int = 42,
    likelihood: Literal["poisson", "negative_binomial"] = "poisson",
) -> Dict[str, Any]:
    """
    df: must contain columns t, season_idx, admissions (historical data, counts >= 0)
    future_periods: list of {"t": int, "season_idx": 0/1, "period_label": str}

    Returns a dict with per-period forecasts (mean, median, 50%/95% credible
    intervals -- all guaranteed >= 0, all integers) and posterior summaries
    for the trend and seasonal effect, reported as percent change.
    """
    t = df["t"].values.astype(float)
    season_idx = df["season_idx"].values.astype(float)
    y = df["admissions"].values.astype(float)

    if (y < 0).any():
        raise ValueError("Admissions values must be non-negative counts.")

    t_mean, t_std = t.mean(), (t.std() or 1.0)
    t_scaled = (t - t_mean) / t_std

    log_y_mean = float(np.log(max(y.mean(), 1e-3)))

    with pm.Model():
        beta0 = pm.Normal("beta0", mu=log_y_mean, sigma=0.5)  # log baseline level
        beta1 = pm.Normal("beta1", mu=0, sigma=0.3)  # log-trend, scaled t
        beta2 = pm.Normal("beta2", mu=0, sigma=0.4)  # log seasonal effect

        log_mu = beta0 + beta1 * t_scaled + beta2 * season_idx
        mu = pm.math.exp(log_mu)

        if likelihood == "poisson":
            pm.Poisson("y_obs", mu=mu, observed=y)
        elif likelihood == "negative_binomial":
            alpha = pm.Gamma("alpha", alpha=5, beta=1)  # dispersion; mean 5, still weak
            pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y)
        else:
            raise ValueError(f"Unknown likelihood: {likelihood!r}")

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            target_accept=0.98,
        )

    post = trace.posterior
    beta0_s = post["beta0"].values.flatten()
    beta1_s = post["beta1"].values.flatten()
    beta2_s = post["beta2"].values.flatten()
    alpha_s = (
        post["alpha"].values.flatten() if likelihood == "negative_binomial" else None
    )

    rng = np.random.default_rng(seed)
    forecasts = []
    for fp in future_periods:
        t_scaled_f = (fp["t"] - t_mean) / t_std
        mu_samples = np.exp(beta0_s + beta1_s * t_scaled_f + beta2_s * fp["season_idx"])

        if likelihood == "poisson":
            y_samples = rng.poisson(mu_samples)
        else:
            # NegativeBinomial(mu, alpha) <-> numpy's (n=alpha, p=alpha/(alpha+mu))
            p_samples = alpha_s / (alpha_s + mu_samples)
            y_samples = rng.negative_binomial(alpha_s, p_samples)

        forecasts.append(
            {
                "period_label": fp["period_label"],
                "t": fp["t"],
                "mean": float(np.mean(y_samples)),
                "median": float(np.median(y_samples)),
                "lower_95": float(np.percentile(y_samples, 2.5)),
                "upper_95": float(np.percentile(y_samples, 97.5)),
                "lower_50": float(np.percentile(y_samples, 25)),
                "upper_50": float(np.percentile(y_samples, 75)),
            }
        )

    trend_pct = (np.exp(beta1_s / t_std) - 1) * 100  # % change per period, real units
    season_pct = (np.exp(beta2_s) - 1) * 100  # % effect of season_idx=1 vs season_idx=0

    params = {
        "trend_pct_per_period": {
            "mean": float(np.mean(trend_pct)),
            "lower_95": float(np.percentile(trend_pct, 2.5)),
            "upper_95": float(np.percentile(trend_pct, 97.5)),
        },
        "seasonal_effect_pct": {
            "mean": float(np.mean(season_pct)),
            "lower_95": float(np.percentile(season_pct, 2.5)),
            "upper_95": float(np.percentile(season_pct, 97.5)),
        },
        "likelihood": likelihood,
    }
    if alpha_s is not None:
        params["dispersion_alpha"] = {"mean": float(np.mean(alpha_s))}

    return {"forecasts": forecasts, "params": params}


def point_forecast(
    train_df: pd.DataFrame,
    target_period: Dict[str, Any],
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    likelihood: Literal["poisson", "negative_binomial"] = "poisson",
    seed: int = 42,
) -> float:
    """
    Convenience wrapper for backtesting: fit on `train_df`, forecast a
    single `target_period`, return just the posterior mean point forecast.
    Uses smaller draws/tune/chains than a "real" forecast by default, since
    this gets called once per backtest point and only a point estimate is
    needed -- not the full interval.
    """
    result = fit_and_forecast(
        train_df,
        [target_period],
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        likelihood=likelihood,
    )
    return result["forecasts"][0]["mean"]
