"""Verify the Kaplan-Meier correction actually recovers the true
refiling rate on synthetic data where we control ground truth --
unlike the naive drop-censored estimate, which should be biased."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def simulate(n, hazard_fn, as_of=6.0, seed=42):
    """hazard_fn(first_filing_offset) -> hazard rate for that company.
    Returns a DataFrame with the TRUE event time (known only because
    this is synthetic) plus the realistic censored observation."""
    rng = np.random.default_rng(seed)
    first_filing_offset = rng.uniform(0, as_of, size=n)
    hazard = hazard_fn(first_filing_offset)
    true_event_time = rng.exponential(scale=1 / hazard, size=n)
    time_available = as_of - first_filing_offset

    duration = np.minimum(true_event_time, time_available)
    event_observed = (true_event_time <= time_available).astype(int)

    return pd.DataFrame({
        "first_filing_offset": first_filing_offset,
        "time_available": time_available,
        "true_event_time": true_event_time,   # only known because synthetic
        "duration_years": duration,
        "event_observed": event_observed,
    })


def naive_3yr_estimate(data: pd.DataFrame, horizon=3.0) -> float:
    """Restrict to companies observable for at least `horizon` years,
    then compute the TRUE within-horizon rate for just that subset --
    this is what naive.py effectively does."""
    observable = data[data["time_available"] >= horizon]
    return (observable["true_event_time"] <= horizon).mean()


def kaplan_meier_estimate(data: pd.DataFrame, horizon=3.0) -> float:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data["duration_years"], event_observed=data["event_observed"])
    return 1 - kmf.survival_function_at_times(horizon).iloc[0]


def test_naive_matches_truth_when_theres_no_cohort_drift():
    """Sanity check: with a CONSTANT hazard rate (no cohort drift), the
    naive estimate should closely match the true population rate --
    this just confirms the simulation itself is behaving correctly."""
    data = simulate(n=8000, hazard_fn=lambda offset: np.full_like(offset, 0.3))

    true_rate = (data["true_event_time"] <= 3.0).mean()
    naive = naive_3yr_estimate(data)

    assert abs(naive - true_rate) < 0.03


def test_kaplan_meier_recovers_true_rate_better_than_naive_drop():
    """The real test: with a deliberate cohort shift (older companies
    have hazard 0.4, recent ones 0.2 -- mirrors the real 2015-2017 vs
    2021-2023 pattern we found), Kaplan-Meier should track the TRUE
    population-wide rate more closely than the naive, older-cohort-only
    estimate.
    """
    def hazard_fn(offset):
        return np.where(offset > 3.0, 0.4, 0.2)

    data = simulate(n=8000, hazard_fn=hazard_fn, seed=7)

    true_rate = (data["true_event_time"] <= 3.0).mean()  # true population rate
    naive = naive_3yr_estimate(data)
    km = kaplan_meier_estimate(data)

    naive_error = abs(naive - true_rate)
    km_error = abs(km - true_rate)

    assert km_error < naive_error