"""Verify build_duration_table() and the Kaplan-Meier correction on a
synthetic panel shaped like the real one (CIK, FILING_DATE rows) --
exercising the actual repo code, not a reimplementation of it."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from biastk.survival_analysis import build_duration_table


def make_synthetic_panel(n, hazard_fn, as_of_years=6.0, seed=42):
    """Build a panel DataFrame shaped like the real one: one row per
    filing, with CIK and FILING_DATE."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp("2024-01-01")

    # years_since_first_filing: how long ago (relative to as_of) this
    # company's first filing happened -- this IS the elapsed/available
    # observation time, not "years before as_of" needing further inversion.
    years_since_first_filing = rng.uniform(0, as_of_years, size=n)
    first_filing_date = as_of - pd.to_timedelta(years_since_first_filing * 365.25, unit="D")

    hazard = hazard_fn(years_since_first_filing)
    true_event_gap_years = rng.exponential(scale=1 / hazard, size=n)
    time_available_years = years_since_first_filing  # fixed: was backwards before

    has_second_filing = true_event_gap_years <= time_available_years
    second_filing_date = first_filing_date + pd.to_timedelta(
        true_event_gap_years * 365.25, unit="D"
    )

    rows = []
    for i in range(n):
        cik = f"C{i:06d}"
        rows.append({"CIK": cik, "FILING_DATE": first_filing_date[i]})
        if has_second_filing[i]:
            rows.append({"CIK": cik, "FILING_DATE": second_filing_date[i]})

    panel = pd.DataFrame(rows)
    return panel, as_of, years_since_first_filing, true_event_gap_years


def test_build_duration_table_matches_known_ground_truth():
    panel, as_of, _, _ = make_synthetic_panel(
        n=4000, hazard_fn=lambda offset: np.full_like(offset, 0.3)
    )

    table = build_duration_table(panel, as_of)
    table = table[table["duration_years"] > 0]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=table["duration_years"], event_observed=table["event_observed"])
    km_estimate = 1 - kmf.survival_function_at_times(3).iloc[0]

    true_rate = 1 - np.exp(-0.3 * 3)
    assert abs(km_estimate - true_rate) < 0.03


def test_kaplan_meier_beats_naive_on_real_code_path_with_cohort_drift():
    def hazard_fn(offset):
        return np.where(offset > 3.0, 0.4, 0.2)

    panel, as_of, years_since_first_filing, true_event_gap_years = make_synthetic_panel(
        n=4000, hazard_fn=hazard_fn, seed=7
    )

    table = build_duration_table(panel, as_of)
    table = table[table["duration_years"] > 0]

    true_rate = (true_event_gap_years <= 3.0).mean()

    observable_mask = years_since_first_filing >= 3.0  # fixed: was backwards before
    naive = (true_event_gap_years[observable_mask] <= 3.0).mean()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=table["duration_years"], event_observed=table["event_observed"])
    km = 1 - kmf.survival_function_at_times(3).iloc[0]

    naive_error = abs(naive - true_rate)
    km_error = abs(km - true_rate)

    assert km_error < naive_error