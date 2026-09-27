"""Kaplan-Meier survival analysis: the corrected version of the naive
model's 3-year 'survived' estimate. Unlike the naive approach (which
drops every company that hasn't had enough time to reveal an outcome),
this treats those companies as *censored* -- we don't know their outcome
yet, but that's different from assuming they failed or excluding them.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter

try:
    from .load_panel import load_all_quarters
except ImportError:
    from load_panel import load_all_quarters


def build_duration_table(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per company: time-to-second-filing (the 'event'), or
    time-to-as_of if no second filing has happened yet (censored)."""
    grouped = panel.sort_values("FILING_DATE").groupby("CIK")["FILING_DATE"]

    first_filing = grouped.min()
    filing_counts = grouped.count()

    # second filing date, if it exists, else NaT
    def second_or_nat(dates):
        sorted_dates = dates.sort_values().values
        return sorted_dates[1] if len(sorted_dates) > 1 else pd.NaT

    second_filing = grouped.apply(second_or_nat)

    table = pd.DataFrame({
        "first_filing": first_filing,
        "second_filing": second_filing,
        "n_filings": filing_counts,
    }).reset_index()

    event_observed = table["second_filing"].notna()
    event_time = table["second_filing"].fillna(as_of)

    duration_days = (event_time - table["first_filing"]).dt.days
    table["duration_years"] = duration_days / 365.25
    table["event_observed"] = event_observed.astype(int)

    return table


def run_survival_analysis():
    panel = load_all_quarters()
    panel = panel[panel["is_operating_company"]]
    as_of = panel["FILING_DATE"].max()

    table = build_duration_table(panel, as_of)
    # drop any zero/negative durations (data artifacts, e.g. same-day duplicate filings)
    table = table[table["duration_years"] > 0]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=table["duration_years"], event_observed=table["event_observed"])

    survival_at_3y = kmf.survival_function_at_times(3).iloc[0]
    corrected_refile_prob_3y = 1 - survival_at_3y

    # naive comparison: just the observable subset, dropping censored rows,
    # exactly like naive.py did
    observable = table[table["first_filing"] + pd.DateOffset(years=3) <= as_of]
    naive_refile_prob_3y = observable["event_observed"].mean()

    print(f"Total companies in survival analysis: {len(table)}")
    print(f"Companies with observed 2nd filing (event): {table['event_observed'].sum()}")
    print(f"Companies censored (no 2nd filing yet, or ever): {(1 - table['event_observed']).sum()}")
    print()
    print(f"NAIVE estimate (drop censored companies): {naive_refile_prob_3y:.3f}")
    print(f"CORRECTED estimate (Kaplan-Meier, uses all companies): {corrected_refile_prob_3y:.3f}")
    print(f"Difference: {corrected_refile_prob_3y - naive_refile_prob_3y:+.3f}")

    return kmf, table

def compare_cohorts_by_year(table: pd.DataFrame):
    """Fit separate Kaplan-Meier curves by first-filing cohort year, to
    test whether early-duration hazard actually shifted over time (the
    mechanism we suspect explains the naive-vs-corrected gap)."""
    table = table.copy()
    table["cohort"] = pd.cut(
        table["first_filing"].dt.year,
        bins=[2014, 2017, 2020, 2023],
        labels=["2015-2017", "2018-2020", "2021-2023"],
    )

    print("\n--- Cohort comparison: probability of refiling within 1 year ---")
    for cohort_label, group in table.groupby("cohort", observed=True):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=group["duration_years"], event_observed=group["event_observed"])
        refile_1y = 1 - kmf.survival_function_at_times(1).iloc[0]
        print(f"{cohort_label}: n={len(group):>6}  P(refile within 1yr) = {refile_1y:.3f}")


if __name__ == "__main__":
    kmf, table = run_survival_analysis()
    compare_cohorts_by_year(table)