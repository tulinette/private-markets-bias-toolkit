"""Verify the top-level pipeline functions (run_naive_model,
run_survival_analysis, compare_cohorts_by_year, load_all_quarters) run
end-to-end without error, using a mocked/synthetic panel instead of
real downloaded data."""

import numpy as np
import pandas as pd
from unittest.mock import patch

import biastk.naive as naive
import biastk.survival_analysis as survival_analysis
from biastk.survival_analysis import run_cox_year_trend_model
import biastk.load_panel as load_panel
from tests.test_load_panel import make_fake_quarter


def build_rich_synthetic_panel(n=80, seed=1):
    as_of = pd.Timestamp("2023-01-01")
    rng = np.random.default_rng(seed)
    states = ["CA", "NY", "TX", "MA"]
    entity_types = ["Corporation", "Limited Liability Company"]
    industries = ["Other Technology", "Manufacturing", "Biotechnology", "Other"]

    years_since_first = rng.uniform(0.5, 6.0, size=n)
    first_filing = as_of - pd.to_timedelta(years_since_first * 365.25, unit="D")

    gap_to_second = rng.exponential(scale=1 / 0.3, size=n)
    has_second = gap_to_second <= years_since_first
    second_filing = first_filing + pd.to_timedelta(gap_to_second * 365.25, unit="D")

    rows = []
    for i in range(n):
        cik = f"C{i:05d}"
        common = {
            "CIK": cik, "ENTITYNAME": f"Company {i}", "quarter": "synthetic",
            "is_operating_company": True,
            "STATEORCOUNTRY": states[i % len(states)],
            "ENTITYTYPE": entity_types[i % len(entity_types)],
            "INDUSTRYGROUPTYPE": industries[i % len(industries)],
            "TOTALOFFERINGAMOUNT": str(10000 * (i + 1)),
        }
        rows.append({**common, "FILING_DATE": first_filing[i]})
        if has_second[i]:
            rows.append({**common, "FILING_DATE": second_filing[i]})

    return pd.DataFrame(rows)


def test_run_naive_model_executes_end_to_end():
    panel = build_rich_synthetic_panel()
    with patch("biastk.naive.load_all_quarters", return_value=panel):
        model, data = naive.run_naive_model()
    assert len(data) > 0
    assert "survived" in data.columns


def test_run_survival_analysis_executes_end_to_end():
    panel = build_rich_synthetic_panel()
    with patch("biastk.survival_analysis.load_all_quarters", return_value=panel):
        kmf, table = survival_analysis.run_survival_analysis()
    assert len(table) > 0
    survival_analysis.compare_cohorts_by_year(table)


def test_load_all_quarters_concatenates_multiple_quarters(tmp_path, monkeypatch):
    make_fake_quarter(tmp_path)  # writes into tmp_path / "2019q1"
    monkeypatch.setattr(load_panel, "RAW_DIR", tmp_path)

    result = load_panel.load_all_quarters()
    assert len(result) == 2
    assert set(result["CIK"]) == {"C001", "C002"}

def test_run_cox_year_trend_model_executes_end_to_end(monkeypatch):
    panel = build_rich_synthetic_panel()
    monkeypatch.setattr("biastk.survival_analysis.load_all_quarters", lambda: panel)
    cph = run_cox_year_trend_model()
    assert "first_filing_year" in cph.hazard_ratios_