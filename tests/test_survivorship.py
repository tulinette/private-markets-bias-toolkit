"""Verify build_survival_labels(): observability and survival flags."""

import pandas as pd
from biastk.survivorship import build_survival_labels


def test_observable_and_survived_flags():
    panel = pd.DataFrame({
        "CIK": ["A", "A", "B", "C", "C"],
        "FILING_DATE": pd.to_datetime([
            "2015-01-01", "2019-06-01",   # A: first 2015, refiled 2019 (4.4y later)
            "2015-01-01",                  # B: first 2015, never refiled
            "2022-01-01", "2022-06-01",    # C: first 2022, refiled same year (too recent to be "observable" at 3y)
        ]),
    })
    as_of = pd.Timestamp("2023-01-01")

    labels = build_survival_labels(panel, as_of).set_index("CIK")

    # A: first filing 2015-01-01, +3y = 2018-01-01 <= as_of (2023) -> observable
    #    last filing 2019-06-01 >= 2018-01-01 -> survived
    assert labels.loc["A", "observable"] == True
    assert labels.loc["A", "survived"] == True

    # B: first filing 2015-01-01, observable, but never refiled -> not survived
    assert labels.loc["B", "observable"] == True
    assert labels.loc["B", "survived"] == False

    # C: first filing 2022-01-01, +3y = 2025-01-01 > as_of (2023) -> NOT observable
    assert labels.loc["C", "observable"] == False
    assert pd.isna(labels.loc["C", "survived"])