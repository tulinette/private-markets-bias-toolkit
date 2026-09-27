"""Verify build_first_filing_features() only uses each company's
FIRST filing -- the core no-lookahead claim of the naive baseline."""

import pandas as pd
from biastk.naive import build_first_filing_features


def test_uses_first_filing_features_not_later_ones():
    panel = pd.DataFrame({
        "CIK": ["A", "A"],
        "FILING_DATE": pd.to_datetime(["2015-01-01", "2019-06-01"]),
        "STATEORCOUNTRY": ["CA", "NY"],       # company "moved" between filings
        "ENTITYTYPE": ["Corporation", "Corporation"],
        "INDUSTRYGROUPTYPE": ["Other Technology", "Manufacturing"],  # changed too
        "TOTALOFFERINGAMOUNT": ["100000", "5000000"],  # raised much more later
    })

    features = build_first_filing_features(panel).set_index("CIK")

    # must reflect the FIRST filing (2015, CA, Other Technology, 100000)
    # not the second (2019, NY, Manufacturing, 5000000)
    assert features.loc["A", "STATEORCOUNTRY"] == "CA"
    assert features.loc["A", "INDUSTRYGROUPTYPE"] == "Other Technology"