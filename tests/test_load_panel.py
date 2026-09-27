"""Verify load_quarter()'s three-way join and fund/operating-company
filter, using tiny fake TSV files instead of real downloaded data."""

import pandas as pd
from pathlib import Path
from biastk.load_panel import load_quarter


def make_fake_quarter(tmp_path: Path) -> Path:
    quarter_dir = tmp_path / "2019q1"
    inner = quarter_dir / "2019Q1_d"
    inner.mkdir(parents=True)

    pd.DataFrame({
        "ACCESSIONNUMBER": ["ACC1", "ACC2"],
        "FILING_DATE": ["01-JAN-2019", "02-JAN-2019"],
    }).to_csv(inner / "FORMDSUBMISSION.tsv", sep="\t", index=False)

    pd.DataFrame({
        "ACCESSIONNUMBER": ["ACC1", "ACC2"],
        "IS_PRIMARYISSUER_FLAG": ["YES", "YES"],
        "CIK": ["C001", "C002"],
        "ENTITYNAME": ["Real Operating Co", "Some Fund LP"],
        "STATEORCOUNTRY": ["CA", "NY"],
        "ENTITYTYPE": ["Corporation", "Limited Partnership"],
    }).to_csv(inner / "ISSUERS.tsv", sep="\t", index=False)

    pd.DataFrame({
        "ACCESSIONNUMBER": ["ACC1", "ACC2"],
        "INDUSTRYGROUPTYPE": ["Manufacturing", "Pooled Investment Fund"],
        "ISPOOLEDINVESTMENTFUNDTYPE": ["", "true"],
        "TOTALOFFERINGAMOUNT": ["100000", "5000000"],
    }).to_csv(inner / "OFFERING.tsv", sep="\t", index=False)

    return quarter_dir


def test_load_quarter_joins_and_flags_funds_correctly(tmp_path):
    quarter_dir = make_fake_quarter(tmp_path)
    result = load_quarter(quarter_dir).set_index("CIK")

    assert result.loc["C001", "ENTITYNAME"] == "Real Operating Co"
    assert result.loc["C001", "is_operating_company"] == True

    assert result.loc["C002", "ENTITYNAME"] == "Some Fund LP"
    assert result.loc["C002", "is_operating_company"] == False