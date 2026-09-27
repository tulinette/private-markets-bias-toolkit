"""Load Form D filings across quarters into one panel: for each
company (CIK), which quarters did it file in."""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_quarter(quarter_dir: Path) -> pd.DataFrame:
    inner = next(quarter_dir.glob("*_d"))

    submissions = pd.read_csv(
        inner / "FORMDSUBMISSION.tsv", sep="\t",
        usecols=["ACCESSIONNUMBER", "FILING_DATE"],
        dtype=str,
    )
    issuers = pd.read_csv(
        inner / "ISSUERS.tsv", sep="\t",
        usecols=["ACCESSIONNUMBER", "CIK", "ENTITYNAME", "IS_PRIMARYISSUER_FLAG",
                  "STATEORCOUNTRY", "ENTITYTYPE"],
        dtype=str,
    )
    offering = pd.read_csv(
        inner / "OFFERING.tsv", sep="\t",
        usecols=["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "ISPOOLEDINVESTMENTFUNDTYPE",
                  "TOTALOFFERINGAMOUNT"],
        dtype=str,
    )

    issuers = issuers[issuers["IS_PRIMARYISSUER_FLAG"] == "YES"]

    merged = issuers.merge(submissions, on="ACCESSIONNUMBER", how="inner")
    merged = merged.merge(offering, on="ACCESSIONNUMBER", how="left")
    merged["FILING_DATE"] = pd.to_datetime(merged["FILING_DATE"], format="mixed")
    merged["quarter"] = quarter_dir.name

    is_fund = (
        (merged["INDUSTRYGROUPTYPE"] == "Pooled Investment Fund")
        | (merged["ISPOOLEDINVESTMENTFUNDTYPE"] == "true")
    )
    merged["is_operating_company"] = ~is_fund

    cols = ["CIK", "ENTITYNAME", "FILING_DATE", "quarter", "is_operating_company",
            "STATEORCOUNTRY", "ENTITYTYPE", "INDUSTRYGROUPTYPE", "TOTALOFFERINGAMOUNT"]
    return merged[cols]


def load_all_quarters() -> pd.DataFrame:
    frames = []
    for quarter_dir in sorted(RAW_DIR.iterdir()):
        if quarter_dir.is_dir():
            print(f"Loading {quarter_dir.name}...")
            frames.append(load_quarter(quarter_dir))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":  # pragma: no cover
    panel = load_all_quarters()
    print(f"\nTotal filings loaded: {len(panel)}")
    print(f"Unique companies (CIK): {panel['CIK'].nunique()}")
    print(panel.head())