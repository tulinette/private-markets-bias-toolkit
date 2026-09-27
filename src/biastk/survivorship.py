"""Label companies as 'survived' or not based on whether they filed
again at least SURVIVAL_YEARS after their first observed filing."""

import pandas as pd

SURVIVAL_YEARS = 3


def build_survival_labels(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Given the full filings panel, return one row per company (CIK)
    with its first filing date and whether it survived.

    A company's outcome is only *observable* if enough time (SURVIVAL_YEARS)
    has passed since its first filing, relative to `as_of` (the last date
    we have data for). Companies without enough elapsed time get
    observable=False — this is exactly the survivorship-bias mechanism:
    a naive model would silently drop these rows and never mention it.
    """
    first_filing = panel.groupby("CIK")["FILING_DATE"].min().rename("first_filing")
    last_filing = panel.groupby("CIK")["FILING_DATE"].max().rename("last_filing")

    companies = pd.concat([first_filing, last_filing], axis=1).reset_index()

    horizon = companies["first_filing"] + pd.DateOffset(years=SURVIVAL_YEARS)
    companies["observable"] = horizon <= as_of
    companies["survived"] = (companies["last_filing"] >= horizon).astype("boolean")

    # survived is only meaningful where observable is True
    companies.loc[~companies["observable"], "survived"] = pd.NA

    return companies


if __name__ == "__main__":  # pragma: no cover
    from load_panel import load_all_quarters

    panel = load_all_quarters()
    panel = panel[panel["is_operating_company"]]
    print(f"\nFilings after excluding pooled investment funds: {len(panel)}")

    as_of = panel["FILING_DATE"].max()
    print(f"Data runs through: {as_of}")

    labels = build_survival_labels(panel, as_of)
    print(f"\nTotal companies: {len(labels)}")
    print(f"Observable (enough time elapsed): {labels['observable'].sum()}")
    print(f"Not yet observable: {(~labels['observable']).sum()}")
    print(f"\nOf observable companies:")
    print(labels.loc[labels['observable'], 'survived'].value_counts())