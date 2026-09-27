"""Download SEC Form D filings (structured data sets) for use as our
'population of companies that raised private capital' dataset."""

import requests
import zipfile
import io
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# SEC.gov requires a descriptive User-Agent identifying the requester —
# generic/missing User-Agents get a 403. Keep this accurate to your identity.
HEADERS = {"User-Agent": "Tuline Dachraoui research project tuline.dachraoui@epfl.ch"}


def fetch_form_d_quarter(year: int, quarter: int) -> Path:
    """Download one quarter's Form D structured data set from SEC.gov.
    Example: fetch_form_d_quarter(2026, 2) gets Q2 2026.
    """
    url = (
        "https://www.sec.gov/files/structureddata/data/"
        f"form-d-data-sets/{year}q{quarter}_d.zip"
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RAW_DIR / f"{year}q{quarter}"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(out_dir)

    return out_dir

def fetch_form_d_range(start_year: int, start_quarter: int, end_year: int, end_quarter: int):
    """Download every quarter from (start_year, start_quarter) through
    (end_year, end_quarter) inclusive."""
    y, q = start_year, start_quarter
    downloaded = []
    while (y, q) <= (end_year, end_quarter):
        print(f"Fetching {y}Q{q}...")
        path = fetch_form_d_quarter(y, q)
        downloaded.append(path)
        q += 1
        if q > 4:
            q = 1
            y += 1
    return downloaded


if __name__ == "__main__":
    paths = fetch_form_d_range(2015, 1, 2023, 4)
    print(f"Downloaded {len(paths)} quarters:")
    for p in paths:
        print(" -", p)