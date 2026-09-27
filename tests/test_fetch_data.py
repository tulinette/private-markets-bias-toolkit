"""Verify fetch_data.py's URL construction, zip extraction, and the
multi-quarter range loop -- using a mocked network response instead of
hitting SEC.gov for real."""

import io
import zipfile
from unittest.mock import patch, MagicMock

import biastk.fetch_data as fetch_data


def make_fake_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "2019Q1_d/FORMDSUBMISSION.tsv",
            "ACCESSIONNUMBER\tFILING_DATE\nACC1\t01-JAN-2019\n",
        )
    return buf.getvalue()


def test_fetch_form_d_quarter_downloads_and_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)

    fake_response = MagicMock()
    fake_response.content = make_fake_zip_bytes()
    fake_response.raise_for_status = MagicMock()

    with patch("biastk.fetch_data.requests.get", return_value=fake_response) as mock_get:
        out_dir = fetch_data.fetch_form_d_quarter(2019, 1)

    called_url = mock_get.call_args.args[0]
    assert "2019q1_d.zip" in called_url
    assert (out_dir / "2019Q1_d" / "FORMDSUBMISSION.tsv").exists()


def test_fetch_form_d_range_iterates_correct_quarters(monkeypatch):
    calls = []

    def fake_fetch(year, quarter):
        calls.append((year, quarter))
        return f"{year}q{quarter}"

    monkeypatch.setattr(fetch_data, "fetch_form_d_quarter", fake_fetch)

    result = fetch_data.fetch_form_d_range(2020, 3, 2021, 2)

    assert calls == [(2020, 3), (2020, 4), (2021, 1), (2021, 2)]
    assert len(result) == 4