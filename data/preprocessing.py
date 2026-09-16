"""
MPLADS Sentinel — Data Preprocessing
======================================
Loads and cleans Works Sanctioned.csv.

Actual columns (inspected from CSV):
  Sr. No. | Work category | Work | State | IDA | Hon'ble Members of Parliament |
  Constituency | Work description | Recommended date | Sanction Date |
  Sanction Amount ( ₹ ) | Work Status
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import os

# ── Constants ────────────────────────────────────────────────────────────────

CSV_FILENAME = "Works Sanctioned.csv"

# Canonical internal column names mapped from actual CSV headers
COL_MAP = {
    "Sr. No.": "sr_no",
    "Work category": "work_category",
    "Work": "work_code",
    "State": "state",
    "IDA": "ida",
    "Hon'ble Members of Parliament": "mp_name",
    "Constituency": "constituency",
    "Work description": "work_description",
    "Recommended date": "recommended_date",
    "Sanction Date": "sanction_date",
    "Sanction Amount ( ₹ )": "sanction_amount",
    "Work Status": "work_status",
}

# Status groupings
INCOMPLETE_STATUSES = {
    "Vendor Identification",
    "Physical Inspection",
    "Sanction",
    "Time Estimation",
    "Work partially Completed",
    "Work in Progress",
}
COMPLETE_STATUSES = {"Work Completed"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_csv(app_dir: Path) -> Path:
    """Locate the CSV relative to the app or in parent dirs."""
    candidates = [
        app_dir / CSV_FILENAME,
        app_dir.parent / CSV_FILENAME,
        Path(CSV_FILENAME),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find '{CSV_FILENAME}'. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _extract_district(ida: str) -> str:
    """Extract district name from IDA column, e.g. 'SAMBHAL(...)' → 'Sambhal'."""
    if pd.isna(ida):
        return "Unknown"
    match = re.match(r"^([^(]+)", str(ida).strip())
    if match:
        return match.group(1).strip().title()
    return str(ida).strip().title()


def _extract_work_sub_category(work_code: str) -> str:
    """
    Extract human-readable sub-category from the Work code column.
    e.g. 'WS/MP187/2023-2024/1199-Construction of rooms and halls in school and colleges'
    → 'Construction of rooms and halls in school and colleges'
    """
    if pd.isna(work_code):
        return "Unknown"
    parts = str(work_code).split("-", maxsplit=4)
    # The sub-category is after the last numeric segment
    if len(parts) >= 2:
        return parts[-1].strip()
    return str(work_code).strip()


def _extract_financial_year(work_code: str) -> str:
    """Extract FY like '2023-2024' from work code."""
    match = re.search(r"(\d{4}-\d{4})", str(work_code))
    if match:
        return match.group(1)
    return "Unknown"


def _parse_amount(val) -> float:
    """Parse sanction amount — handles commas and strings."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("₹", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_date(s, fmt="%d-%b-%Y"):
    return pd.to_datetime(s, format=fmt, errors="coerce")


# ── Main loader (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading MPLADS data…", ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Load, clean, and enrich the MPLADS Works Sanctioned CSV.
    Returns a clean DataFrame with internal column names.
    """
    app_dir = Path(__file__).resolve().parent.parent  # mplads-sentinel/
    csv_path = _find_csv(app_dir)

    # ── Read ────────────────────────────────────────────────────────────────
    raw = pd.read_csv(
        csv_path,
        encoding="utf-8",
        dtype=str,          # read everything as string first
        low_memory=False,
    )

    # Drop the last "Grand Total" row if present
    raw = raw[raw.iloc[:, 0].str.strip() != "Grand Total"].copy()

    # ── Rename ──────────────────────────────────────────────────────────────
    df = raw.rename(columns=COL_MAP)

    # Keep only known columns (drop any unnamed extras)
    known_cols = list(COL_MAP.values())
    df = df[[c for c in known_cols if c in df.columns]].copy()

    # ── Type casting ────────────────────────────────────────────────────────
    df["sanction_amount"] = df["sanction_amount"].apply(_parse_amount)
    df["recommended_date"] = df["recommended_date"].apply(_parse_date)
    df["sanction_date"] = df["sanction_date"].apply(_parse_date)

    # ── Derived columns ─────────────────────────────────────────────────────
    df["district"] = df["ida"].apply(_extract_district)
    df["sub_category"] = df["work_code"].apply(_extract_work_sub_category)
    df["financial_year"] = df["work_code"].apply(_extract_financial_year)

    # MP name cleanup — strip tenure bracket e.g. "(2022-28)"
    df["mp_name_clean"] = df["mp_name"].str.replace(r"\s*\(\d{4}-\d{2,4}\)", "", regex=True).str.strip()

    # Recommendation → Sanction gap (days)
    df["rec_to_sanction_days"] = (
        df["sanction_date"] - df["recommended_date"]
    ).dt.days

    # Days since sanction (as of today)
    today = pd.Timestamp.today().normalize()
    df["days_since_sanction"] = (today - df["sanction_date"]).dt.days

    # Amount in lakhs (for display)
    df["amount_lakhs"] = (df["sanction_amount"] / 1e5).round(2)

    # Completion flag
    df["is_completed"] = df["work_status"].isin(COMPLETE_STATUSES)
    df["is_incomplete"] = df["work_status"].isin(INCOMPLETE_STATUSES) | ~df["is_completed"]

    # Numeric row index for ML (use sr_no or positional)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics for the dashboard."""
    today = pd.Timestamp.today().normalize()

    total = len(df)
    total_amount = df["sanction_amount"].sum()
    completed = df["is_completed"].sum()
    incomplete = df["is_incomplete"].sum()

    # Old pending: not completed AND sanctioned more than 365 days ago
    old_threshold = 365
    old_pending = df[
        ~df["is_completed"] & (df["days_since_sanction"] > old_threshold)
    ]

    return {
        "total_works": total,
        "total_amount": total_amount,
        "total_amount_cr": round(total_amount / 1e7, 2),
        "completed": int(completed),
        "incomplete": int(incomplete),
        "old_pending": len(old_pending),
        "unique_states": df["state"].nunique(),
        "unique_mps": df["mp_name_clean"].nunique(),
        "unique_districts": df["district"].nunique(),
        "financial_years": sorted(df["financial_year"].unique().tolist()),
    }
