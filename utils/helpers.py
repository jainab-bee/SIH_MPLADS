"""
MPLADS Sentinel — Shared Helpers
=================================
Formatting, display utilities, and common constants.
"""

import pandas as pd
import numpy as np


# ── Currency formatting ───────────────────────────────────────────────────

def fmt_inr(amount: float, decimals: int = 2) -> str:
    """Format a rupee amount as ₹X.XXL (lakhs) or ₹X.XXCr (crores)."""
    if pd.isna(amount) or amount == 0:
        return "₹ N/A"
    if amount >= 1e7:
        return f"₹ {amount/1e7:.{decimals}f} Cr"
    if amount >= 1e5:
        return f"₹ {amount/1e5:.{decimals}f} L"
    return f"₹ {amount:,.0f}"


def fmt_inr_short(amount: float) -> str:
    """Short form for KPI cards."""
    if pd.isna(amount):
        return "N/A"
    if amount >= 1e9:
        return f"₹{amount/1e9:.1f}B"
    if amount >= 1e7:
        return f"₹{amount/1e7:.1f}Cr"
    if amount >= 1e5:
        return f"₹{amount/1e5:.1f}L"
    return f"₹{amount:,.0f}"


# ── Date formatting ───────────────────────────────────────────────────────

def fmt_date(dt) -> str:
    if pd.isna(dt):
        return "N/A"
    try:
        return pd.Timestamp(dt).strftime("%d %b %Y")
    except Exception:
        return str(dt)


# ── Risk badge HTML ───────────────────────────────────────────────────────

RISK_COLORS = {
    "HIGH":   ("#7f1d1d", "#fca5a5", "🔴"),
    "MEDIUM": ("#78350f", "#fde68a", "🟡"),
    "LOW":    ("#14532d", "#bbf7d0", "🟢"),
}

def risk_badge_html(level: str, score: float = None) -> str:
    bg, fg, emoji = RISK_COLORS.get(level, ("#1e293b", "#e2e8f0", "⚪"))
    score_str = f" {score:.0f}/100" if score is not None else ""
    return (
        f'<span style="background:{fg};color:{bg};padding:3px 10px;'
        f'border-radius:12px;font-weight:700;font-size:0.8rem;">'
        f'{emoji} {level}{score_str}</span>'
    )


def risk_badge_md(level: str) -> str:
    """Return a simple text badge for use in st.dataframe columns."""
    emoji = RISK_COLORS.get(level, ("", "", "⚪"))[2]
    return f"{emoji} {level}"


# ── Status badge ─────────────────────────────────────────────────────────

STATUS_COLORS = {
    "Work Completed":           "🟢",
    "Work partially Completed": "🟡",
    "Physical Inspection":      "🔵",
    "Vendor Identification":    "🟠",
    "Sanction":                 "🟤",
    "Time Estimation":          "⚪",
    "Work in Progress":         "🔵",
}

def status_emoji(status: str) -> str:
    return STATUS_COLORS.get(str(status).strip(), "⚫")


# ── Number helpers ────────────────────────────────────────────────────────

def safe_int(val) -> int:
    try:
        return int(val)
    except Exception:
        return 0

def safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


# ── Sidebar filter helper ────────────────────────────────────────────────

def multiselect_all(label: str, options: list, key: str) -> list:
    """
    A multiselect that defaults to ALL options selected.
    Returns the selected list.
    """
    import streamlit as st
    all_opt = ["All"] + sorted([str(o) for o in options if pd.notna(o)])
    selected = st.sidebar.multiselect(label, all_opt, default=["All"], key=key)
    if "All" in selected or not selected:
        return list(options)
    return selected


# ── Data availability manifest ────────────────────────────────────────────

AVAILABLE_FIELDS = [
    "Work category",
    "Work code / ID",
    "State",
    "District (IDA)",
    "MP Name",
    "Constituency",
    "Work description",
    "Recommended date",
    "Sanction date",
    "Sanction amount (₹)",
    "Work status",
]

MISSING_FIELDS = [
    "Actual expenditure / amount spent",
    "Physical progress percentage",
    "Payment / installment history",
    "Expected completion date",
    "Actual completion date",
    "GPS coordinates / geo-tag",
    "Asset photographs",
    "Contractor / vendor details",
    "Amendment / revision history",
]

FUTURE_IMPROVEMENTS = [
    "Connect to live eSAKSHI API for real-time data",
    "Add expenditure analysis once payment data available",
    "Add geo-tagged progress photos verification",
    "Integrate GPS coordinates for spatial overlap detection",
    "Add contractor blacklist matching",
    "Incorporate CAG audit reports for validation",
]
