import pandas as pd
import numpy as np

def fmt_inr(amount, decimals=2):
    if pd.isna(amount) or amount == 0:
        return "\u20b9 N/A"
    if amount >= 1e7:
        return f"\u20b9 {amount/1e7:.{decimals}f} Cr"
    if amount >= 1e5:
        return f"\u20b9 {amount/1e5:.{decimals}f} L"
    return f"\u20b9 {amount:,.0f}"

def fmt_inr_short(amount):
    if pd.isna(amount):
        return "N/A"
    if amount >= 1e9:
        return f"\u20b9{amount/1e9:.1f}B"
    if amount >= 1e7:
        return f"\u20b9{amount/1e7:.1f}Cr"
    if amount >= 1e5:
        return f"\u20b9{amount/1e5:.1f}L"
    return f"\u20b9{amount:,.0f}"

def fmt_date(dt):
    if pd.isna(dt):
        return "N/A"
    try:
        return pd.Timestamp(dt).strftime("%d %b %Y")
    except Exception:
        return str(dt)

RISK_COLORS = {
    "HIGH":   ("#7f1d1d", "#fca5a5", "\U0001f534"),
    "MEDIUM": ("#78350f", "#fde68a", "\U0001f7e1"),
    "LOW":    ("#14532d", "#bbf7d0", "\U0001f7e2"),
}

def risk_badge_html(level, score=None):
    bg, fg, emoji = RISK_COLORS.get(level, ("#1e293b", "#e2e8f0", "\u26aa"))
    score_str = f" {score:.0f}/100" if score is not None else ""
    return (
        f'<span style="background:{fg};color:{bg};padding:3px 10px;'
        f'border-radius:12px;font-weight:700;font-size:0.8rem;">'
        f'{emoji} {level}{score_str}</span>'
    )

def risk_badge_md(level):
    emoji = RISK_COLORS.get(level, ("", "", "\u26aa"))[2]
    return f"{emoji} {level}"

STATUS_COLORS = {
    "Work Completed":           "\U0001f7e2",
    "Work partially Completed": "\U0001f7e1",
    "Physical Inspection":      "\U0001f535",
    "Vendor Identification":    "\U0001f7e0",
    "Sanction":                 "\U0001f7e4",
    "Time Estimation":          "\u26aa",
    "Work in Progress":         "\U0001f535",
}

def status_emoji(status):
    return STATUS_COLORS.get(str(status).strip(), "\u26ab")

def safe_int(val):
    try:
        return int(val)
    except Exception:
        return 0

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return 0.0

AVAILABLE_FIELDS = [
    "Work category", "Work code / ID", "State", "District (IDA)",
    "MP Name", "Constituency", "Work description",
    "Recommended date", "Sanction date", "Sanction amount (\u20b9)", "Work status",
]

MISSING_FIELDS = [
    "Actual expenditure / amount spent", "Physical progress percentage",
    "Payment / installment history", "Expected completion date",
    "Actual completion date", "GPS coordinates / geo-tag",
    "Asset photographs", "Contractor / vendor details",
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
