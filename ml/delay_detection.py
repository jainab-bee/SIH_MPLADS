"""
MPLADS Sentinel — Delay / Old Pending Work Detector
=====================================================
Uses date and status fields that actually exist in the CSV.

Available signals:
  - recommended_date   (Recommended date)
  - sanction_date      (Sanction Date)
  - work_status        (Work Status)
  - days_since_sanction (derived: today - sanction_date)
  - rec_to_sanction_days (derived: sanction_date - recommended_date)

NOT available in this CSV:
  - expected completion date
  - actual completion date
  - physical progress %
  - expenditure

We label this "Potential Delay / Old Pending Work" — NOT confirmed delay.
"""

import numpy as np
import pandas as pd
import streamlit as st

# ── Thresholds (easy to tune) ──────────────────────────────────────────────

# Works not completed after this many days since sanction = old pending
OLD_PENDING_DAYS   = 365     # 1 year (MPLADS target)
VERY_OLD_DAYS      = 730     # 2 years = very old pending
CRITICAL_OLD_DAYS  = 1095    # 3 years = critically old

# Recommendation → Sanction gap thresholds (days)
REC_TO_SANCTION_NORMAL  = 75     # Expected: 75 days per MPLADS guidelines
REC_TO_SANCTION_HIGH    = 180
REC_TO_SANCTION_VERY_HIGH = 365

# Statuses that indicate the work is still active / not complete
PENDING_STATUSES = {
    "Vendor Identification",
    "Physical Inspection",
    "Sanction",
    "Time Estimation",
    "Work partially Completed",
    "Work in Progress",
}

COMPLETE_STATUSES = {"Work Completed"}


# ── Scoring functions ──────────────────────────────────────────────────────

def _old_pending_score(row: pd.Series) -> tuple[float, str]:
    """
    Score how 'old' a non-completed work is.
    Returns (score 0–1, reason string).
    """
    if row["is_completed"]:
        return 0.0, "Work completed"

    days = row.get("days_since_sanction", np.nan)
    if pd.isna(days):
        return 0.0, "Sanction date not available"

    days = float(days)

    if days >= CRITICAL_OLD_DAYS:
        score = 1.0
        reason = (
            f"Work sanctioned {int(days)} days ago ({days/365:.1f} years) "
            f"with status '{row['work_status']}' — critically old pending"
        )
    elif days >= VERY_OLD_DAYS:
        score = 0.75 + 0.25 * (days - VERY_OLD_DAYS) / (CRITICAL_OLD_DAYS - VERY_OLD_DAYS)
        reason = (
            f"Work sanctioned {int(days)} days ago ({days/365:.1f} years) "
            f"with status '{row['work_status']}' — very old pending"
        )
    elif days >= OLD_PENDING_DAYS:
        score = 0.4 + 0.35 * (days - OLD_PENDING_DAYS) / (VERY_OLD_DAYS - OLD_PENDING_DAYS)
        reason = (
            f"Work sanctioned {int(days)} days ago "
            f"with status '{row['work_status']}' — beyond 1-year MPLADS target"
        )
    else:
        # Less than 1 year — partial score proportional to days
        score = max(0.0, (days - 90) / (OLD_PENDING_DAYS - 90)) * 0.3
        reason = f"Work in progress ({int(days)} days since sanction)"

    return round(min(1.0, score), 4), reason


def _sanction_delay_score(row: pd.Series) -> tuple[float, str]:
    """
    Score based on recommendation-to-sanction processing time.
    Returns (score 0–1, reason string).
    """
    days = row.get("rec_to_sanction_days", np.nan)
    if pd.isna(days) or days < 0:
        return 0.0, "Processing gap not calculable"

    days = float(days)

    if days >= REC_TO_SANCTION_VERY_HIGH:
        score = min(1.0, 0.7 + (days - REC_TO_SANCTION_VERY_HIGH) / 730)
        reason = f"Recommendation to sanction took {int(days)} days (expected ≤75 days)"
    elif days >= REC_TO_SANCTION_HIGH:
        score = 0.4
        reason = f"Recommendation to sanction took {int(days)} days"
    elif days >= REC_TO_SANCTION_NORMAL:
        score = 0.15
        reason = f"Recommendation to sanction took {int(days)} days (slightly above norm)"
    else:
        score = 0.0
        reason = f"Sanction processed in {int(days)} days (within guideline)"

    return round(score, 4), reason


# ── Main entry point ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running delay/pending detection…", ttl=3600)
def detect_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds delay-related columns to the DataFrame:
      - delay_score        : 0–1 composite delay risk
      - delay_flag         : True if high risk
      - delay_reason       : primary human-readable reason
      - sanction_gap_days  : rec → sanction days
      - old_pending_score  : 0–1 old-pending sub-score
      - sanction_delay_score : 0–1 sanction-gap sub-score
    """
    result_df = df.copy()

    old_scores, old_reasons = [], []
    san_scores, san_reasons = [], []

    for _, row in result_df.iterrows():
        os_, or_ = _old_pending_score(row)
        ss_, sr_ = _sanction_delay_score(row)
        old_scores.append(os_)
        old_reasons.append(or_)
        san_scores.append(ss_)
        san_reasons.append(sr_)

    result_df["old_pending_score"] = old_scores
    result_df["sanction_delay_score"] = san_scores

    # Composite: old pending weighted more heavily (70/30)
    result_df["delay_score"] = (
        result_df["old_pending_score"] * 0.70
        + result_df["sanction_delay_score"] * 0.30
    ).round(4)

    # Flag as delayed if composite > 0.4
    result_df["delay_flag"] = result_df["delay_score"] > 0.4

    # Build combined reason
    delay_reasons = []
    for i, row in result_df.iterrows():
        parts = []
        if old_reasons[row.name]:
            parts.append(old_reasons[row.name])
        if san_reasons[row.name] and san_scores[row.name] > 0.1:
            parts.append(san_reasons[row.name])
        delay_reasons.append(" | ".join(parts) if parts else "No delay signals")

    result_df["delay_reason"] = delay_reasons

    # Normalise to 0–100 for risk fusion
    result_df["delay_score_pct"] = (result_df["delay_score"] * 100).round(1)

    return result_df
