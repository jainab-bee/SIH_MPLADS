"""
MPLADS Sentinel — Risk Fusion Engine
======================================
Combines signals from all detectors into a transparent 0–100
REVIEW PRIORITY SCORE with per-component explanations.

Weights (clearly defined, easy to adjust):
  Cost Anomaly     → 40%  (0–40 pts)
  Duplicate Work   → 30%  (0–30 pts)
  Delay / Pending  → 30%  (0–30 pts)

Risk levels:
   0–39  🟢 LOW
  40–69  🟡 MEDIUM
  70–100 🔴 HIGH
"""

import pandas as pd
import numpy as np
import streamlit as st

# ── Weights (sum = 100) ───────────────────────────────────────────────────

WEIGHT_COST      = 40   # max points from cost anomaly
WEIGHT_DUPLICATE = 30   # max points from duplicate/similar detection
WEIGHT_DELAY     = 30   # max points from delay/old-pending detection

# ── Risk level thresholds ─────────────────────────────────────────────────

RISK_HIGH_THRESHOLD   = 70
RISK_MEDIUM_THRESHOLD = 40


def _risk_level(score: float) -> str:
    if score >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    if score >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _risk_emoji(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")


# ── Component score normalisation ─────────────────────────────────────────

def _to_component(raw_score_01: float, weight: int) -> float:
    """Convert a 0–1 raw score to a weighted component (0–weight)."""
    return round(float(np.clip(raw_score_01, 0, 1)) * weight, 2)


# ── Explanation builder ───────────────────────────────────────────────────

def _build_explanation(row: pd.Series) -> dict:
    """
    Build a structured 'why flagged' dict for a single project row.
    """
    cost_pts  = row.get("cost_component", 0)
    dup_pts   = row.get("dup_component", 0)
    delay_pts = row.get("delay_component", 0)
    total     = row.get("risk_score", 0)
    level     = row.get("risk_level", "LOW")

    reasons = []

    if cost_pts >= 15:
        reasons.append({
            "signal": "⚠ Cost Anomaly",
            "points": f"+{cost_pts:.0f} pts",
            "detail": row.get("cost_reason", ""),
            "severity": "HIGH" if cost_pts >= 28 else "MEDIUM",
        })

    if dup_pts >= 10:
        reasons.append({
            "signal": "⚠ Potential Similar / Duplicate Work",
            "points": f"+{dup_pts:.0f} pts",
            "detail": row.get("dup_reason", ""),
            "severity": "HIGH" if dup_pts >= 22 else "MEDIUM",
        })

    if delay_pts >= 10:
        reasons.append({
            "signal": "⚠ Old Pending / Potential Delay",
            "points": f"+{delay_pts:.0f} pts",
            "detail": row.get("delay_reason", ""),
            "severity": "HIGH" if delay_pts >= 22 else "MEDIUM",
        })

    if not reasons:
        reasons.append({
            "signal": "✅ No significant anomaly detected",
            "points": "",
            "detail": "This work is within expected parameters.",
            "severity": "LOW",
        })

    return {
        "risk_score": round(total, 1),
        "risk_level": level,
        "risk_emoji": _risk_emoji(level),
        "cost_pts": cost_pts,
        "dup_pts": dup_pts,
        "delay_pts": delay_pts,
        "reasons": reasons,
        "recommended_action": _recommended_action(level),
    }


def _recommended_action(level: str) -> str:
    actions = {
        "HIGH": (
            "Recommended for priority review. Verify sanction records, "
            "physical progress, and implementing agency credentials. "
            "Consider field inspection."
        ),
        "MEDIUM": (
            "Flag for routine audit review. Cross-check with district "
            "authority records and similar works in the same area."
        ),
        "LOW": (
            "No immediate action required. Include in standard periodic review."
        ),
    }
    return actions.get(level, "Review as per standard procedure.")


# ── Main entry point ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing risk scores…", ttl=3600)
def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to already have columns from all three detectors:
      cost_anomaly_score, dup_score, delay_score

    Adds:
      cost_component    : 0–40
      dup_component     : 0–30
      delay_component   : 0–30
      risk_score        : 0–100
      risk_level        : LOW / MEDIUM / HIGH
      risk_emoji        : 🟢 / 🟡 / 🔴
      main_reason       : short string summary
    """
    result_df = df.copy()

    result_df["cost_component"] = result_df["cost_anomaly_score"].apply(
        lambda x: _to_component(x, WEIGHT_COST)
    )
    result_df["dup_component"] = result_df["dup_score"].apply(
        lambda x: _to_component(x, WEIGHT_DUPLICATE)
    )
    result_df["delay_component"] = result_df["delay_score"].apply(
        lambda x: _to_component(x, WEIGHT_DELAY)
    )

    result_df["risk_score"] = (
        result_df["cost_component"]
        + result_df["dup_component"]
        + result_df["delay_component"]
    ).clip(0, 100).round(1)

    result_df["risk_level"] = result_df["risk_score"].apply(_risk_level)
    result_df["risk_emoji"] = result_df["risk_level"].apply(_risk_emoji)

    # Short main reason for table display
    def _main_reason(row):
        parts = []
        if row["cost_component"] >= 15:
            parts.append("High cost anomaly")
        if row["dup_component"] >= 10:
            parts.append("Similar work detected")
        if row["delay_component"] >= 10:
            parts.append("Old pending work")
        return " · ".join(parts) if parts else "Within normal range"

    result_df["main_reason"] = result_df.apply(_main_reason, axis=1)

    return result_df


def get_explanation(row: pd.Series) -> dict:
    """Public API: get full WHY FLAGGED explanation for a single row."""
    return _build_explanation(row)
