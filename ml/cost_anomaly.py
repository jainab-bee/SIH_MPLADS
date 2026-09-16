"""
MPLADS Sentinel — Cost Anomaly Detector
========================================
Uses Isolation Forest on peer groups (work_category × state)
to detect unusually high-cost sanctioned works.

No single universal threshold.  Each peer group gets its own model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Minimum number of peers required to fit a model for a group
MIN_PEER_SIZE = 8

# IsolationForest contamination — expected fraction of anomalies per group
CONTAMINATION = 0.05

# ── Peer group definition ─────────────────────────────────────────────────────

def _peer_group_key(row: pd.Series) -> str:
    """Define the peer comparison group for a given work record."""
    return f"{row['work_category']}|{row['state']}"


def _amount_bin(amount: float) -> str:
    """Bucket amount into scale bins for additional context."""
    if pd.isna(amount):
        return "unknown"
    if amount < 3e5:
        return "small"
    if amount < 10e5:
        return "medium"
    if amount < 30e5:
        return "large"
    return "very_large"


# ── Isolation Forest scoring ──────────────────────────────────────────────────

def _score_group(group_df: pd.DataFrame) -> pd.Series:
    """
    Fit an Isolation Forest on a peer group and return anomaly scores.
    Returns a Series of float scores (0–1, higher = more anomalous).
    """
    amounts = group_df["sanction_amount"].fillna(0).values.reshape(-1, 1)
    log_amounts = np.log1p(amounts)

    scaler = RobustScaler()
    X = scaler.fit_transform(log_amounts)

    iso = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)

    # decision_function: more negative = more anomalous
    raw_scores = iso.decision_function(X)

    # Normalise to 0–1 (1 = most anomalous)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s == min_s:
        normalised = np.zeros(len(raw_scores))
    else:
        normalised = 1 - (raw_scores - min_s) / (max_s - min_s)

    return pd.Series(normalised, index=group_df.index)


# ── Main entry point ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running cost anomaly detection…", ttl=3600)
def detect_cost_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each project, compute:
      - cost_anomaly_score  : 0–1 (raw IF score, normalised within peer group)
      - peer_median         : median sanction amount in the peer group
      - peer_size           : number of peers
      - peer_group_label    : human-readable group description
      - cost_anomaly_flag   : True if top CONTAMINATION percentile within group
      - cost_reason         : human-readable explanation string

    Returns original df with these new columns appended.
    """
    result_df = df.copy()
    result_df["peer_group_key"] = result_df.apply(_peer_group_key, axis=1)

    # Initialise output columns
    result_df["cost_anomaly_score"] = 0.0
    result_df["peer_median"] = np.nan
    result_df["peer_p75"] = np.nan
    result_df["peer_p90"] = np.nan
    result_df["peer_size"] = 0
    result_df["peer_group_label"] = ""
    result_df["cost_anomaly_flag"] = False
    result_df["cost_reason"] = "Within normal peer range"

    groups = result_df.groupby("peer_group_key")

    for key, group_idx in groups.groups.items():
        group = result_df.loc[group_idx].copy()
        valid = group["sanction_amount"].notna() & (group["sanction_amount"] > 0)
        valid_group = group[valid]

        peer_size = len(valid_group)
        peer_median = valid_group["sanction_amount"].median()
        peer_p75 = valid_group["sanction_amount"].quantile(0.75)
        peer_p90 = valid_group["sanction_amount"].quantile(0.90)

        result_df.loc[group_idx, "peer_median"] = peer_median
        result_df.loc[group_idx, "peer_p75"] = peer_p75
        result_df.loc[group_idx, "peer_p90"] = peer_p90
        result_df.loc[group_idx, "peer_size"] = peer_size

        cat, state = key.split("|", 1)
        label = f"{cat} / {state}"
        result_df.loc[group_idx, "peer_group_label"] = label

        if peer_size < MIN_PEER_SIZE:
            # Not enough peers for reliable ML — use simple ratio instead
            for idx in group_idx:
                amt = result_df.loc[idx, "sanction_amount"]
                if pd.notna(amt) and peer_median > 0:
                    ratio = amt / peer_median
                    score = min(1.0, max(0.0, (ratio - 1) / 4))
                    result_df.loc[idx, "cost_anomaly_score"] = round(score, 4)
                    if ratio > 3:
                        result_df.loc[idx, "cost_anomaly_flag"] = True
                        result_df.loc[idx, "cost_reason"] = (
                            f"Amount ₹{amt/1e5:.1f}L is {ratio:.1f}× peer median "
                            f"₹{peer_median/1e5:.1f}L (small peer group — {peer_size} works)"
                        )
            continue

        # Fit Isolation Forest on valid rows only
        if_scores = _score_group(valid_group)
        result_df.loc[valid_group.index, "cost_anomaly_score"] = if_scores.round(4)

        # Flag top CONTAMINATION fraction as anomalies
        threshold = if_scores.quantile(1 - CONTAMINATION)
        for idx in valid_group.index:
            amt = result_df.loc[idx, "sanction_amount"]
            score = result_df.loc[idx, "cost_anomaly_score"]
            ratio = amt / peer_median if peer_median > 0 else 1.0

            if score >= threshold:
                result_df.loc[idx, "cost_anomaly_flag"] = True
                if ratio >= 2:
                    result_df.loc[idx, "cost_reason"] = (
                        f"Amount ₹{amt/1e5:.1f}L is {ratio:.1f}× the peer median "
                        f"₹{peer_median/1e5:.1f}L for {label} ({peer_size} similar works)"
                    )
                else:
                    result_df.loc[idx, "cost_reason"] = (
                        f"Isolation Forest flagged unusual cost pattern "
                        f"in peer group: {label} ({peer_size} works)"
                    )

    # Normalise score to 0–100 contribution for risk fusion
    result_df["cost_anomaly_score_pct"] = (
        result_df["cost_anomaly_score"] * 100
    ).round(1)

    return result_df
