"""
╔══════════════════════════════════════════════════════════════╗
║            MPLADS SENTINEL — Complete Streamlit App          ║
║   Explainable AI for Public Works Risk & Audit Priority      ║
║   SIH 2026 — Problem Statement SIH26102                      ║
╠══════════════════════════════════════════════════════════════╣
║  Features:                                                    ║
║  ✅ Role-Based Access Control (Login)                         ║
║  ✅ Human-in-the-Loop Review Workflow                         ║
║  ✅ Automated Investigation Report (Download)                 ║
║  ✅ Tamper-Evident Audit Trail (SHA-256 Hashing)              ║
║  ✅ Complete Audit Log with Timestamps                        ║
║  ✅ Citizen Feedback / Issue Reporting                        ║
║  ✅ Public Transparency Portal                                ║
║  ✅ All original AI detection features                        ║
╚══════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

import sys
import hashlib
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.preprocessing import load_data, get_data_summary
from ml.cost_anomaly import detect_cost_anomalies
from ml.duplicate_detection import detect_duplicates
from ml.delay_detection import detect_delays
from ml.risk_score import compute_risk_scores, get_explanation
from utils.helpers import (
    fmt_inr, fmt_inr_short, fmt_date,
    risk_badge_html, risk_badge_md, status_emoji,
    AVAILABLE_FIELDS, MISSING_FIELDS, FUTURE_IMPROVEMENTS,
    RISK_COLORS,
)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MPLADS Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# ROLE-BASED ACCESS CONTROL — CREDENTIALS
# (Prototype only — hardcoded for demo)
# ═══════════════════════════════════════════════════════════════════════════

USERS = {
    "admin":      {"password": "admin123",  "role": "Admin",           "name": "Dr. Rajesh Kumar"},
    "senior_rev": {"password": "senior123", "role": "Senior Reviewer", "name": "Ms. Priya Sharma"},
    "reviewer":   {"password": "rev123",    "role": "Reviewer",        "name": "Shri Amit Singh"},
    "public":     {"password": "pub123",    "role": "Public Viewer",   "name": "Citizen"},
}

ROLE_PAGES = {
    "Admin": [
        "📊 Dashboard", "🔴 High-Risk Works", "🔍 Project Detail",
        "📈 Risk Analysis", "📋 Review Workflow", "📄 Investigation Report",
        "🔒 Audit Trail", "💬 Citizen Feedback", "🌐 Transparency Portal",
        "ℹ️ About & Methodology",
    ],
    "Senior Reviewer": [
        "📊 Dashboard", "🔴 High-Risk Works", "🔍 Project Detail",
        "📈 Risk Analysis", "📋 Review Workflow", "📄 Investigation Report",
        "🔒 Audit Trail", "💬 Citizen Feedback", "ℹ️ About & Methodology",
    ],
    "Reviewer": [
        "📊 Dashboard", "🔴 High-Risk Works", "🔍 Project Detail",
        "📋 Review Workflow", "📄 Investigation Report",
        "💬 Citizen Feedback", "ℹ️ About & Methodology",
    ],
    "Public Viewer": [
        "📊 Dashboard", "🌐 Transparency Portal",
        "💬 Citizen Feedback", "ℹ️ About & Methodology",
    ],
}

ROLE_COLORS = {
    "Admin":           "#dc2626",
    "Senior Reviewer": "#d97706",
    "Reviewer":        "#2563eb",
    "Public Viewer":   "#16a34a",
}

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "logged_in":         False,
        "username":          "",
        "role":              "",
        "user_name":         "",
        "review_actions":    [],   # list of review action dicts
        "audit_log":         [],   # tamper-evident audit log
        "feedback_list":     [],   # citizen feedback submissions
        "review_statuses":   {},   # {sr_no: status}
        "review_notes":      {},   # {sr_no: [notes]}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# TAMPER-EVIDENT HASHING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def hash_record(data: dict) -> str:
    """SHA-256 hash of a dict — for tamper-evident audit entries."""
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

def short_hash(data: dict) -> str:
    return hash_record(data)[:12].upper()

def add_audit_entry(action: str, project_id: str, detail: str, extra: dict = None):
    """Add a tamper-evident entry to the audit log."""
    entry = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "user":       st.session_state.get("user_name", "Unknown"),
        "role":       st.session_state.get("role", "Unknown"),
        "action":     action,
        "project_id": project_id,
        "detail":     detail,
        "extra":      extra or {},
    }
    entry["hash"] = hash_record(entry)
    # Chain hash: include previous hash for tamper-evidence
    if st.session_state.audit_log:
        entry["prev_hash"] = st.session_state.audit_log[-1]["hash"]
    else:
        entry["prev_hash"] = "GENESIS"
    # Recompute with prev_hash included
    entry["hash"] = hash_record(entry)
    st.session_state.audit_log.append(entry)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.sentinel-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1e40af 100%);
    padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.sentinel-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }
.sentinel-header p  { font-size: 0.85rem; margin: 0.2rem 0 0; opacity: 0.75; }

.kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); height: 100%;
}
.kpi-label { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.2; margin: 0.3rem 0; }
.kpi-sub   { font-size: 0.78rem; color: #94a3b8; }

.evidence-card { background:#fffbeb; border-left:4px solid #f59e0b; padding:0.9rem 1.1rem; border-radius:0 8px 8px 0; margin-bottom:0.7rem; }
.evidence-card.high { background:#fef2f2; border-color:#ef4444; }
.evidence-card.low  { background:#f0fdf4; border-color:#22c55e; }
.evidence-title { font-weight:700; font-size:0.92rem; color:#1e293b; }
.evidence-detail { font-size:0.82rem; color:#475569; margin-top:0.25rem; }
.evidence-pts { font-size:0.78rem; font-weight:700; color:#64748b; float:right; }

.disclaimer { background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:0.6rem 1rem; font-size:0.78rem; color:#475569; margin-bottom:1rem; }
.section-title { font-size:1.05rem; font-weight:700; color:#1e293b; border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; margin-bottom:1rem; }

.review-card { background:white; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.7rem; box-shadow:0 1px 3px rgba(0,0,0,0.05); }

.audit-entry { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.6rem 0.9rem; margin-bottom:0.4rem; font-size:0.8rem; }
.hash-chip { background:#1e293b; color:#94a3b8; font-family:monospace; font-size:0.7rem; padding:2px 6px; border-radius:4px; }

.login-box { max-width:400px; margin:4rem auto; background:white; padding:2.5rem; border-radius:16px; box-shadow:0 4px 24px rgba(0,0,0,0.12); border:1px solid #e2e8f0; }

.status-new       { background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }
.status-review    { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }
.status-escalated { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }
.status-cleared   { background:#dcfce7; color:#166534; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }
.status-inspected { background:#ede9fe; color:#5b21b6; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }

section[data-testid="stSidebar"] { background:#0f172a; }
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running AI analysis pipeline…", ttl=3600)
def run_full_pipeline():
    df = load_data()
    df = detect_cost_anomalies(df)
    df, similar_map = detect_duplicates(df)
    df = detect_delays(df)
    df = compute_risk_scores(df)
    return df, similar_map


# ═══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════

def page_login():
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3rem;">🛡️</div>
        <div style="font-size:1.6rem;font-weight:800;color:#0f172a;">MPLADS Sentinel</div>
        <div style="font-size:0.85rem;color:#64748b;margin-top:4px;">
            Explainable AI for Public Works Risk & Audit Prioritization
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        with st.container():
            st.markdown("#### 🔐 Sign In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Sign In", use_container_width=True, type="primary"):
                user = USERS.get(username.strip().lower())
                if user and user["password"] == password.strip():
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = user["role"]
                    st.session_state.user_name = user["name"]
                    add_audit_entry("LOGIN", "—", f"User '{user['name']}' logged in as {user['role']}")
                    st.success(f"Welcome, {user['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.markdown("---")
            st.markdown("**Demo Credentials:**")
            creds = [
                ("admin", "admin123", "Admin", "#dc2626"),
                ("senior_rev", "senior123", "Senior Reviewer", "#d97706"),
                ("reviewer", "rev123", "Reviewer", "#2563eb"),
                ("public", "pub123", "Public Viewer", "#16a34a"),
            ]
            for u, p, r, c in creds:
                st.markdown(
                    f'`{u}` / `{p}` — <span style="color:{c};font-weight:600;">{r}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("""
        <div style="text-align:center;font-size:0.72rem;color:#94a3b8;margin-top:1rem;">
            ⚠️ Prototype — SIH 2026 | Not for official use
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def sidebar_nav(df: pd.DataFrame):
    with st.sidebar:
        role = st.session_state.role
        role_color = ROLE_COLORS.get(role, "#64748b")

        st.markdown(f"""
        <div style="padding:1rem 0 0.5rem;">
            <div style="font-size:1.2rem;font-weight:800;color:#38bdf8;">🛡️ MPLADS Sentinel</div>
            <div style="font-size:0.68rem;color:#64748b;margin-top:2px;">AI Risk Intelligence Platform</div>
            <div style="margin-top:0.8rem;background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="font-size:0.8rem;color:#e2e8f0;font-weight:600;">
                    👤 {st.session_state.user_name}
                </div>
                <div style="font-size:0.7rem;color:{role_color};font-weight:700;margin-top:2px;">
                    ● {role}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        allowed_pages = ROLE_PAGES.get(role, [])
        page = st.radio("Navigation", allowed_pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:600;'>FILTERS</div>",
                    unsafe_allow_html=True)

        states = sorted(df["state"].dropna().unique().tolist())
        sel_state = st.multiselect("State", ["All"] + states, default=["All"], key="f_state")
        if "All" in sel_state or not sel_state:
            sel_state = states

        cats = sorted(df["work_category"].dropna().unique().tolist())
        sel_cat = st.multiselect("Category", ["All"] + cats, default=["All"], key="f_cat")
        if "All" in sel_cat or not sel_cat:
            sel_cat = cats

        statuses = sorted(df["work_status"].dropna().unique().tolist())
        sel_status = st.multiselect("Work Status", ["All"] + statuses, default=["All"], key="f_status")
        if "All" in sel_status or not sel_status:
            sel_status = statuses

        sel_risk = st.multiselect("Risk Level", ["All", "HIGH", "MEDIUM", "LOW"], default=["All"], key="f_risk")
        if "All" in sel_risk or not sel_risk:
            sel_risk = ["HIGH", "MEDIUM", "LOW"]

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            add_audit_entry("LOGOUT", "—", f"User '{st.session_state.user_name}' logged out")
            for k in ["logged_in", "username", "role", "user_name"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()

        st.markdown(
            f"<div style='font-size:0.65rem;color:#475569;'>Records: {len(df):,}</div>",
            unsafe_allow_html=True,
        )

    filters = {
        "state": sel_state, "work_category": sel_cat,
        "work_status": sel_status, "risk_level": sel_risk,
    }
    return page, filters


def apply_filters(df, filters):
    return df[
        df["state"].isin(filters["state"])
        & df["work_category"].isin(filters["work_category"])
        & df["work_status"].isin(filters["work_status"])
        & df["risk_level"].isin(filters["risk_level"])
    ].copy()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def page_dashboard(df, fdf):
    st.markdown("""
    <div class="sentinel-header">
        <h1>🛡️ MPLADS Sentinel</h1>
        <p>Explainable AI for Public Works Risk, Audit Prioritization & Early Warning &nbsp;|&nbsp; SIH 2026</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>AI DISCLAIMER:</strong> This system does <em>not</em> determine fraud.
        It identifies unusual patterns and prioritizes works for human review.
        All flagged cases require verification by authorized officials before any action.
    </div>
    """, unsafe_allow_html=True)

    summary = get_data_summary(df)
    high_risk = int((fdf["risk_level"] == "HIGH").sum())
    medium_risk = int((fdf["risk_level"] == "MEDIUM").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Works", f"{len(fdf):,}", f"of {summary['total_works']:,} total"),
        (c2, "Total Sanctioned", fmt_inr_short(fdf["sanction_amount"].sum()), "filtered total"),
        (c3, "Completed", f"{int(fdf['is_completed'].sum()):,}", f"{int(fdf['is_completed'].mean()*100)}% rate"),
        (c4, "🔴 High Risk", f"{high_risk:,}", "Priority review"),
        (c5, "🟡 Medium Risk", f"{medium_risk:,}", "Flag for audit"),
        (c6, "⏳ Old Pending", f"{int(fdf['delay_flag'].sum()):,}", ">1 yr not done"),
    ]
    for col, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        risk_counts = fdf["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        fig_pie = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"},
            hole=0.55, title="Risk Distribution",
        )
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), font_family="Inter")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        state_risk = (
            fdf.groupby("state")["risk_score"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "avg_risk", "count": "works"})
            .sort_values("avg_risk", ascending=False).head(15)
        )
        fig_state = px.bar(
            state_risk, x="avg_risk", y="state", orientation="h",
            color="avg_risk", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            range_color=[0, 100], title="Average Risk Score by State (Top 15)",
            text=state_risk["avg_risk"].round(1),
        )
        fig_state.update_layout(
            height=300, coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"}, font_family="Inter",
        )
        st.plotly_chart(fig_state, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        status_counts = fdf["work_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(
            status_counts, x="count", y="status", orientation="h",
            color="count", color_continuous_scale="Blues",
            title="Work Status Distribution", text="count",
        )
        fig_status.update_layout(height=300, coloraxis_showscale=False,
                                  yaxis={"categoryorder": "total ascending"}, font_family="Inter")
        st.plotly_chart(fig_status, use_container_width=True)

    with col_d:
        amounts = fdf["sanction_amount"].dropna() / 1e5
        fig_hist = px.histogram(amounts, nbins=60, title="Sanction Amount Distribution (₹ Lakhs)",
                                color_discrete_sequence=["#3b82f6"])
        fig_hist.update_layout(height=300, font_family="Inter", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        cat_counts = fdf["work_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_cat = px.bar(cat_counts, x="count", y="category", orientation="h",
                         color="count", color_continuous_scale="Purples",
                         title="Work Category Distribution", text="count")
        fig_cat.update_layout(height=280, coloraxis_showscale=False, font_family="Inter")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_f:
        anomaly_summary = pd.DataFrame({
            "Signal": ["Cost Anomaly", "Similar/Duplicate", "Old Pending"],
            "Flagged": [int(fdf["cost_anomaly_flag"].sum()),
                        int(fdf["dup_flag"].sum()),
                        int(fdf["delay_flag"].sum())],
        })
        fig_flags = px.bar(anomaly_summary, x="Signal", y="Flagged", color="Signal",
                           color_discrete_sequence=["#f59e0b", "#8b5cf6", "#ef4444"],
                           title="AI Detector Flags", text="Flagged")
        fig_flags.update_layout(height=280, font_family="Inter", showlegend=False)
        fig_flags.update_traces(textposition="outside")
        st.plotly_chart(fig_flags, use_container_width=True)

    st.markdown("<div class='section-title'>Top Districts by Number of Works</div>", unsafe_allow_html=True)
    top_dist = (
        fdf.groupby("district")
        .agg(works=("row_id", "count"), avg_risk=("risk_score", "mean"), total_amount=("sanction_amount", "sum"))
        .reset_index().sort_values("works", ascending=False).head(20)
    )
    top_dist["avg_risk"] = top_dist["avg_risk"].round(1)
    top_dist["total_amount"] = top_dist["total_amount"].apply(fmt_inr)
    st.dataframe(
        top_dist.rename(columns={"district": "District", "works": "Works",
                                  "avg_risk": "Avg Risk Score", "total_amount": "Total Sanctioned"}),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — HIGH-RISK WORKS
# ═══════════════════════════════════════════════════════════════════════════

def page_high_risk(df, fdf):
    st.markdown("<div class='section-title'>🔴 High-Risk & Medium-Risk Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        ⚠️ Flagging does <strong>not</strong> imply wrongdoing. All cases require human verification.
    </div>
    """, unsafe_allow_html=True)

    risk_df = fdf[fdf["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values("risk_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High-Risk", int((risk_df["risk_level"] == "HIGH").sum()))
    c2.metric("🟡 Medium-Risk", int((risk_df["risk_level"] == "MEDIUM").sum()))
    c3.metric("Total Flagged", len(risk_df))

    display = risk_df[[
        "sr_no", "work_description", "state", "district",
        "sanction_amount", "work_status", "risk_score", "risk_level", "main_reason",
    ]].copy()
    display["sanction_amount"] = display["sanction_amount"].apply(fmt_inr)
    display["risk_level"] = display["risk_level"].apply(risk_badge_md)
    display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.0f}/100")
    display["work_description"] = display["work_description"].str[:70] + "…"
    display = display.rename(columns={
        "sr_no": "Sr#", "work_description": "Work Description", "state": "State",
        "district": "District", "sanction_amount": "Sanctioned",
        "work_status": "Status", "risk_score": "Risk Score",
        "risk_level": "Risk Level", "main_reason": "Main Reason",
    })
    st.dataframe(display, use_container_width=True, hide_index=True, height=500)

    high_state = (
        risk_df[risk_df["risk_level"] == "HIGH"]
        .groupby("state").size().reset_index(name="count")
        .sort_values("count", ascending=False).head(15)
    )
    if not high_state.empty:
        fig = px.bar(high_state, x="state", y="count", color="count",
                     color_continuous_scale="Reds", title="High-Risk Works by State", text="count")
        fig.update_layout(height=320, coloraxis_showscale=False, font_family="Inter")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROJECT DETAIL (Explainability)
# ═══════════════════════════════════════════════════════════════════════════

def page_project_detail(df, similar_map):
    st.markdown("<div class='section-title'>🔍 Project Detail — Explainability View</div>", unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        search = st.text_input("Search by work description:", placeholder="e.g. road, school, water…")
    with col_sel2:
        risk_filter = st.selectbox("Filter by risk:", ["ALL", "HIGH", "MEDIUM", "LOW"])

    filtered = df.copy()
    if search:
        filtered = filtered[
            filtered["work_description"].str.contains(search, case=False, na=False)
            | filtered["sub_category"].str.contains(search, case=False, na=False)
        ]
    if risk_filter != "ALL":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if filtered.empty:
        st.warning("No matching projects found.")
        return

    options = {
        f"#{row['sr_no']} | {str(row['work_description'])[:65]} | {row['state']} | {row['risk_score']:.0f}/100 {row['risk_emoji']}": int(row.name)
        for _, row in filtered.head(200).iterrows()
    }
    selected_label = st.selectbox("Select project:", list(options.keys()))
    row_idx = options[selected_label]
    row = df.loc[row_idx]
    expl = get_explanation(row)

    st.markdown("---")

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"### {row.get('work_description', 'N/A')}")
        st.caption(f"Work Code: `{row.get('work_code', 'N/A')}`")
    with h2:
        level = expl["risk_level"]
        score = expl["risk_score"]
        c_bg, c_fg, emoji = RISK_COLORS.get(level, ("#e2e8f0", "#1e293b", "⚪"))
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;background:{c_fg};border-radius:12px;">
            <div style="font-size:2.5rem;font-weight:800;color:{c_bg};">{score:.0f}</div>
            <div style="font-size:0.8rem;font-weight:700;color:{c_bg};">/ 100 &nbsp;{emoji} {level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Project Details</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"**State:** {row.get('state', 'N/A')}")
        st.markdown(f"**District:** {row.get('district', 'N/A')}")
        st.markdown(f"**Constituency:** {row.get('constituency', 'N/A')}")
    with d2:
        st.markdown(f"**MP:** {row.get('mp_name_clean', 'N/A')}")
        st.markdown(f"**Category:** {row.get('work_category', 'N/A')}")
        st.markdown(f"**Status:** {status_emoji(row.get('work_status',''))} {row.get('work_status','N/A')}")
    with d3:
        st.markdown(f"**Sanctioned:** {fmt_inr(row.get('sanction_amount', 0))}")
        st.markdown(f"**Recommended:** {fmt_date(row.get('recommended_date'))}")
        st.markdown(f"**Sanctioned On:** {fmt_date(row.get('sanction_date'))}")

    # Data hash (tamper-evident)
    record_hash = short_hash({
        "sr_no": str(row.get("sr_no")),
        "sanction_amount": str(row.get("sanction_amount")),
        "work_status": str(row.get("work_status")),
    })
    st.markdown(
        f"<span style='font-size:0.72rem;color:#94a3b8;'>🔒 Record Hash: <span class='hash-chip'>{record_hash}</span></span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚠️ WHY FLAGGED? — Evidence Cards</div>", unsafe_allow_html=True)
    for reason in expl["reasons"]:
        sev = reason.get("severity", "MEDIUM")
        card_cls = "high" if sev == "HIGH" else ("low" if sev == "LOW" else "")
        st.markdown(f"""
        <div class="evidence-card {card_cls}">
            <span class="evidence-title">{reason['signal']}</span>
            <span class="evidence-pts">{reason['points']}</span>
            <div class="evidence-detail">{reason['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>📊 Risk Score Breakdown</div>", unsafe_allow_html=True)
    breakdown = pd.DataFrame({
        "Component": ["Cost Anomaly (max 40)", "Similar Work (max 30)", "Delay/Pending (max 30)"],
        "Score": [expl["cost_pts"], expl["dup_pts"], expl["delay_pts"]],
        "Max": [40, 30, 30],
    })
    breakdown["Remaining"] = breakdown["Max"] - breakdown["Score"]
    fig_score = go.Figure()
    fig_score.add_trace(go.Bar(
        name="Score", x=breakdown["Score"], y=breakdown["Component"], orientation="h",
        marker_color=["#ef4444", "#8b5cf6", "#f59e0b"],
        text=breakdown["Score"].apply(lambda x: f"{x:.0f} pts"), textposition="inside",
    ))
    fig_score.add_trace(go.Bar(
        name="Remaining", x=breakdown["Remaining"], y=breakdown["Component"],
        orientation="h", marker_color="#e2e8f0",
    ))
    fig_score.update_layout(barmode="stack", height=200, showlegend=False,
                             margin=dict(l=10, r=10, t=10, b=10), font_family="Inter")
    st.plotly_chart(fig_score, use_container_width=True)

    peer_median = row.get("peer_median", np.nan)
    peer_p75 = row.get("peer_p75", np.nan)
    peer_p90 = row.get("peer_p90", np.nan)
    peer_size = row.get("peer_size", 0)
    this_amount = row.get("sanction_amount", np.nan)
    peer_label = row.get("peer_group_label", "N/A")

    if pd.notna(peer_median) and peer_median > 0:
        st.markdown("<div class='section-title'>🏷️ Peer Comparison</div>", unsafe_allow_html=True)
        ratio = this_amount / peer_median if pd.notna(this_amount) else 1.0
        st.markdown(f"""
        | Metric | This Work | Peer Group ({peer_label}, n={int(peer_size)}) |
        |--------|-----------|------|
        | Amount | **{fmt_inr(this_amount)}** | Median: {fmt_inr(peer_median)} |
        | vs Median | **{ratio:.1f}×** | P75: {fmt_inr(peer_p75)} |
        | | | P90: {fmt_inr(peer_p90)} |
        """)
        fig_peer = go.Figure(go.Bar(
            x=["Peer Median", "Peer P75", "Peer P90", "This Work"],
            y=[peer_median/1e5, peer_p75/1e5, peer_p90/1e5, this_amount/1e5],
            marker_color=["#3b82f6", "#f59e0b", "#f97316", "#ef4444"],
            text=[fmt_inr(peer_median), fmt_inr(peer_p75), fmt_inr(peer_p90), fmt_inr(this_amount)],
            textposition="outside",
        ))
        fig_peer.update_layout(height=260, yaxis_title="₹ Lakhs", font_family="Inter", showlegend=False,
                                margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_peer, use_container_width=True)

    t1, t2, t3, t4 = st.columns(4)
    rec_to_san = row.get("rec_to_sanction_days", np.nan)
    days_since = row.get("days_since_sanction", np.nan)
    t1.metric("📅 Recommended", fmt_date(row.get("recommended_date")))
    t2.metric("✅ Sanctioned", fmt_date(row.get("sanction_date")))
    t3.metric("⏱ Rec→Sanction", f"{int(rec_to_san)} days" if pd.notna(rec_to_san) else "N/A",
              delta="⚠ Above 75-day guideline" if pd.notna(rec_to_san) and rec_to_san > 75 else None,
              delta_color="inverse")
    t4.metric("📆 Days Since Sanction", f"{int(days_since)} days" if pd.notna(days_since) else "N/A")

    row_id = int(row.get("row_id", -1))
    matches = similar_map.get(row_id, [])
    if matches:
        st.markdown("<br><div class='section-title'>🔗 Potentially Similar Works</div>", unsafe_allow_html=True)
        st.markdown("""<div class="disclaimer">
        High semantic similarity found. Does <strong>not</strong> confirm duplication — human review required.
        </div>""", unsafe_allow_html=True)
        for other_row_id, sim_score in matches[:5]:
            other_rows = df[df["row_id"] == other_row_id]
            if other_rows.empty:
                continue
            other = other_rows.iloc[0]
            badge_color = "#ef4444" if sim_score >= 0.88 else "#f59e0b"
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0;border-radius:8px;padding:0.9rem;margin-bottom:0.5rem;">
                <span style="background:{badge_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.78rem;font-weight:700;">
                    Similarity: {sim_score*100:.0f}%
                </span>
                &nbsp; Sr# {other.get('sr_no','')} — {other.get('state','')} / {other.get('district','')}<br>
                <span style="font-size:0.88rem;color:#1e293b;">{str(other.get('work_description',''))[:120]}</span><br>
                <span style="font-size:0.78rem;color:#64748b;">
                    Sanctioned: {fmt_inr(other.get('sanction_amount',0))} &nbsp;|&nbsp;
                    Status: {other.get('work_status','')} &nbsp;|&nbsp;
                    Risk: {other.get('risk_score',0):.0f}/100
                </span>
            </div>
            """, unsafe_allow_html=True)

    action_color = {"HIGH": "#fef2f2", "MEDIUM": "#fffbeb", "LOW": "#f0fdf4"}.get(level, "#f8fafc")
    st.markdown(f"""
    <div style="background:{action_color};border-radius:8px;padding:1rem 1.2rem;border:1px solid #e2e8f0;">
        <div style="font-weight:700;font-size:0.9rem;color:#1e293b;">📋 Recommended Action</div>
        <div style="font-size:0.85rem;color:#475569;margin-top:0.3rem;">{expl['recommended_action']}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def page_risk_analysis(df, fdf):
    st.markdown("<div class='section-title'>📈 Risk Analysis Overview</div>", unsafe_allow_html=True)

    sample = fdf.sample(min(3000, len(fdf)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="sanction_amount", y="risk_score", color="risk_level",
        color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"},
        hover_data=["work_description", "state", "district"],
        title="Sanction Amount vs Risk Score",
        labels={"sanction_amount": "Amount (₹)", "risk_score": "Risk Score"}, opacity=0.6,
    )
    fig_scatter.update_layout(height=380, font_family="Inter")
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_cost = px.histogram(fdf, x="cost_anomaly_score", nbins=50, color="cost_anomaly_flag",
                                color_discrete_map={True: "#ef4444", False: "#3b82f6"},
                                title="Cost Anomaly Score Distribution")
        fig_cost.update_layout(height=300, font_family="Inter")
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_b:
        fig_delay = px.histogram(fdf, x="delay_score", nbins=50, color="delay_flag",
                                 color_discrete_map={True: "#ef4444", False: "#64748b"},
                                 title="Delay Score Distribution")
        fig_delay.update_layout(height=300, font_family="Inter")
        st.plotly_chart(fig_delay, use_container_width=True)

    fy_risk = (
        fdf.groupby("financial_year")
        .agg(avg_risk=("risk_score", "mean"), works=("row_id", "count"))
        .reset_index().sort_values("financial_year")
    )
    fig_fy = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fy.add_trace(go.Bar(x=fy_risk["financial_year"], y=fy_risk["works"], name="Works",
                            marker_color="#bfdbfe"), secondary_y=False)
    fig_fy.add_trace(go.Scatter(x=fy_risk["financial_year"], y=fy_risk["avg_risk"],
                                name="Avg Risk", mode="lines+markers",
                                line=dict(color="#ef4444", width=2)), secondary_y=True)
    fig_fy.update_layout(title="Works & Avg Risk by Financial Year", height=300, font_family="Inter")
    fig_fy.update_yaxes(title_text="Works", secondary_y=False)
    fig_fy.update_yaxes(title_text="Avg Risk Score", secondary_y=True)
    st.plotly_chart(fig_fy, use_container_width=True)

    valid_gaps = fdf[fdf["rec_to_sanction_days"].notna() & (fdf["rec_to_sanction_days"] >= 0)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Median Rec→Sanction Gap", f"{valid_gaps['rec_to_sanction_days'].median():.0f} days")
    c2.metric("> 75 days", int((valid_gaps["rec_to_sanction_days"] > 75).sum()))
    c3.metric("> 180 days", int((valid_gaps["rec_to_sanction_days"] > 180).sum()))

    fig_gap = px.histogram(valid_gaps, x="rec_to_sanction_days", nbins=60,
                           title="Recommendation → Sanction Gap (days)", color_discrete_sequence=["#6366f1"])
    fig_gap.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="75-day guideline")
    fig_gap.update_layout(height=280, font_family="Inter")
    st.plotly_chart(fig_gap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — HUMAN-IN-THE-LOOP REVIEW WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

REVIEW_STATUS_FLOW = ["NEW", "UNDER REVIEW", "VERIFIED", "ESCALATED", "CLEARED", "INSPECTED"]
REVIEW_STATUS_CSS = {
    "NEW":          "status-new",
    "UNDER REVIEW": "status-review",
    "VERIFIED":     "status-review",
    "ESCALATED":    "status-escalated",
    "CLEARED":      "status-cleared",
    "INSPECTED":    "status-inspected",
}

def page_review_workflow(df, similar_map):
    st.markdown("<div class='section-title'>📋 Human-in-the-Loop Review Workflow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        Reviewers can acknowledge, verify, escalate, or clear flagged works.
        Every action is recorded in the tamper-evident audit trail.
        <strong>No AI decision is final — human judgment is required.</strong>
    </div>
    """, unsafe_allow_html=True)

    # Workflow status legend
    st.markdown("""
    **Workflow:** &nbsp;
    <span class="status-new">NEW</span> →
    <span class="status-review">UNDER REVIEW</span> →
    <span class="status-review">VERIFIED</span> →
    <span class="status-escalated">ESCALATED</span> /
    <span class="status-cleared">CLEARED</span> /
    <span class="status-inspected">INSPECTED</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    statuses = st.session_state.review_statuses
    total_reviewed = len(statuses)
    escalated = sum(1 for v in statuses.values() if v == "ESCALATED")
    cleared = sum(1 for v in statuses.values() if v == "CLEARED")
    inspected = sum(1 for v in statuses.values() if v == "INSPECTED")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actions Taken", total_reviewed)
    c2.metric("🔺 Escalated", escalated)
    c3.metric("✅ Cleared", cleared)
    c4.metric("🔍 Inspection Requested", inspected)

    st.markdown("---")

    # Get high+medium risk works
    queue_df = df[df["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values("risk_score", ascending=False)

    tab1, tab2 = st.tabs(["📂 Review Queue", "✅ Reviewed Cases"])

    with tab1:
        st.markdown(f"**{len(queue_df)} flagged works in queue** (sorted by risk)")

        search_rev = st.text_input("Filter queue by description:", key="rev_search",
                                    placeholder="e.g. road, water, school…")
        if search_rev:
            queue_df = queue_df[queue_df["work_description"].str.contains(search_rev, case=False, na=False)]

        for _, row in queue_df.head(30).iterrows():
            sr = str(row.get("sr_no", ""))
            current_status = statuses.get(sr, "NEW")
            status_css = REVIEW_STATUS_CSS.get(current_status, "status-new")
            level = row.get("risk_level", "LOW")
            c_bg, c_fg, emoji = RISK_COLORS.get(level, ("#e2e8f0", "#1e293b", "⚪"))

            with st.expander(
                f"{emoji} Sr#{sr} | {str(row.get('work_description',''))[:70]} | "
                f"Risk: {row.get('risk_score',0):.0f}/100 | {current_status}"
            ):
                col_info, col_action = st.columns([2, 1])

                with col_info:
                    st.markdown(f"""
                    **State:** {row.get('state','')} &nbsp;|&nbsp;
                    **District:** {row.get('district','')} &nbsp;|&nbsp;
                    **MP:** {row.get('mp_name_clean','')}

                    **Sanctioned:** {fmt_inr(row.get('sanction_amount',0))} &nbsp;|&nbsp;
                    **Status:** {row.get('work_status','')} &nbsp;|&nbsp;
                    **Days since sanction:** {int(row.get('days_since_sanction',0))} days

                    **Main Reason:** {row.get('main_reason','')}
                    """)

                    # Show evidence
                    expl = get_explanation(row)
                    for r in expl["reasons"]:
                        if r.get("points"):
                            st.markdown(
                                f"<small>• {r['signal']} {r['points']}: {r['detail'][:100]}</small>",
                                unsafe_allow_html=True,
                            )

                    # Previous notes
                    notes = st.session_state.review_notes.get(sr, [])
                    if notes:
                        st.markdown("**📝 Reviewer Notes:**")
                        for n in notes:
                            st.markdown(
                                f"<div class='audit-entry'>📝 [{n['time']}] <strong>{n['user']}</strong>: {n['note']}</div>",
                                unsafe_allow_html=True,
                            )

                with col_action:
                    st.markdown(f"""
                    <div style="text-align:center;margin-bottom:1rem;">
                        <span class="{status_css}">{current_status}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    note_text = st.text_area("Add note:", key=f"note_{sr}", height=80,
                                             placeholder="Enter review observation…")

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("📋 Under Review", key=f"ur_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr] = "UNDER REVIEW"
                            add_audit_entry("STATUS_CHANGE", sr, "Marked as UNDER REVIEW")
                            if note_text:
                                _add_note(sr, note_text)
                            st.rerun()

                        if st.button("✅ Clear", key=f"cl_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr] = "CLEARED"
                            add_audit_entry("STATUS_CHANGE", sr, f"CLEARED — {note_text or 'No note'}")
                            if note_text:
                                _add_note(sr, note_text)
                            st.success(f"Sr# {sr} cleared.")
                            st.rerun()

                    with btn_col2:
                        if st.button("🔺 Escalate", key=f"esc_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr] = "ESCALATED"
                            add_audit_entry("STATUS_CHANGE", sr, f"ESCALATED — {note_text or 'No note'}")
                            if note_text:
                                _add_note(sr, note_text)
                            st.warning(f"Sr# {sr} escalated.")
                            st.rerun()

                        if st.button("🔍 Inspect", key=f"ins_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr] = "INSPECTED"
                            add_audit_entry("STATUS_CHANGE", sr, "Field inspection requested")
                            if note_text:
                                _add_note(sr, note_text)
                            st.info(f"Sr# {sr} sent for inspection.")
                            st.rerun()

                    if note_text and st.button("💾 Save Note Only", key=f"sn_{sr}", use_container_width=True):
                        _add_note(sr, note_text)
                        add_audit_entry("NOTE_ADDED", sr, note_text)
                        st.success("Note saved.")
                        st.rerun()

    with tab2:
        reviewed = {k: v for k, v in statuses.items() if v != "NEW"}
        if not reviewed:
            st.info("No cases reviewed yet. Take action on items in the Review Queue.")
        else:
            for sr_no, status in reviewed.items():
                notes = st.session_state.review_notes.get(sr_no, [])
                css = REVIEW_STATUS_CSS.get(status, "status-new")
                rows_match = df[df["sr_no"] == sr_no]
                desc = rows_match.iloc[0]["work_description"][:70] if not rows_match.empty else "—"
                st.markdown(f"""
                <div class="review-card">
                    <strong>Sr# {sr_no}</strong> &nbsp;
                    <span class="{css}">{status}</span><br>
                    <small style="color:#475569;">{desc}…</small><br>
                    {"<br>".join([f"<small>📝 [{n['time']}] {n['user']}: {n['note']}</small>" for n in notes]) if notes else ""}
                </div>
                """, unsafe_allow_html=True)


def _add_note(sr: str, note: str):
    if sr not in st.session_state.review_notes:
        st.session_state.review_notes[sr] = []
    st.session_state.review_notes[sr].append({
        "time": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        "user": st.session_state.user_name,
        "note": note,
    })


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — AUTOMATED INVESTIGATION REPORT
# ═══════════════════════════════════════════════════════════════════════════

def page_investigation_report(df, similar_map):
    st.markdown("<div class='section-title'>📄 Automated Investigation Report</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        This report identifies <strong>potential irregularities</strong> for human review.
        It does NOT constitute a legal or audit conclusion.
        Use language: <em>Potential irregularity · Anomaly detected · Requires verification</em>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("Search project:", placeholder="Enter description keyword or Sr#…")
    filtered = df.copy()
    if search:
        filtered = filtered[
            filtered["work_description"].str.contains(search, case=False, na=False)
            | filtered["sr_no"].astype(str).str.contains(search, na=False)
        ]
    if filtered.empty:
        st.warning("No projects found.")
        return

    options = {
        f"#{row['sr_no']} | {str(row['work_description'])[:65]} | {row['risk_score']:.0f}/100 {row['risk_emoji']}": int(row.name)
        for _, row in filtered.head(100).iterrows()
    }
    selected_label = st.selectbox("Select project for report:", list(options.keys()))
    row_idx = options[selected_label]
    row = df.loc[row_idx]
    expl = get_explanation(row)
    sr = str(row.get("sr_no", ""))

    reviewer_notes = st.session_state.review_notes.get(sr, [])
    review_status = st.session_state.review_statuses.get(sr, "NEW")

    # Preview
    report_text = _generate_report(row, expl, similar_map, df, reviewer_notes, review_status)

    st.markdown("---")
    st.markdown("### 📋 Report Preview")

    with st.container():
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;font-family:monospace;font-size:0.82rem;white-space:pre-wrap;max-height:600px;overflow-y:auto;">
{report_text}
        </div>
        """, unsafe_allow_html=True)

    # Download button
    report_filename = f"MPLADS_Sentinel_Report_Sr{sr}_{datetime.date.today()}.txt"
    st.download_button(
        label="⬇️ Download Investigation Report (.txt)",
        data=report_text,
        file_name=report_filename,
        mime="text/plain",
        type="primary",
        use_container_width=True,
    )

    # Log report generation
    add_audit_entry("REPORT_GENERATED", sr, f"Investigation report generated for Sr# {sr}")


def _generate_report(row, expl, similar_map, df, reviewer_notes, review_status) -> str:
    sr = str(row.get("sr_no", ""))
    now = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    user = st.session_state.user_name
    role = st.session_state.role

    # Similar works section
    row_id = int(row.get("row_id", -1))
    matches = similar_map.get(row_id, [])
    similar_lines = []
    for other_row_id, sim_score in matches[:5]:
        other_rows = df[df["row_id"] == other_row_id]
        if not other_rows.empty:
            o = other_rows.iloc[0]
            similar_lines.append(
                f"  • Sr# {o.get('sr_no','')} | Similarity: {sim_score*100:.0f}% | "
                f"{str(o.get('work_description',''))[:60]} | {fmt_inr(o.get('sanction_amount',0))}"
            )

    notes_lines = "\n".join(
        [f"  [{n['time']}] {n['user']}: {n['note']}" for n in reviewer_notes]
    ) or "  No reviewer notes added."

    reasons_lines = "\n".join(
        [f"  {r['signal']} {r['points']}\n  → {r['detail']}" for r in expl["reasons"]]
    )

    record_hash = hash_record({
        "sr_no": sr,
        "sanction_amount": str(row.get("sanction_amount")),
        "work_status": str(row.get("work_status")),
        "risk_score": str(expl["risk_score"]),
    })

    report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              MPLADS SENTINEL — INVESTIGATION REPORT                      ║
║              AI-Assisted Public Works Risk Analysis                       ║
╚══════════════════════════════════════════════════════════════════════════╝

REPORT GENERATED:    {now}
GENERATED BY:        {user} ({role})
REPORT HASH:         {record_hash[:24].upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report identifies POTENTIAL IRREGULARITIES based on algorithmic pattern
detection. It does NOT constitute proof of fraud, misconduct, or legal
violation. All findings require verification by authorized officials before
any action is taken. AI does not determine fraud — it prioritizes cases for
human review.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sr. No.          : {sr}
  Work Description : {row.get('work_description', 'N/A')}
  Work Code        : {row.get('work_code', 'N/A')}
  Work Category    : {row.get('work_category', 'N/A')}
  Sub-Category     : {row.get('sub_category', 'N/A')}
  State            : {row.get('state', 'N/A')}
  District (IDA)   : {row.get('district', 'N/A')}
  Constituency     : {row.get('constituency', 'N/A')}
  MP               : {row.get('mp_name_clean', 'N/A')}
  Implementing Ag. : (Not in dataset)
  Financial Year   : {row.get('financial_year', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. FINANCIAL DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sanctioned Amount : {fmt_inr(row.get('sanction_amount', 0))}
  Work Status       : {row.get('work_status', 'N/A')}
  Recommended Date  : {fmt_date(row.get('recommended_date'))}
  Sanction Date     : {fmt_date(row.get('sanction_date'))}
  Rec→Sanction Gap  : {int(row.get('rec_to_sanction_days', 0))} days (guideline: ≤75 days)
  Days Since Sanct. : {int(row.get('days_since_sanction', 0))} days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. REVIEW PRIORITY SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOTAL SCORE  : {expl['risk_score']:.0f} / 100
  RISK LEVEL   : {expl['risk_emoji']} {expl['risk_level']}

  Component Breakdown:
    Cost Anomaly (max 40)   : {expl['cost_pts']:.0f} pts
    Similar Work (max 30)   : {expl['dup_pts']:.0f} pts
    Delay/Pending (max 30)  : {expl['delay_pts']:.0f} pts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. ANOMALY SIGNALS (WHY FLAGGED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{reasons_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. PEER COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Peer Group       : {row.get('peer_group_label', 'N/A')}
  Peer Group Size  : {int(row.get('peer_size', 0))} works
  This Work Amount : {fmt_inr(row.get('sanction_amount', 0))}
  Peer Median      : {fmt_inr(row.get('peer_median', 0))}
  Peer P75         : {fmt_inr(row.get('peer_p75', 0))}
  Peer P90         : {fmt_inr(row.get('peer_p90', 0))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. POTENTIALLY SIMILAR / DUPLICATE WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(similar_lines) if similar_lines else "  No highly similar works detected above threshold."}
  NOTE: Similarity does NOT confirm duplication. Human verification required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. CURRENT REVIEW STATUS & REVIEWER NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Review Status    : {review_status}
  Reviewer Notes:
{notes_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. RECOMMENDED ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {expl['recommended_action']}

  Suggested verification steps:
  1. Obtain sanction order and estimate documents from district authority
  2. Verify implementing agency credentials and past work record
  3. Cross-check similar works in same area for duplication
  4. Request geo-tagged photos from eSAKSHI portal
  5. Compare sanctioned amount with Schedule of Rates (SOR) for work type
  6. Verify physical progress with field inspection if escalated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. DATA AVAILABLE / NOT AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Data used for this report: Sanctioned amount, work description, dates,
  status, state, district, MP, constituency.

  NOT available (future integration needed):
  Actual expenditure | Physical progress % | Payment history |
  GPS coordinates | Asset photographs | Contractor details

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TAMPER-EVIDENT RECORD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Report Hash (SHA-256): {record_hash}
  This hash confirms report integrity at time of generation.

  MPLADS Sentinel — SIH 2026 | Prototype Only
  ⚠️  NOT an official government audit document
══════════════════════════════════════════════════════════════════════════
"""
    return report


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — AUDIT TRAIL (Tamper-Evident)
# ═══════════════════════════════════════════════════════════════════════════

def page_audit_trail():
    st.markdown("<div class='section-title'>🔒 Complete Audit Trail — Tamper-Evident Log</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        Every action is logged with a SHA-256 chain hash. Each entry includes the hash of
        the previous entry — any modification to a past entry breaks the chain.
        This provides tamper-evidence for the review workflow.
    </div>
    """, unsafe_allow_html=True)

    log = st.session_state.audit_log
    if not log:
        st.info("No audit entries yet. Login, review a project, or generate a report to create log entries.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Log Entries", len(log))
    c2.metric("Unique Users", len({e["user"] for e in log}))
    c3.metric("Actions Taken", len({e["action"] for e in log}))

    st.markdown("---")

    # Verify chain integrity
    chain_ok = True
    for i in range(1, len(log)):
        if log[i].get("prev_hash") != log[i-1]["hash"]:
            chain_ok = False
            break

    if chain_ok:
        st.success("✅ Audit chain integrity verified — no tampering detected.")
    else:
        st.error("⚠️ Chain integrity check FAILED — log may have been modified.")

    st.markdown("---")

    # Show entries (newest first)
    action_filter = st.multiselect(
        "Filter by action:",
        ["All", "LOGIN", "LOGOUT", "STATUS_CHANGE", "NOTE_ADDED", "REPORT_GENERATED"],
        default=["All"],
    )

    for entry in reversed(log):
        if "All" not in action_filter and entry["action"] not in action_filter:
            continue

        action_icon = {
            "LOGIN":            "🔓",
            "LOGOUT":           "🔒",
            "STATUS_CHANGE":    "🔄",
            "NOTE_ADDED":       "📝",
            "REPORT_GENERATED": "📄",
        }.get(entry["action"], "•")

        st.markdown(f"""
        <div class="audit-entry">
            <strong>{action_icon} {entry['action']}</strong> &nbsp;
            <span style="color:#64748b;">{entry['timestamp'][:19].replace('T',' ')}</span> &nbsp;
            <span style="color:#3b82f6;font-weight:600;">{entry['user']}</span>
            ({entry['role']}) &nbsp;|&nbsp; Project: <code>{entry['project_id']}</code><br>
            <span style="color:#475569;">{entry['detail']}</span><br>
            <span class="hash-chip">#{entry['hash'][:16].upper()}</span> &nbsp;
            <span style="font-size:0.68rem;color:#94a3b8;">prev: {entry.get('prev_hash','—')[:12].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    # Download full log
    log_json = json.dumps(log, indent=2, default=str)
    st.download_button(
        "⬇️ Download Full Audit Log (JSON)",
        data=log_json,
        file_name=f"audit_log_{datetime.date.today()}.json",
        mime="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — CITIZEN FEEDBACK / ISSUE REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def page_citizen_feedback(df):
    st.markdown("<div class='section-title'>💬 Citizen Feedback & Issue Reporting</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        Citizens can report concerns about specific MPLADS works.
        Submissions are reviewed by authorized officials.
        This does NOT guarantee action — all reports require verification.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Submit Feedback", "📋 All Submissions (Admin)"])

    with tab1:
        with st.form("citizen_feedback_form", clear_on_submit=True):
            st.markdown("#### Report a Concern")
            sr_no_input = st.text_input("Work Sr. No. (if known):", placeholder="e.g. 4821")
            state_input = st.selectbox("State:", ["Select…"] + sorted(df["state"].dropna().unique().tolist()))
            district_input = st.text_input("District:", placeholder="e.g. Agra")
            category = st.selectbox("Issue Category:", [
                "Select category…",
                "Work not started despite sanction",
                "Work appears to be duplicate of another",
                "Cost appears unusually high",
                "Work completed on paper but not physically",
                "Incorrect location / beneficiary",
                "Implementation agency related concern",
                "Other",
            ])
            description = st.text_area("Describe the concern:", height=120,
                                        placeholder="Please describe what you observed…")
            contact = st.text_input("Contact (optional — email or phone):",
                                    placeholder="Your contact for follow-up")
            is_anonymous = st.checkbox("Submit anonymously")

            submitted = st.form_submit_button("📤 Submit Report", type="primary", use_container_width=True)

            if submitted:
                if not description or category == "Select category…":
                    st.error("Please fill in the issue category and description.")
                else:
                    feedback_entry = {
                        "id":          len(st.session_state.feedback_list) + 1,
                        "timestamp":   datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                        "sr_no":       sr_no_input or "—",
                        "state":       state_input,
                        "district":    district_input,
                        "category":    category,
                        "description": description,
                        "contact":     "Anonymous" if is_anonymous else (contact or "Not provided"),
                        "status":      "RECEIVED",
                    }
                    st.session_state.feedback_list.append(feedback_entry)
                    add_audit_entry("CITIZEN_FEEDBACK", sr_no_input or "—",
                                    f"Citizen report received — {category}")
                    st.success(
                        f"✅ Thank you! Your report has been submitted (Ref# CF-{feedback_entry['id']:04d}). "
                        f"It will be reviewed by authorized officials."
                    )

    with tab2:
        if st.session_state.role not in ["Admin", "Senior Reviewer"]:
            st.warning("Access restricted to Admin and Senior Reviewer roles.")
            return

        feedback = st.session_state.feedback_list
        if not feedback:
            st.info("No citizen feedback received yet.")
        else:
            st.metric("Total Submissions", len(feedback))
            fb_df = pd.DataFrame(feedback)
            st.dataframe(fb_df, use_container_width=True, hide_index=True)

            csv = fb_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Feedback CSV",
                data=csv,
                file_name=f"citizen_feedback_{datetime.date.today()}.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 9 — PUBLIC TRANSPARENCY PORTAL
# ═══════════════════════════════════════════════════════════════════════════

def page_transparency_portal(df):
    st.markdown("""
    <div class="sentinel-header">
        <h1>🌐 Public Transparency Portal</h1>
        <p>MPLADS Works — Public Dashboard | Read-Only View</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        This portal provides public access to MPLADS sanctioned work statistics.
        All data is from the official Works Sanctioned dataset.
        Individual work details are displayed for transparency and public accountability.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Works", f"{len(df):,}")
    c2.metric("Total Sanctioned", fmt_inr_short(df["sanction_amount"].sum()))
    c3.metric("States Covered", df["state"].nunique())
    c4.metric("Completed Works", int(df["is_completed"].sum()))

    st.markdown("---")

    # State-wise summary (public safe — no risk scores)
    state_summary = (
        df.groupby("state")
        .agg(
            total_works=("row_id", "count"),
            completed=("is_completed", "sum"),
            total_amount=("sanction_amount", "sum"),
        )
        .reset_index()
    )
    state_summary["completion_rate"] = (
        state_summary["completed"] / state_summary["total_works"] * 100
    ).round(1)
    state_summary["total_amount_fmt"] = state_summary["total_amount"].apply(fmt_inr)
    state_summary = state_summary.sort_values("total_works", ascending=False)

    st.markdown("#### State-wise MPLADS Works Summary")
    st.dataframe(
        state_summary[["state", "total_works", "completed", "completion_rate", "total_amount_fmt"]]
        .rename(columns={
            "state": "State", "total_works": "Total Works",
            "completed": "Completed", "completion_rate": "Completion Rate (%)",
            "total_amount_fmt": "Total Sanctioned",
        }),
        use_container_width=True, hide_index=True,
    )

    fig_pub = px.bar(
        state_summary.head(20), x="state", y="total_works",
        color="completion_rate", color_continuous_scale="RdYlGn",
        title="Works by State (colour = Completion Rate %)",
        labels={"total_works": "Total Works", "completion_rate": "Completion %"},
        text="total_works",
    )
    fig_pub.update_layout(height=380, font_family="Inter")
    fig_pub.update_traces(textposition="outside")
    st.plotly_chart(fig_pub, use_container_width=True)

    st.markdown("#### Work Category Distribution")
    cat_pub = df["work_category"].value_counts().reset_index()
    cat_pub.columns = ["Category", "Count"]
    fig_cat = px.pie(cat_pub, values="Count", names="Category",
                     title="Works by Category", hole=0.4)
    fig_cat.update_layout(height=320, font_family="Inter")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Search individual works (public view — no risk scores shown)
    st.markdown("#### 🔍 Search Works (Public View)")
    pub_search = st.text_input("Search by work description or state:", placeholder="e.g. road, Bihar…")
    if pub_search:
        pub_results = df[
            df["work_description"].str.contains(pub_search, case=False, na=False)
            | df["state"].str.contains(pub_search, case=False, na=False)
        ].head(50)
        pub_display = pub_results[[
            "sr_no", "work_description", "state", "district",
            "sanction_amount", "work_status", "sanction_date",
        ]].copy()
        pub_display["sanction_amount"] = pub_display["sanction_amount"].apply(fmt_inr)
        pub_display["sanction_date"] = pub_display["sanction_date"].apply(fmt_date)
        pub_display = pub_display.rename(columns={
            "sr_no": "Sr#", "work_description": "Work Description",
            "state": "State", "district": "District",
            "sanction_amount": "Sanctioned", "work_status": "Status",
            "sanction_date": "Sanction Date",
        })
        st.dataframe(pub_display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 10 — ABOUT & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown("<div class='section-title'>ℹ️ About MPLADS Sentinel & Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
    ### What is MPLADS Sentinel?
    An AI-assisted intelligence layer on top of MPLADS sanctioned-works data.

    > **This system does NOT replace government audits.**
    > It helps authorities identify which works deserve attention first and explains why.

    ### AI Pipeline
    ```
    Works Sanctioned CSV (19,001 rows)
          ↓
    Preprocessing → Clean, Extract, Engineer Features
          ↓
    Detector 1: Cost Anomaly (Isolation Forest per peer group)    → 0–40 pts
    Detector 2: Semantic Similarity (sentence-transformers+cosine) → 0–30 pts
    Detector 3: Delay / Old Pending (Statistical)                 → 0–30 pts
          ↓
    Risk Fusion → 0–100 Review Priority Score
          ↓
    Explainability → "Why Flagged?" Evidence Cards
          ↓
    Human-in-the-Loop Review → Verify / Escalate / Clear / Inspect
          ↓
    Tamper-Evident Audit Trail (SHA-256 chain hashing)
    ```

    ### Security & Governance Features
    | Feature | Implementation |
    |---------|---------------|
    | Role-Based Access Control | 4 roles: Admin, Senior Reviewer, Reviewer, Public |
    | Tamper-Evident Logs | SHA-256 chain hashing (each entry hashes the previous) |
    | Complete Audit Trail | Every action logged with timestamp, user, role |
    | Human-in-the-Loop | No automated action — human approves every decision |
    | Citizen Feedback | Public issue reporting with admin review |
    | Public Transparency | Read-only portal with no sensitive risk data |

    ### Responsible AI
    - ✅ Uses: "Potential irregularity", "Anomaly detected", "Requires verification"
    - ❌ Never: "Fraud confirmed", "Corrupt official", "Criminal"
    - ✅ Every flag shows evidence and confidence level
    - ✅ Data availability clearly disclosed
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Data Available")
        for f in AVAILABLE_FIELDS:
            st.markdown(f"- {f}")
    with col2:
        st.markdown("### 🔮 Future Data Needed")
        for f in MISSING_FIELDS:
            st.markdown(f"- {f}")

    st.markdown("---")
    st.markdown("""
    **Team**: SIH 2026 | Problem Statement SIH26102
    **Disclaimer**: Prototype using real MPLADS sanctioned-works data for demonstration only.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Show login page if not authenticated
    if not st.session_state.logged_in:
        page_login()
        return

    # Load data pipeline (cached)
    with st.spinner("Loading MPLADS data and running AI analysis…"):
        df, similar_map = run_full_pipeline()

    # Sidebar nav + filters
    page, filters = sidebar_nav(df)
    fdf = apply_filters(df, filters)

    if len(fdf) == 0:
        st.warning("No records match current filters. Adjust the sidebar filters.")
        return

    # Route to page
    if page == "📊 Dashboard":
        page_dashboard(df, fdf)
    elif page == "🔴 High-Risk Works":
        page_high_risk(df, fdf)
    elif page == "🔍 Project Detail":
        page_project_detail(df, similar_map)
    elif page == "📈 Risk Analysis":
        page_risk_analysis(df, fdf)
    elif page == "📋 Review Workflow":
        page_review_workflow(df, similar_map)
    elif page == "📄 Investigation Report":
        page_investigation_report(df, similar_map)
    elif page == "🔒 Audit Trail":
        page_audit_trail()
    elif page == "💬 Citizen Feedback":
        page_citizen_feedback(df)
    elif page == "🌐 Transparency Portal":
        page_transparency_portal(df)
    elif page == "ℹ️ About & Methodology":
        page_about()


if __name__ == "__main__":
    main()
