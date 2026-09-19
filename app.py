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
from ml.rule_signals import detect_rule_violations
from ml.risk_score import compute_risk_scores, get_explanation
from utils.helpers import (
    fmt_inr, fmt_inr_short, fmt_date,
    risk_badge_html, risk_badge_md, status_emoji,
    AVAILABLE_FIELDS, MISSING_FIELDS, FUTURE_IMPROVEMENTS,
    RISK_COLORS,
)
from db.database import (
    init_db, authenticate_user, register_user, add_audit_entry as db_add_audit,
    get_audit_logs, verify_chain, tamper_demo_record,
    add_evidence, get_evidence_for_work, get_all_evidence_counts,
    hash_record
)

init_db()

st.set_page_config(page_title="MPLADS Sentinel", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

ROLE_PAGES = {
    "MoSPI Admin": ["📊 Dashboard", "⚙️ System Tuning & Audits", "🔴 High-Risk Works", "🔍 Project Detail", "📈 Risk Analysis", "📋 Review Workflow", "📄 Investigation Report", "🔒 Audit Trail", "💬 Citizen Feedback", "🌐 Transparency Portal", "ℹ️ About & Methodology"],
    "District Nodal Officer": ["📊 Dashboard", "🔴 High-Risk Works", "🔍 Project Detail", "📋 Review Workflow", "📄 Investigation Report", "💬 Citizen Feedback", "ℹ️ About & Methodology"],
    "Member of Parliament": ["🏛️ My Constituency", "💬 Citizen Feedback", "ℹ️ About & Methodology"],
    "Public Viewer": ["🌐 Transparency Portal", "💬 Citizen Feedback", "ℹ️ About & Methodology"],
}

SAFE_COLUMNS = ["sr_no", "row_id", "work_code", "work_description", "work_category", "state", "district", "financial_year", "sanction_date", "sanction_amount", "work_status", "days_since_sanction", "recommended_date", "constituency", "mp_name_clean", "amount_lakhs", "is_completed", "is_incomplete"]

ROLE_COLORS = {"MoSPI Admin":"#dc2626","District Nodal Officer":"#d97706","Member of Parliament":"#8b5cf6","Public Viewer":"#16a34a"}

def init_session_state():
    defaults = {
        "logged_in": False, "username": "", "role": "", "user_name": "",
        "district": "", "constituency": "",
        "review_actions": [], "feedback_list": [],
        "review_statuses": {}, "review_notes": {},
        # Admin Tuning Weights
        "w_cost": 30, "w_dup": 25, "w_del": 20, "w_pat": 25, "min_amt": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

def add_audit_entry(action, project_id, detail, extra=None):
    # Delegate to SQLite Database
    user = st.session_state.get("user_name", "Unknown")
    role = st.session_state.get("role", "Unknown")
    db_add_audit(user, role, action, project_id, detail, extra)

def short_hash(data):
    s = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

/* Main Header */
.sentinel-header{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#1e40af 100%);padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.5rem;color:white;}
.sentinel-header h1{font-size:1.8rem;font-weight:700;margin:0;color:#ffffff !important;}
.sentinel-header p{font-size:0.85rem;margin:0.2rem 0 0;opacity:0.85;color:#e2e8f0 !important;}

/* Section Titles — VIBRANT & 100% VISIBLE IN DARK & LIGHT THEMES */
.section-title{
    font-size:1.35rem !important;
    font-weight:700 !important;
    color: #38bdf8 !important;
    border-bottom:2px solid #0284c7 !important;
    padding-bottom:0.45rem !important;
    margin-top:1rem !important;
    margin-bottom:1.25rem !important;
    letter-spacing:-0.01em;
    display:flex;
    align-items:center;
}

/* Ensure all subheadings are clearly visible */
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: var(--text-color, #f8fafc) !important;
}

/* KPI Cards — High contrast dark navy */
.kpi-card{
    background:#0f172a !important;
    border-radius:12px;
    padding:1.2rem 1.4rem;
    border:1px solid #334155;
    box-shadow:0 4px 6px -1px rgba(0,0,0,0.25);
    height:100%;
}
.kpi-label{font-size:0.75rem;font-weight:600;color:#94a3b8 !important;text-transform:uppercase;letter-spacing:0.5px;}
.kpi-value{font-size:2rem;font-weight:700;color:#ffffff !important;line-height:1.2;margin:0.3rem 0;}
.kpi-sub{font-size:0.78rem;color:#64748b !important;}

/* Evidence Cards */
.evidence-card{
    background:rgba(245, 158, 11, 0.12) !important;
    border-left:4px solid #f59e0b !important;
    padding:0.9rem 1.1rem;
    border-radius:0 8px 8px 0;
    margin-bottom:0.75rem;
}
.evidence-card.high{background:rgba(239, 68, 68, 0.14) !important;border-color:#ef4444 !important;}
.evidence-card.low{background:rgba(34, 197, 94, 0.12) !important;border-color:#22c55e !important;}
.evidence-title{font-weight:700;font-size:0.95rem;color:#ffffff !important;}
.evidence-detail{font-size:0.84rem;color:#cbd5e1 !important;margin-top:0.25rem;}
.evidence-pts{font-size:0.82rem;font-weight:700;color:#f59e0b;float:right;}
.evidence-card.high .evidence-pts{color:#ef4444;}
.evidence-card.low .evidence-pts{color:#22c55e;}

/* Disclaimers */
.disclaimer{
    background:rgba(59, 130, 246, 0.1) !important;
    border:1px solid rgba(59, 130, 246, 0.35) !important;
    border-radius:8px;
    padding:0.7rem 1.1rem;
    font-size:0.82rem;
    color:#e2e8f0 !important;
    margin-bottom:1.2rem;
}

/* Status Chips */
.status-new{background:#1e3a8a;color:#93c5fd;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;}
.status-review{background:#78350f;color:#fde68a;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;}
.status-escalated{background:#7f1d1d;color:#fca5a5;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;}
.status-cleared{background:#14532d;color:#86efac;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;}
.status-inspected{background:#4c1d95;color:#d8b4fe;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;}

section[data-testid="stSidebar"]{background:#0f172a;}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] div {color:#e2e8f0;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Running Core AI Models…", ttl=3600)
def run_core_models():
    df = load_data()
    df = detect_cost_anomalies(df)
    df, similar_map = detect_duplicates(df)
    df = detect_delays(df)
    df = detect_rule_violations(df)
    return df, similar_map


def page_login():
    st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3rem;">🛡️</div>
        <div style="font-size:1.8rem;font-weight:800;color:#38bdf8;">MPLADS Sentinel</div>
        <div style="font-size:0.88rem;color:#94a3b8;margin-top:4px;">Explainable AI for Public Works Risk & Audit Prioritization</div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        tab_in, tab_reg = st.tabs(["🔐 Sign In", "📝 Register"])
        
        with tab_in:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            if st.button("Sign In", use_container_width=True, type="primary"):
                user = authenticate_user(username.strip().lower(), password.strip())
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]
                    st.session_state.user_name = user["name"]
                    st.session_state.district = user.get("district", "")
                    st.session_state.constituency = user.get("constituency", "")
                    add_audit_entry("LOGIN", "—", f"User '{user['name']}' logged in as {user['role']}")
                    st.success(f"Welcome, {user['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.markdown("---")
            st.markdown("**Demo Credentials:** `admin`/`admin123`, `nodal`/`nodal123`, `mp`/`mp123`, `public`/`pub123`")

        with tab_reg:
            reg_user = st.text_input("New Username")
            reg_pass = st.text_input("New Password", type="password")
            reg_name = st.text_input("Full Name")
            reg_role = st.selectbox("Role", ["MoSPI Admin", "District Nodal Officer", "Member of Parliament", "Public Viewer"])
            reg_dist = st.text_input("District (if applicable)")
            reg_const = st.text_input("Constituency (if applicable)")
            
            if st.button("Register", use_container_width=True):
                if register_user(reg_user.strip().lower(), reg_pass.strip(), reg_role, reg_name.strip(), reg_dist.strip(), reg_const.strip()):
                    st.success("Registered successfully! Please Sign In.")
                else:
                    st.error("Username already exists or error occurred.")

    st.markdown('<div style="text-align:center;font-size:0.72rem;color:#94a3b8;margin-top:1rem;">⚠️ Prototype — SIH 2026 | Not for official use</div>', unsafe_allow_html=True)

def sidebar_nav(df):
    with st.sidebar:
        role = st.session_state.role
        role_color = ROLE_COLORS.get(role, "#64748b")
        st.markdown(f"""<div style="padding:1rem 0 0.5rem;">
            <div style="font-size:1.2rem;font-weight:800;color:#38bdf8;">🛡️ MPLADS Sentinel</div>
            <div style="font-size:0.68rem;color:#64748b;margin-top:2px;">AI Risk Intelligence Platform</div>
            <div style="margin-top:0.8rem;background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;">
                <div style="font-size:0.8rem;color:#e2e8f0;font-weight:600;">👤 {st.session_state.user_name}</div>
                <div style="font-size:0.7rem;color:{role_color};font-weight:700;margin-top:2px;">● {role}</div>
            </div></div>""", unsafe_allow_html=True)
        
        # Apply strict data filtering based on role
        filtered_df = df.copy()
        if role == "District Nodal Officer":
            available_districts = sorted(df["district"].dropna().unique().tolist())
            user_district = st.session_state.get("district")
            if not user_district or user_district not in available_districts:
                user_district = "Ajmer" if "Ajmer" in available_districts else available_districts[0]
            default_idx = available_districts.index(user_district)
            sel_district = st.selectbox("📍 Assigned District:", available_districts, index=default_idx)
            st.session_state.district = sel_district
            filtered_df = filtered_df[filtered_df["district"] == sel_district].copy()
            st.caption(f"Jurisdiction: **{sel_district}** ({len(filtered_df):,} works)")
        elif role == "Member of Parliament":
            available_constituencies = sorted(df["constituency"].dropna().unique().tolist())
            user_const = st.session_state.get("constituency")
            if not user_const or user_const not in available_constituencies:
                user_const = "Agra" if "Agra" in available_constituencies else available_constituencies[0]
            default_idx = available_constituencies.index(user_const)
            sel_const = st.selectbox("🏛️ My Constituency:", available_constituencies, index=default_idx)
            st.session_state.constituency = sel_const
            filtered_df = filtered_df[filtered_df["constituency"] == sel_const].copy()
            st.caption(f"Constituency: **{sel_const}** ({len(filtered_df):,} works)")
            
        allowed_pages = ROLE_PAGES.get(role, [])
        page = st.radio("MAIN MENU", allowed_pages)
        
        st.markdown("---")
        # Only show risk filters to administrative roles
        if role in ["MoSPI Admin", "District Nodal Officer"]:
            st.markdown("<div style='font-size:0.75rem;color:#94a3b8;font-weight:700;'>FILTERS</div>", unsafe_allow_html=True)
            states = sorted(filtered_df["state"].dropna().unique().tolist())
            sel_state = st.multiselect("State", ["All"] + states, default=["All"], key="f_state")
            if "All" in sel_state or not sel_state: sel_state = states
            
            cats = sorted(filtered_df["work_category"].dropna().unique().tolist()) if "work_category" in filtered_df.columns else []
            sel_cat = st.multiselect("Category", ["All"] + cats, default=["All"], key="f_cat")
            if "All" in sel_cat or not sel_cat: sel_cat = cats
            
            statuses = sorted(filtered_df["work_status"].dropna().unique().tolist()) if "work_status" in filtered_df.columns else []
            sel_status = st.multiselect("Work Status", ["All"] + statuses, default=["All"], key="f_status")
            if "All" in sel_status or not sel_status: sel_status = statuses
            
            sel_risk = st.multiselect("Risk Level", ["All", "HIGH", "MEDIUM", "LOW"], default=["All"], key="f_risk")
            if "All" in sel_risk or not sel_risk: sel_risk = ["HIGH", "MEDIUM", "LOW"]
        else:
            # MP & Public Viewer do not see or use risk level filters
            sel_state = sorted(filtered_df["state"].dropna().unique().tolist())
            sel_cat = sorted(filtered_df["work_category"].dropna().unique().tolist()) if "work_category" in filtered_df.columns else []
            sel_status = sorted(filtered_df["work_status"].dropna().unique().tolist()) if "work_status" in filtered_df.columns else []
            sel_risk = []
            
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            add_audit_entry("LOGOUT", "—", f"User '{st.session_state.user_name}' logged out")
            for k in ["logged_in","username","role","user_name"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()
            
        st.markdown(f"<div style='font-size:0.65rem;color:#94a3b8;'>Records Visible: {len(filtered_df):,}</div>", unsafe_allow_html=True)
        
    return page, filtered_df, {"state": sel_state, "work_category": sel_cat, "work_status": sel_status, "risk_level": sel_risk}

def apply_filters(df, filters):
    mask = pd.Series(True, index=df.index)
    if "state" in df.columns and "state" in filters and filters["state"]:
        mask &= df["state"].isin(filters["state"])
    if "work_category" in df.columns and "work_category" in filters and filters["work_category"]:
        mask &= df["work_category"].isin(filters["work_category"])
    if "work_status" in df.columns and "work_status" in filters and filters["work_status"]:
        mask &= df["work_status"].isin(filters["work_status"])
    if "risk_level" in df.columns and "risk_level" in filters and filters["risk_level"]:
        mask &= df["risk_level"].isin(filters["risk_level"])
    return df[mask].copy()

def page_dashboard(df, fdf):
    st.markdown('<div class="sentinel-header"><h1>🛡️ MPLADS Sentinel</h1><p>Explainable AI for Public Works Risk, Audit Prioritization & Early Warning &nbsp;|&nbsp; SIH 2026</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ <strong>AI DISCLAIMER:</strong> This system does <em>not</em> determine fraud. It identifies unusual patterns and prioritizes works for human review. All flagged cases require verification by authorized officials.</div>', unsafe_allow_html=True)
    summary = get_data_summary(df)
    high_risk = int((fdf["risk_level"] == "HIGH").sum())
    medium_risk = int((fdf["risk_level"] == "MEDIUM").sum())
    
    if st.session_state.role == "District Nodal Officer":
        evidence_counts = get_all_evidence_counts()
        
        # Rule: Status is Completed or Partially Completed OR >180 days, AND 0 evidence uploaded
        def is_missing_evidence(row):
            sr = str(row.get("sr_no", ""))
            has_evidence = evidence_counts.get(sr, 0) > 0
            if has_evidence:
                return False
            status = str(row.get("work_status", "")).lower()
            if "complete" in status:
                return True
            if row.get("days_since_sanction", 0) > 180:
                return True
            return False
            
        missing_evidence_count = fdf.apply(is_missing_evidence, axis=1).sum()
        
        if missing_evidence_count > 0:
            st.markdown(f"""
            <div style="background:#fef2f2;border:1px solid #ef4444;border-left:5px solid #dc2626;padding:12px 16px;border-radius:6px;margin-bottom:1.5rem;">
                <h4 style="margin:0;color:#991b1b;font-size:1.05rem;">🚨 Missing Evidence Alerts</h4>
                <p style="margin:4px 0 0 0;color:#7f1d1d;font-size:0.85rem;">
                    <strong>{missing_evidence_count} works</strong> in your district are missing required Geotagged Photos (either >180 days since sanction, or marked Completed). Please review these immediately in the Project Detail view to ensure compliance.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (c1,"Total Works",f"{len(fdf):,}",f"of {len(df):,} total"),
        (c2,"Total Sanctioned",fmt_inr_short(fdf["sanction_amount"].sum()),"filtered total"),
        (c3,"Completed",f"{int(fdf['is_completed'].sum()):,}",f"{int(fdf['is_completed'].mean()*100)}% rate"),
        (c4,"🔴 High Risk",f"{high_risk:,}","Priority review"),
        (c5,"🟡 Medium Risk",f"{medium_risk:,}","Flag for audit"),
        (c6,"⏳ Old Pending",f"{int(fdf['delay_flag'].sum()):,}",">1 yr not done"),
    ]
    for col,label,val,sub in kpis:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        risk_counts = fdf["risk_level"].value_counts().reindex(["HIGH","MEDIUM","LOW"], fill_value=0)
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, color=risk_counts.index,
                     color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}, hole=0.55, title="Risk Distribution")
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10), font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        state_risk = fdf.groupby("state")["risk_score"].agg(["mean","count"]).reset_index().rename(columns={"mean":"avg_risk","count":"works"}).sort_values("avg_risk", ascending=False).head(15)
        fig = px.bar(state_risk, x="avg_risk", y="state", orientation="h", color="avg_risk",
                     color_continuous_scale=["#22c55e","#f59e0b","#ef4444"], range_color=[0,100],
                     title="Average Risk Score by State (Top 15)", text=state_risk["avg_risk"].round(1))
        fig.update_layout(height=300, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"}, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    col_c, col_d = st.columns(2)
    with col_c:
        sc = fdf["work_status"].value_counts().reset_index(); sc.columns=["status","count"]
        fig = px.bar(sc, x="count", y="status", orientation="h", color="count", color_continuous_scale="Blues", title="Work Status Distribution", text="count")
        fig.update_layout(height=300, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"}, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        fig = px.histogram(fdf["sanction_amount"].dropna()/1e5, nbins=60, title="Sanction Amount (₹ Lakhs)", color_discrete_sequence=["#3b82f6"])
        fig.update_layout(height=300, font_family="Inter", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    col_e, col_f = st.columns(2)
    with col_e:
        cc = fdf["work_category"].value_counts().reset_index(); cc.columns=["category","count"]
        fig = px.bar(cc, x="count", y="category", orientation="h", color="count", color_continuous_scale="Purples", title="Work Category Distribution", text="count")
        fig.update_layout(height=280, coloraxis_showscale=False, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    with col_f:
        af = pd.DataFrame({"Signal":["Cost Anomaly","Similar/Duplicate","Old Pending"],"Flagged":[int(fdf["cost_anomaly_flag"].sum()),int(fdf["dup_flag"].sum()),int(fdf["delay_flag"].sum())]})
        fig = px.bar(af, x="Signal", y="Flagged", color="Signal", color_discrete_sequence=["#f59e0b","#8b5cf6","#ef4444"], title="AI Detector Flags", text="Flagged")
        fig.update_layout(height=280, font_family="Inter", showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

def page_mospi_admin_settings(df, fdf):
    st.markdown("<div class='section-title'>⚙️ System Tuning & Global Audits (MoSPI Admin)</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Adjust global AI risk thresholds and observe live changes to priority queues. Initiate macro-level state audits.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎛️ AI Sensitivity Tuning")
        st.markdown("Adjust the weight (0-50) given to each risk component. Changes apply instantly.")
        w_c = st.slider("Cost Anomaly Sensitivity", 0, 50, st.session_state.w_cost)
        w_d = st.slider("Semantic Similarity Sensitivity", 0, 50, st.session_state.w_dup)
        w_l = st.slider("Delay / Pending Sensitivity", 0, 50, st.session_state.w_del)
        w_p = st.slider("Pattern Violation Sensitivity", 0, 50, st.session_state.w_pat)
        min_a = st.number_input("Minimum Sanction Amount to Flag (₹ Lakhs)", 0.0, 500.0, st.session_state.min_amt, 1.0)
        
        if st.button("Apply New Sensitivity Weights", type="primary"):
            st.session_state.w_cost = w_c
            st.session_state.w_dup = w_d
            st.session_state.w_del = w_l
            st.session_state.w_pat = w_p
            st.session_state.min_amt = min_a
            st.rerun()
            
    with col2:
        st.subheader("🚔 Initiate Global Audit")
        target_state = st.selectbox("Target State", ["Select State..."] + sorted([str(x) for x in df["state"].dropna().unique()]))
        target_district = st.selectbox("Target District", ["Select District..."] + sorted([str(x) for x in df[df["state"]==target_state]["district"].dropna().unique()] if target_state != "Select State..." else []))
        audit_reason = st.text_area("Reason for Audit", placeholder="e.g., Unusually high concentration of delayed works...")
        if st.button("Trigger Audit Order", type="primary"):
            if target_state != "Select State...":
                add_audit_entry("AUDIT_TRIGGERED", "GLOBAL", f"Initiated audit for {target_state} - {target_district}. Reason: {audit_reason}")
                st.success("✅ Audit order dispatched to District Magistrate and CAG.")
            else:
                st.error("Select a state first.")
                
    st.markdown("---")
    st.subheader("🗺️ Macro Insights & Heatmaps")
    
    mc1, mc2 = st.columns(2)
    with mc1:
        # Top 10 states by HIGH-risk %
        state_totals = fdf.groupby("state").size()
        state_highs = fdf[fdf["risk_level"] == "HIGH"].groupby("state").size()
        pct_high = (state_highs / state_totals * 100).fillna(0).reset_index(name="high_risk_pct")
        pct_high = pct_high[state_totals.reset_index(name="counts")["counts"] > 20] # only states with >20 works
        pct_high = pct_high.sort_values("high_risk_pct", ascending=False).head(10)
        
        fig_st = px.bar(pct_high, x="high_risk_pct", y="state", orientation="h", color="high_risk_pct", 
                        color_continuous_scale="Reds", title="Top 10 States by % of HIGH-Risk Works")
        fig_st.update_layout(height=400, yaxis={"categoryorder":"total ascending"}, font_family="Inter", coloraxis_showscale=False)
        st.plotly_chart(fig_st, use_container_width=True)
        
    with mc2:
        # Heatmap: State x Work Status, cell = median days since sanction
        valid_states = pct_high["state"].tolist()
        hmap_df = fdf[fdf["state"].isin(valid_states)]
        pivot = hmap_df.pivot_table(index="state", columns="work_status", values="days_since_sanction", aggfunc="median").fillna(0)
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="YlOrRd",
            text=np.round(pivot.values, 0),
            texttemplate="%{text} days",
            textfont={"size":10}
        ))
        fig_hm.update_layout(title="Median Delay (Days) by State & Status", height=400, font_family="Inter")
        st.plotly_chart(fig_hm, use_container_width=True)

def page_mp_dashboard(fdf):
    st.markdown("<div class='section-title'>🏛️ My Constituency Dashboard</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Track the progress of your recommended works. This view is strictly restricted to works recommended by you. Risk scores and AI anomalies are kept confidential for auditors.</div>', unsafe_allow_html=True)
    
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Recommended", f"{len(fdf):,}")
    c2.metric("Total Sanctioned", fmt_inr_short(fdf["sanction_amount"].sum()))
    c3.metric("Completed Works", f"{int(fdf['is_completed'].sum()):,}")
    c4.metric("Pending Execution", f"{int(fdf['is_incomplete'].sum()):,}")
    st.markdown("---")
    
    st.subheader("🚦 Bottleneck Tracker")
    st.markdown("Track the exact stage of your pending works.")
    
    pending = fdf[fdf["is_incomplete"] == True].sort_values("days_since_sanction", ascending=False).head(15)
    if pending.empty:
        st.success("🎉 All recommended works have been completed!")
    else:
        for _, row in pending.iterrows():
            days = row.get("days_since_sanction", 0)
            status = str(row.get("work_status", "")).lower()
            
            # Determine active stage index (0 to 4)
            if "complete" in status: active_idx = 4
            elif "progress" in status or "inspect" in status: active_idx = 3
            elif "tender" in status or "vendor" in status or "award" in status: active_idx = 2
            elif "estimat" in status: active_idx = 1
            else: active_idx = 0
                
            stages = ["Sanctioned", "Time Estimation", "Vendor Identification", "Physical Inspection", "Completed"]
            
            stuck_badge = " <span style='background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:bold;margin-left:10px;'>⚠️ STUCK? (>180 Days)</span>" if days > 180 and active_idx < 4 else ""
            
            st.markdown(f"**{row.get('work_description','')}** — {fmt_inr(row.get('sanction_amount',0))}{stuck_badge}", unsafe_allow_html=True)
            
            # Build CSS Progress Bar
            html = "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 25px; margin-top:10px;'>"
            for i, stage in enumerate(stages):
                color = "#22c55e" if i < active_idx else ("#3b82f6" if i == active_idx else "#e2e8f0")
                fw = "bold" if i == active_idx else "normal"
                html += f"<div style='text-align:center; flex:1;'><div style='height:8px; background-color:{color}; border-radius:4px; margin:0 5px;'></div><div style='font-size:0.75rem; margin-top:5px; color:#475569; font-weight:{fw};'>{stage}</div></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📋 Complete Works List")
    display = fdf[["sr_no","work_description","sanction_amount","work_status","sanction_date","days_since_sanction"]].copy()
    display["sanction_amount"] = display["sanction_amount"].apply(fmt_inr)
    display["sanction_date"] = display["sanction_date"].apply(fmt_date)
    display = display.rename(columns={"sr_no":"Sr#","work_description":"Work Description","sanction_amount":"Sanctioned","work_status":"Status","sanction_date":"Sanction Date"})
    st.dataframe(display, use_container_width=True, hide_index=True, height=500)


def page_high_risk(df, fdf):
    st.markdown("<div class='section-title'>🔴 High-Risk & Medium-Risk Works</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ Flagging does <strong>not</strong> imply wrongdoing. All cases require human verification.</div>', unsafe_allow_html=True)
    risk_df = fdf[fdf["risk_level"].isin(["HIGH","MEDIUM"])].sort_values("risk_score", ascending=False)
    c1,c2,c3 = st.columns(3)
    c1.metric("🔴 High-Risk", int((risk_df["risk_level"]=="HIGH").sum()))
    c2.metric("🟡 Medium-Risk", int((risk_df["risk_level"]=="MEDIUM").sum()))
    c3.metric("Total Flagged", len(risk_df))
    display = risk_df[["sr_no","work_description","state","district","sanction_amount","work_status","risk_score","risk_level","main_reason"]].copy()
    display["sanction_amount"] = display["sanction_amount"].apply(fmt_inr)
    display["risk_level"] = display["risk_level"].apply(risk_badge_md)
    display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.0f}/100")
    display["work_description"] = display["work_description"].str[:70]+"…"
    display = display.rename(columns={"sr_no":"Sr#","work_description":"Work Description","state":"State","district":"District","sanction_amount":"Sanctioned","work_status":"Status","risk_score":"Risk Score","risk_level":"Risk Level","main_reason":"Main Reason"})
    st.dataframe(display, use_container_width=True, hide_index=True, height=500)
    hs = risk_df[risk_df["risk_level"]=="HIGH"].groupby("state").size().reset_index(name="count").sort_values("count",ascending=False).head(15)
    if not hs.empty:
        fig = px.bar(hs, x="state", y="count", color="count", color_continuous_scale="Reds", title="High-Risk Works by State", text="count")
        fig.update_layout(height=320, coloraxis_showscale=False, font_family="Inter")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

def page_project_detail(df, similar_map):
    st.markdown("<div class='section-title'>🔍 Project Detail — Explainability View</div>", unsafe_allow_html=True)
    col_sel1, col_sel2 = st.columns([2,1])
    with col_sel1:
        search = st.text_input("Search by work description:", placeholder="e.g. road, school, water…")
    with col_sel2:
        risk_filter = st.selectbox("Filter by risk:", ["ALL","HIGH","MEDIUM","LOW"])
    filtered = df.copy()
    if search:
        filtered = filtered[filtered["work_description"].str.contains(search, case=False, na=False) | filtered["sub_category"].str.contains(search, case=False, na=False)]
    if risk_filter != "ALL":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if filtered.empty:
        st.warning("No matching projects found."); return
    options = {f"#{row['sr_no']} | {str(row['work_description'])[:65]} | {row['state']} | {row['risk_score']:.0f}/100 {row['risk_emoji']}": int(row.name) for _, row in filtered.head(200).iterrows()}
    row_idx = options[st.selectbox("Select project:", list(options.keys()))]
    row = df.loc[row_idx]
    expl = get_explanation(row)
    st.markdown("---")
    h1, h2 = st.columns([3,1])
    with h1:
        st.markdown(f"### {row.get('work_description','N/A')}")
        st.caption(f"Work Code: `{row.get('work_code','N/A')}`")
    with h2:
        level = expl["risk_level"]; score = expl["risk_score"]
        c_bg, c_fg, emoji = RISK_COLORS.get(level, ("#e2e8f0","#1e293b","⚪"))
        st.markdown(f'<div style="text-align:center;padding:1rem;background:{c_fg};border-radius:12px;"><div style="font-size:2.5rem;font-weight:800;color:{c_bg};">{score:.0f}</div><div style="font-size:0.8rem;font-weight:700;color:{c_bg};">/ 100 &nbsp;{emoji} {level} RISK</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Project Details</div>", unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)
    with d1:
        st.markdown(f"**State:** {row.get('state','N/A')}\n\n**District:** {row.get('district','N/A')}\n\n**Constituency:** {row.get('constituency','N/A')}")
    with d2:
        st.markdown(f"**MP:** {row.get('mp_name_clean','N/A')}\n\n**Category:** {row.get('work_category','N/A')}\n\n**Status:** {status_emoji(row.get('work_status',''))} {row.get('work_status','N/A')}")
    with d3:
        st.markdown(f"**Sanctioned:** {fmt_inr(row.get('sanction_amount',0))}\n\n**Recommended:** {fmt_date(row.get('recommended_date'))}\n\n**Sanctioned On:** {fmt_date(row.get('sanction_date'))}")
    record_hash = short_hash({"sr_no":str(row.get("sr_no")),"sanction_amount":str(row.get("sanction_amount")),"work_status":str(row.get("work_status"))})
    st.markdown(f"<span style='font-size:0.72rem;color:#94a3b8;'>🔒 Record Hash: <span class='hash-chip'>{record_hash}</span></span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if row.get("data_quality_flag", False):
        st.markdown("<div style='background:rgba(239, 68, 68, 0.12); border:1px solid #ef4444; border-left:5px solid #dc2626; padding:10px 14px; border-radius:6px; margin-bottom:15px;'><span style='color:#f87171; font-weight:bold;'>⚠️ Missing / Corrupt Data Indicator</span><br><span style='font-size:0.85rem; color:#cbd5e1;'>This record contains garbled descriptions or unparseable amounts. Proceed with caution.</span></div>", unsafe_allow_html=True)
        
    st.markdown("<div class='section-title'>⚠️ WHY FLAGGED? — Evidence Cards</div>", unsafe_allow_html=True)
    for reason in expl["reasons"]:
        sev = reason.get("severity","MEDIUM")
        card_cls = "high" if sev=="HIGH" else ("low" if sev=="LOW" else "")
        st.markdown(f'<div class="evidence-card {card_cls}"><span class="evidence-title">{reason["signal"]}</span><span class="evidence-pts">{reason["points"]}</span><div class="evidence-detail">{reason["detail"]}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br><div class='section-title'>📊 Risk Score Breakdown</div>", unsafe_allow_html=True)
    bd = pd.DataFrame({
        "Component":["Cost Anomaly (max 30)","Similar Work (max 25)","Delay/Pending (max 20)","Rule Violations (max 25)"],
        "Score":[expl["cost_pts"], expl["dup_pts"], expl["delay_pts"], expl.get("rule_pts",0)],
        "Max":[30, 25, 20, 25]
    })
    bd["Remaining"] = bd["Max"] - bd["Score"]
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(name="Score", x=bd["Score"], y=bd["Component"], orientation="h", marker_color=["#ef4444","#8b5cf6","#f59e0b","#0ea5e9"], text=bd["Score"].apply(lambda x: f"{x:.0f} pts" if x>0 else ""), textposition="inside"))
    fig_s.add_trace(go.Bar(name="Remaining", x=bd["Remaining"], y=bd["Component"], orientation="h", marker_color="#334155"))
    fig_s.update_layout(barmode="stack", height=220, showlegend=False, margin=dict(l=10,r=10,t=10,b=10), font_family="Inter")
    st.plotly_chart(fig_s, use_container_width=True)
    
    shap_vals = row.get("shap_values")
    if isinstance(shap_vals, str):
        try:
            shap_vals = json.loads(shap_vals)
        except Exception:
            shap_vals = {}
            
    if isinstance(shap_vals, dict) and shap_vals and sum(abs(v) for v in shap_vals.values()) > 0:
        st.markdown("<br><div class='section-title'>🧠 AI Explanation (SHAP Values)</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem;color:#cbd5e1;margin-bottom:10px;'>This chart shows which features caused the Isolation Forest ML model to increase (+) or decrease (-) the Cost Anomaly score for this specific work.</div>", unsafe_allow_html=True)
        shap_df = pd.DataFrame(list(shap_vals.items()), columns=["Feature", "Impact"])
        shap_df = shap_df.sort_values(by="Impact")
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["Impact"]]
        fig_shap = go.Figure(go.Bar(x=shap_df["Impact"], y=shap_df["Feature"], orientation="h", marker_color=colors, text=shap_df["Impact"].apply(lambda x: f"{x:+.3f}"), textposition="outside"))
        fig_shap.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), font_family="Inter")
        st.plotly_chart(fig_shap, use_container_width=True)
    peer_median = row.get("peer_median", np.nan); peer_p75 = row.get("peer_p75", np.nan)
    peer_p90 = row.get("peer_p90", np.nan); peer_size = row.get("peer_size", 0)
    this_amount = row.get("sanction_amount", np.nan); peer_label = row.get("peer_group_label","N/A")
    ratio = this_amount / peer_median if pd.notna(this_amount) and pd.notna(peer_median) and peer_median > 0 else 1.0
    
    if isinstance(shap_vals, dict) and shap_vals and sum(abs(v) for v in shap_vals.values()) > 0:
        st.markdown(f"**Amount is {ratio:.1f}× peer median.**")
        
    if pd.notna(peer_median) and peer_median > 0:
        st.markdown("<div class='section-title'>🏷️ Peer Comparison</div>", unsafe_allow_html=True)
        st.markdown(f"| Metric | This Work | Peer Group ({peer_label}, n={int(peer_size)}) |\n|--------|-----------|------|\n| Amount | **{fmt_inr(this_amount)}** | Median: {fmt_inr(peer_median)} |\n| vs Median | **{ratio:.1f}×** | P75: {fmt_inr(peer_p75)} |\n| | | P90: {fmt_inr(peer_p90)} |")
        fig_p = go.Figure(go.Bar(x=["Peer Median","Peer P75","Peer P90","This Work"], y=[peer_median/1e5,peer_p75/1e5,peer_p90/1e5,this_amount/1e5], marker_color=["#3b82f6","#f59e0b","#f97316","#ef4444"], text=[fmt_inr(peer_median),fmt_inr(peer_p75),fmt_inr(peer_p90),fmt_inr(this_amount)], textposition="outside"))
        fig_p.update_layout(height=260, yaxis_title="₹ Lakhs", font_family="Inter", showlegend=False, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig_p, use_container_width=True)
    t1,t2,t3,t4 = st.columns(4)
    rec_to_san = row.get("rec_to_sanction_days", np.nan); days_since = row.get("days_since_sanction", np.nan)
    t1.metric("📅 Recommended", fmt_date(row.get("recommended_date")))
    t2.metric("✅ Sanctioned", fmt_date(row.get("sanction_date")))
    t3.metric("⏱ Rec→Sanction", f"{int(rec_to_san)} days" if pd.notna(rec_to_san) else "N/A", delta="⚠ Above 75-day guideline" if pd.notna(rec_to_san) and rec_to_san>75 else None, delta_color="inverse")
    t4.metric("📆 Days Since Sanction", f"{int(days_since)} days" if pd.notna(days_since) else "N/A")
    row_id = int(row.get("row_id",-1)); matches = similar_map.get(row_id,[])
    if matches:
        st.markdown("<br><div class='section-title'>🔗 Potentially Similar Works</div>", unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">High semantic similarity found. Does <strong>not</strong> confirm duplication — human review required.</div>', unsafe_allow_html=True)
        for other_row_id, sim_score in matches[:5]:
            other_rows = df[df["row_id"]==other_row_id]
            if other_rows.empty: continue
            other = other_rows.iloc[0]
            badge_color = "#ef4444" if sim_score>=0.88 else "#f59e0b"
            st.markdown(f'<div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;"><span style="background:{badge_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700;">Similarity: {sim_score*100:.0f}%</span>&nbsp; <span style="color:#cbd5e1;">Sr# {other.get("sr_no","")} — {other.get("state","")} / {other.get("district","")}</span><br><span style="font-size:0.9rem; color:#f8fafc; font-weight:500; margin-top:4px; display:inline-block;">{str(other.get("work_description",""))[:120]}</span><br><span style="font-size:0.8rem; color:#94a3b8;">Sanctioned: {fmt_inr(other.get("sanction_amount",0))} &nbsp;|&nbsp; Status: {other.get("work_status","")} &nbsp;|&nbsp; Risk: {other.get("risk_score",0):.0f}/100</span></div>', unsafe_allow_html=True)
            
    action_border = {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}.get(level,"#3b82f6")
    action_bg = {"HIGH":"rgba(239, 68, 68, 0.12)","MEDIUM":"rgba(245, 158, 11, 0.12)","LOW":"rgba(34, 197, 94, 0.12)"}.get(level,"rgba(59, 130, 246, 0.12)")
    st.markdown(f'<div style="background:{action_bg}; border-left:5px solid {action_border}; border-radius:8px; padding:1.1rem 1.3rem; margin-top:1rem;"><div style="font-weight:700; font-size:0.95rem; color:#ffffff;">📋 Recommended Action</div><div style="font-size:0.88rem; color:#cbd5e1; margin-top:0.35rem; line-height:1.45;">{expl["recommended_action"]}</div></div>', unsafe_allow_html=True)

    # Evidence Upload for Nodal Officer
    if st.session_state.role == "District Nodal Officer":
        st.markdown("<br><div class='section-title'>📸 Geotagged Evidence Upload (eSAKSHI Integration)</div>", unsafe_allow_html=True)
        st.markdown("Upload physical inspection photos to clear Missing Evidence alerts.")
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg","png","jpeg"], key=f"upload_{row_idx}")
        with up_col2:
            img_type = st.selectbox("Image Type", ["Before Execution", "During Execution", "After Completion"], key=f"type_{row_idx}")
            if uploaded_file is not None:
                if st.button("Submit Evidence", type="primary"):
                    add_evidence(row.get("sr_no"), uploaded_file.name, st.session_state.user_name, img_type)
                    st.success("✅ Evidence uploaded successfully and logged to DB.")
                    
        # Show existing uploads
        existing_evidence = get_evidence_for_work(row.get("sr_no"))
        if existing_evidence:
            st.markdown("**Uploaded Evidence:**")
            for ev in existing_evidence:
                st.markdown(f"- 📄 `{ev['filename']}` ({ev['type']}) — uploaded by {ev['uploaded_by']} on {ev['timestamp'][:10]}")

def page_risk_analysis(df, fdf):
    st.markdown("<div class='section-title'>📈 Risk Analysis Overview</div>", unsafe_allow_html=True)
    sample = fdf.sample(min(3000,len(fdf)), random_state=42)
    fig = px.scatter(sample, x="sanction_amount", y="risk_score", color="risk_level", color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}, hover_data=["work_description","state","district"], title="Sanction Amount vs Risk Score", opacity=0.6)
    fig.update_layout(height=380, font_family="Inter")
    st.plotly_chart(fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(fdf, x="cost_anomaly_score", nbins=50, color="cost_anomaly_flag", color_discrete_map={True:"#ef4444",False:"#3b82f6"}, title="Cost Anomaly Score Distribution")
        fig.update_layout(height=300, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.histogram(fdf, x="delay_score", nbins=50, color="delay_flag", color_discrete_map={True:"#ef4444",False:"#64748b"}, title="Delay Score Distribution")
        fig.update_layout(height=300, font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    fy_risk = fdf.groupby("financial_year").agg(avg_risk=("risk_score","mean"),works=("row_id","count")).reset_index().sort_values("financial_year")
    fig_fy = make_subplots(specs=[[{"secondary_y":True}]])
    fig_fy.add_trace(go.Bar(x=fy_risk["financial_year"],y=fy_risk["works"],name="Works",marker_color="#bfdbfe"),secondary_y=False)
    fig_fy.add_trace(go.Scatter(x=fy_risk["financial_year"],y=fy_risk["avg_risk"],name="Avg Risk",mode="lines+markers",line=dict(color="#ef4444",width=2)),secondary_y=True)
    fig_fy.update_layout(title="Works & Avg Risk by Financial Year",height=300,font_family="Inter")
    fig_fy.update_yaxes(title_text="Works",secondary_y=False)
    fig_fy.update_yaxes(title_text="Avg Risk Score",secondary_y=True)
    st.plotly_chart(fig_fy, use_container_width=True)

REVIEW_STATUS_CSS = {"NEW":"status-new","UNDER REVIEW":"status-review","VERIFIED":"status-review","ESCALATED":"status-escalated","CLEARED":"status-cleared","INSPECTED":"status-inspected"}

def _add_note(sr, note):
    if sr not in st.session_state.review_notes:
        st.session_state.review_notes[sr] = []
    st.session_state.review_notes[sr].append({"time":datetime.datetime.now().strftime("%d %b %Y %H:%M"),"user":st.session_state.user_name,"note":note})

def page_review_workflow(df, fdf, similar_map):
    st.markdown("<div class='section-title'>📋 Human-in-the-Loop Review Workflow</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Reviewers can acknowledge, verify, escalate, or clear flagged works. Every action is recorded in the tamper-evident audit trail. <strong>No AI decision is final — human judgment is required.</strong></div>', unsafe_allow_html=True)
    st.markdown('**Workflow:** &nbsp;<span class="status-new">NEW</span> → <span class="status-review">UNDER REVIEW</span> → <span class="status-review">VERIFIED</span> → <span class="status-escalated">ESCALATED</span> / <span class="status-cleared">CLEARED</span> / <span class="status-inspected">INSPECTED</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    statuses = st.session_state.review_statuses
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Actions Taken", len(statuses))
    c2.metric("🔺 Escalated", sum(1 for v in statuses.values() if v=="ESCALATED"))
    c3.metric("✅ Cleared", sum(1 for v in statuses.values() if v=="CLEARED"))
    c4.metric("🔍 Inspection Requested", sum(1 for v in statuses.values() if v=="INSPECTED"))
    st.markdown("---")
    
    # Use fdf (filtered dataframe based on role) instead of full df for queue
    queue_df = fdf[fdf["risk_level"].isin(["HIGH","MEDIUM"])].sort_values("risk_score",ascending=False)
    tab1, tab2 = st.tabs(["📂 Review Queue","✅ Reviewed Cases"])
    with tab1:
        st.markdown(f"**{len(queue_df)} flagged works in queue for your jurisdiction**")
        search_rev = st.text_input("Filter queue:", key="rev_search", placeholder="e.g. road, water…")
        if search_rev:
            queue_df = queue_df[queue_df["work_description"].str.contains(search_rev,case=False,na=False)]
        for _, row in queue_df.head(30).iterrows():
            sr = str(row.get("sr_no",""))
            current_status = statuses.get(sr,"NEW")
            status_css = REVIEW_STATUS_CSS.get(current_status,"status-new")
            level = row.get("risk_level","LOW")
            c_bg, c_fg, emoji = RISK_COLORS.get(level,("#e2e8f0","#1e293b","⚪"))
            
            # Missing Geotagged Photos Alert Logic (Mock simulation)
            missing_photos = (hash(sr) % 3 == 0) and row.get("is_completed", False) # Just randomly flag some completed works for missing evidence
            
            with st.expander(f"{emoji} Sr#{sr} | {str(row.get('work_description',''))[:70]} | Risk: {row.get('risk_score',0):.0f}/100 | {current_status}"):
                if missing_photos and current_status == "NEW" and st.session_state.role == "District Nodal Officer":
                    st.warning("⚠️ **Alert:** Geotagged Completion Evidence is missing from eSAKSHI portal for this work.", icon="📸")
                
                col_info, col_action = st.columns([2,1])
                with col_info:
                    st.markdown(f"**State:** {row.get('state','')} | **District:** {row.get('district','')} | **MP:** {row.get('mp_name_clean','')}\n\n**Sanctioned:** {fmt_inr(row.get('sanction_amount',0))} | **Status:** {row.get('work_status','')} | **Days since sanction:** {int(row.get('days_since_sanction',0))}\n\n**Main Reason:** {row.get('main_reason','')}")
                    expl = get_explanation(row)
                    for r in expl["reasons"]:
                        if r.get("points"):
                            st.markdown(f"<small>• {r['signal']} {r['points']}: {r['detail'][:100]}</small>", unsafe_allow_html=True)
                    notes = st.session_state.review_notes.get(sr,[])
                    if notes:
                        st.markdown("**📝 Reviewer Notes:**")
                        for n in notes:
                            st.markdown(f"<div class='audit-entry'>📝 [{n['time']}] <strong>{n['user']}</strong>: {n['note']}</div>", unsafe_allow_html=True)
                with col_action:
                    st.markdown(f'<div style="text-align:center;margin-bottom:1rem;"><span class="{status_css}">{current_status}</span></div>', unsafe_allow_html=True)
                    note_text = st.text_area("Add note:", key=f"note_{sr}", height=80, placeholder="Enter review observation…")
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("📋 Under Review", key=f"ur_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr]="UNDER REVIEW"; add_audit_entry("STATUS_CHANGE",sr,"Marked as UNDER REVIEW")
                            if note_text: _add_note(sr,note_text)
                            st.rerun()
                        if st.button("✅ Clear", key=f"cl_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr]="CLEARED"; add_audit_entry("STATUS_CHANGE",sr,f"CLEARED — {note_text or 'No note'}")
                            if note_text: _add_note(sr,note_text); st.success(f"Sr# {sr} cleared."); st.rerun()
                    with b2:
                        if st.button("🔺 Escalate", key=f"esc_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr]="ESCALATED"; add_audit_entry("STATUS_CHANGE",sr,f"ESCALATED — {note_text or 'No note'}")
                            if note_text: _add_note(sr,note_text); st.warning(f"Sr# {sr} escalated."); st.rerun()
                        if st.button("🔍 Inspect", key=f"ins_{sr}", use_container_width=True):
                            st.session_state.review_statuses[sr]="INSPECTED"; add_audit_entry("STATUS_CHANGE",sr,"Field inspection requested")
                            if note_text: _add_note(sr,note_text); st.info(f"Sr# {sr} sent for inspection."); st.rerun()
                    if note_text and st.button("💾 Save Note Only", key=f"sn_{sr}", use_container_width=True):
                        _add_note(sr,note_text); add_audit_entry("NOTE_ADDED",sr,note_text); st.success("Note saved."); st.rerun()
    with tab2:
        reviewed = {k:v for k,v in statuses.items() if v!="NEW"}
        if not reviewed:
            st.info("No cases reviewed yet.")
        else:
            for sr_no, status in reviewed.items():
                notes = st.session_state.review_notes.get(sr_no,[])
                css = REVIEW_STATUS_CSS.get(status,"status-new")
                rows_match = df[df["sr_no"]==sr_no]
                desc = rows_match.iloc[0]["work_description"][:70] if not rows_match.empty else "—"
                notes_html = "<br>".join([f"<small style='color:#cbd5e1;'>📝 [{n['time']}] <strong style='color:#38bdf8;'>{n['user']}</strong>: {n['note']}</small>" for n in notes]) if notes else ""
                st.markdown(f'<div style="background:#0f172a; border:1px solid #334155; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.7rem; color:#f8fafc;"><strong style="color:#ffffff;">Sr# {sr_no}</strong> &nbsp;<span class="{css}">{status}</span><br><small style="color:#94a3b8;">{desc}…</small><br>{notes_html}</div>', unsafe_allow_html=True)

def page_investigation_report(df, similar_map):
    st.markdown("<div class='section-title'>📄 Automated Investigation Report</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">This report identifies <strong>potential irregularities</strong> for human review. It does NOT constitute a legal or audit conclusion.</div>', unsafe_allow_html=True)
    search = st.text_input("Search project:", placeholder="Enter description keyword or Sr#…")
    filtered = df.copy()
    if search:
        filtered = filtered[filtered["work_description"].str.contains(search,case=False,na=False)|filtered["sr_no"].astype(str).str.contains(search,na=False)]
    if filtered.empty:
        st.warning("No projects found."); return
    options = {f"#{row['sr_no']} | {str(row['work_description'])[:65]} | {row['risk_score']:.0f}/100 {row['risk_emoji']}": int(row.name) for _, row in filtered.head(100).iterrows()}
    row_idx = options[st.selectbox("Select project for report:", list(options.keys()))]
    row = df.loc[row_idx]; expl = get_explanation(row); sr = str(row.get("sr_no",""))
    reviewer_notes = st.session_state.review_notes.get(sr,[]); review_status = st.session_state.review_statuses.get(sr,"NEW")
    report_text = _generate_report(row, expl, similar_map, df, reviewer_notes, review_status)
    st.markdown("---")
    st.markdown("### 📋 Report Preview")
    st.code(report_text, language="text")
    st.download_button(label="⬇️ Download Investigation Report (.txt)", data=report_text, file_name=f"MPLADS_Sentinel_Report_Sr{sr}_{datetime.date.today()}.txt", mime="text/plain", type="primary", use_container_width=True)
    add_audit_entry("REPORT_GENERATED", sr, f"Investigation report generated for Sr# {sr}")

def _generate_report(row, expl, similar_map, df, reviewer_notes, review_status):
    sr = str(row.get("sr_no","")); now = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    user = st.session_state.user_name; role = st.session_state.role
    row_id = int(row.get("row_id",-1)); matches = similar_map.get(row_id,[])
    similar_lines = []
    for other_row_id, sim_score in matches[:5]:
        other_rows = df[df["row_id"]==other_row_id]
        if not other_rows.empty:
            o = other_rows.iloc[0]
            similar_lines.append(f"  • Sr# {o.get('sr_no','')} | Similarity: {sim_score*100:.0f}% | {str(o.get('work_description',''))[:60]} | {fmt_inr(o.get('sanction_amount',0))}")
    notes_lines = "\n".join([f"  [{n['time']}] {n['user']}: {n['note']}" for n in reviewer_notes]) or "  No reviewer notes added."
    reasons_lines = "\n".join([f"  {r['signal']} {r['points']}\n  → {r['detail']}" for r in expl["reasons"]])
    record_hash = hash_record({"sr_no":sr,"sanction_amount":str(row.get("sanction_amount")),"work_status":str(row.get("work_status")),"risk_score":str(expl["risk_score"])})
    return f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              MPLADS SENTINEL — INVESTIGATION REPORT                      ║
╚══════════════════════════════════════════════════════════════════════════╝

REPORT GENERATED : {now}
GENERATED BY     : {user} ({role})
REPORT HASH      : {record_hash[:24].upper()}

DISCLAIMER: This report identifies POTENTIAL IRREGULARITIES based on
algorithmic pattern detection. It does NOT constitute proof of fraud,
misconduct, or legal violation. All findings require verification by
authorized officials before any action is taken.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sr. No.          : {sr}
  Work Description : {row.get('work_description','N/A')}
  Work Code        : {row.get('work_code','N/A')}
  Work Category    : {row.get('work_category','N/A')}
  State            : {row.get('state','N/A')}
  District (IDA)   : {row.get('district','N/A')}
  Constituency     : {row.get('constituency','N/A')}
  MP               : {row.get('mp_name_clean','N/A')}
  Financial Year   : {row.get('financial_year','N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. FINANCIAL DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sanctioned Amount  : {fmt_inr(row.get('sanction_amount',0))}
  Work Status        : {row.get('work_status','N/A')}
  Recommended Date   : {fmt_date(row.get('recommended_date'))}
  Sanction Date      : {fmt_date(row.get('sanction_date'))}
  Rec→Sanction Gap   : {int(row.get('rec_to_sanction_days',0))} days (guideline: ≤75 days)
  Days Since Sanction: {int(row.get('days_since_sanction',0))} days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. REVIEW PRIORITY SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOTAL SCORE : {expl['risk_score']:.0f} / 100
  RISK LEVEL  : {expl['risk_emoji']} {expl['risk_level']}
  Cost Anomaly (max 30)     : {expl['cost_pts']:.0f} pts
  Similar Work (max 25)     : {expl['dup_pts']:.0f} pts
  Delay/Pending (max 20)    : {expl['delay_pts']:.0f} pts
  Pattern Flags (max 25)    : {expl.get('rule_pts', 0):.0f} pts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. ANOMALY SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{reasons_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. PEER COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Peer Group  : {row.get('peer_group_label','N/A')} (n={int(row.get('peer_size',0))})
  This Work   : {fmt_inr(row.get('sanction_amount',0))}
  Peer Median : {fmt_inr(row.get('peer_median',0))}
  Peer P75    : {fmt_inr(row.get('peer_p75',0))}
  Peer P90    : {fmt_inr(row.get('peer_p90',0))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. POTENTIALLY SIMILAR / DUPLICATE WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(similar_lines) if similar_lines else "  No highly similar works detected above threshold."}
  NOTE: Similarity does NOT confirm duplication. Human verification required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. REVIEW STATUS & NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Review Status : {review_status}
{notes_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. RECOMMENDED ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {expl['recommended_action']}

TAMPER-EVIDENT HASH: {record_hash}
MPLADS Sentinel — SIH 2026 | Prototype Only | NOT an official audit document
══════════════════════════════════════════════════════════════════════════
"""

def page_audit_trail():
    st.markdown("<div class='section-title'>🔒 Complete Audit Trail — Tamper-Evident Log</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Every action is logged with a SHA-256 chain hash. Any modification to a past entry breaks the chain — providing tamper-evidence.</div>', unsafe_allow_html=True)
    log = get_audit_logs()
    if not log:
        st.info("No audit entries yet. Login, review a project, or generate a report to create log entries.")
        return
        
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Log Entries", len(log))
    c2.metric("Unique Users", len({e["user"] for e in log}))
    c3.metric("Actions Taken", len({e["action"] for e in log}))
    st.markdown("---")
    
    col_v, col_t = st.columns([1, 1])
    with col_v:
        if st.button("✅ Verify Chain Integrity", use_container_width=True):
            chain_ok, broken_links = verify_chain()
            if chain_ok:
                st.success(f"✅ Chain intact ({len(log)} entries)")
            else:
                tampered_id = broken_links[0]['id']
                st.error(f"❌ Tampered at entry #{tampered_id}")
                for b in broken_links:
                    st.write(b)
                    
    with col_t:
        if st.button("🚨 Tamper DB (Demo Mode)", type="primary", use_container_width=True):
            tamper_demo_record()
            st.warning("A record in the SQLite database was just manually tampered with! Click 'Verify Chain Integrity' to see the system catch it.")
            
    st.markdown("---")
    
    action_filter = st.multiselect("Filter by action:", ["All","LOGIN","LOGOUT","STATUS_CHANGE","NOTE_ADDED","REPORT_GENERATED","CITIZEN_FEEDBACK", "AUDIT_TRIGGERED"], default=["All"])
    
    filtered_log = [
        e for e in log 
        if "All" in action_filter or e["action"] in action_filter
    ]
    
    tab_cards, tab_table = st.tabs(["📜 Visual Tamper-Evident Ledger", "📊 Tabular Audit Log"])
    
    with tab_cards:
        if not filtered_log:
            st.info("No matching audit entries.")
        for entry in filtered_log:
            act = entry["action"]
            icon = {"LOGIN":"🔓","LOGOUT":"🔒","STATUS_CHANGE":"🔄","NOTE_ADDED":"📝","REPORT_GENERATED":"📄","CITIZEN_FEEDBACK":"💬", "AUDIT_TRIGGERED":"🚔"}.get(act,"•")
            border_color = {
                "LOGIN": "#3b82f6",
                "STATUS_CHANGE": "#f59e0b",
                "AUDIT_TRIGGERED": "#ef4444",
                "REPORT_GENERATED": "#a855f7",
                "CITIZEN_FEEDBACK": "#06b6d4"
            }.get(act, "#3b82f6")
            
            ts_clean = entry["timestamp"][:19].replace("T", " ")
            proj_id = entry.get("project_id") or "—"
            
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #334155; border-left:5px solid {border_color}; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.75rem; color:#f8fafc;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                    <span style="font-weight:700; font-size:0.95rem; color:#60a5fa;">{icon} {act}</span>
                    <span style="font-size:0.8rem; color:#94a3b8; font-family:monospace;">🕒 {ts_clean}</span>
                </div>
                <div style="font-size:0.86rem; color:#e2e8f0; margin-bottom:0.45rem;">
                    <strong style="color:#ffffff;">👤 {entry['user']}</strong> 
                    <span style="color:#94a3b8;">({entry['role']})</span> &nbsp;•&nbsp; 
                    <span style="color:#cbd5e1;">Project ID:</span> <code style="background:#1e293b; color:#38bdf8; padding:2px 7px; border-radius:4px; font-size:0.82rem;">{proj_id}</code>
                </div>
                <div style="font-size:0.85rem; color:#f1f5f9; background:#1e293b; border:1px solid #334155; padding:0.6rem 0.9rem; border-radius:6px; margin-bottom:0.55rem; line-height:1.4;">
                    {entry['detail']}
                </div>
                <div style="display:flex; gap:10px; font-family:monospace; font-size:0.75rem; flex-wrap:wrap;">
                    <span style="background:#0369a1; color:#ffffff; padding:3px 8px; border-radius:4px; font-weight:600;">PREV: {entry['prev_hash'][:14].upper()}</span>
                    <span style="background:#15803d; color:#ffffff; padding:3px 8px; border-radius:4px; font-weight:600;">HASH: {entry['hash'][:14].upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with tab_table:
        log_df = pd.DataFrame(filtered_log)
        if not log_df.empty:
            cols_to_show = ["id", "timestamp", "user", "role", "action", "project_id", "detail", "hash"]
            cols_present = [c for c in cols_to_show if c in log_df.columns]
            st.dataframe(log_df[cols_present], use_container_width=True, hide_index=True)
            
    st.download_button("⬇️ Download Full Audit Log (JSON)", data=json.dumps(log,indent=2,default=str), file_name=f"audit_log_{datetime.date.today()}.json", mime="application/json")

def page_citizen_feedback(fdf):
    st.markdown("<div class='section-title'>💬 Citizen Feedback & Issue Reporting</div>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Citizens can report concerns about specific MPLADS works. All reports require verification by authorized officials before any action is taken.</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📝 Submit Feedback","📋 All Submissions"])
    with tab1:
        with st.form("citizen_feedback_form", clear_on_submit=True):
            st.markdown("#### Report a Concern")
            sr_no_input = st.text_input("Work Sr. No. (if known):", placeholder="e.g. 4821")
            state_input = st.selectbox("State:", ["Select…"]+sorted(fdf["state"].dropna().unique().tolist()))
            district_input = st.text_input("District:", placeholder="e.g. Agra")
            category = st.selectbox("Issue Category:", ["Select category…","Work not started despite sanction","Work appears to be duplicate of another","Cost appears unusually high","Work completed on paper but not physically","Incorrect location / beneficiary","Implementation agency related concern","Other"])
            description = st.text_area("Describe the concern:", height=120, placeholder="Please describe what you observed…")
            contact = st.text_input("Contact (optional):", placeholder="Email or phone for follow-up")
            is_anonymous = st.checkbox("Submit anonymously")
            if st.form_submit_button("📤 Submit Report", type="primary", use_container_width=True):
                if not description or category=="Select category…":
                    st.error("Please fill in the issue category and description.")
                else:
                    entry = {"id":len(st.session_state.feedback_list)+1,"timestamp":datetime.datetime.now().strftime("%d %b %Y %H:%M"),"sr_no":sr_no_input or "—","state":state_input,"district":district_input,"category":category,"description":description,"contact":"Anonymous" if is_anonymous else (contact or "Not provided"),"status":"RECEIVED"}
                    st.session_state.feedback_list.append(entry)
                    add_audit_entry("CITIZEN_FEEDBACK",sr_no_input or "—",f"Citizen report received — {category}")
                    st.success(f"✅ Thank you! Your report has been submitted (Ref# CF-{entry['id']:04d}).")
    with tab2:
        if st.session_state.role not in ["MoSPI Admin","District Nodal Officer"]:
            st.warning("Access restricted to Administrative roles."); return
        feedback = st.session_state.feedback_list
        if not feedback:
            st.info("No citizen feedback received yet.")
        else:
            st.metric("Total Submissions", len(feedback))
            fb_df = pd.DataFrame(feedback)
            st.dataframe(fb_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Feedback CSV", data=fb_df.to_csv(index=False), file_name=f"citizen_feedback_{datetime.date.today()}.csv", mime="text/csv")

def page_transparency_portal(df):
    st.markdown('<div class="sentinel-header"><h1>🌐 Public Transparency Portal</h1><p>MPLADS Works — Public Dashboard | Read-Only View</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Works", f"{len(df):,}")
    c2.metric("Total Sanctioned", fmt_inr_short(df["sanction_amount"].sum()))
    c3.metric("States Covered", df["state"].nunique())
    c4.metric("Completed Works", int(df["is_completed"].sum()))
    ss = df.groupby("state").agg(total_works=("state","count"),completed=("is_completed","sum"),total_amount=("sanction_amount","sum")).reset_index()
    ss["completion_rate"] = (ss["completed"]/ss["total_works"]*100).round(1)
    ss["total_amount_fmt"] = ss["total_amount"].apply(fmt_inr)
    ss = ss.sort_values("total_works",ascending=False)
    st.markdown("#### State-wise MPLADS Works Summary")
    st.dataframe(ss[["state","total_works","completed","completion_rate","total_amount_fmt"]].rename(columns={"state":"State","total_works":"Total Works","completed":"Completed","completion_rate":"Completion Rate (%)","total_amount_fmt":"Total Sanctioned"}), use_container_width=True, hide_index=True)
    fig = px.bar(ss.head(20), x="state", y="total_works", color="completion_rate", color_continuous_scale="RdYlGn", title="Works by State (colour = Completion Rate %)", text="total_works")
    fig.update_layout(height=380, font_family="Inter"); fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🔍 Search Works & View Evidence")
    pub_search = st.text_input("Search (e.g. road, Bihar, school…):", placeholder="Enter keyword...")
    if pub_search:
        pub_results = df[df["work_description"].str.contains(pub_search,case=False,na=False)|df["state"].str.contains(pub_search,case=False,na=False)].head(20)
        if pub_results.empty:
            st.info("No works found.")
        else:
            for _, row in pub_results.iterrows():
                with st.expander(f"Sr#{row.get('sr_no','')} | {row.get('state','')} | {row.get('work_status','')}"):
                    st.write(f"**Description:** {row.get('work_description','')}")
                    st.write(f"**Sanctioned:** {fmt_inr(row.get('sanction_amount',0))} | **Sanction Date:** {fmt_date(row.get('sanction_date'))}")
                    
                    existing_evidence = get_evidence_for_work(row.get('sr_no'))
                    
                    if existing_evidence:
                        st.markdown("**📸 Execution Evidence (Simulated evidence records; eSAKSHI geotag integration planned)**")
                        for ev in existing_evidence:
                            st.info(f"📄 {ev['filename']} ({ev['type']}) — Uploaded: {ev['timestamp'][:10]}")
                    else:
                        st.markdown("*No geotagged evidence uploaded yet.*")

def page_about():
    st.markdown("<div class='section-title'>ℹ️ About MPLADS Sentinel & Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
### What is MPLADS Sentinel?
An AI-assisted intelligence layer on top of MPLADS sanctioned-works data.
> **This system does NOT replace government audits.** It helps authorities identify which works deserve attention first and explains why.

### AI Pipeline
```
Works Sanctioned CSV (19,001 rows)
      ↓
Preprocessing → Clean, Extract, Engineer Features
      ↓
Detector 1: Cost Anomaly (Isolation Forest per peer group)     → 0–30 pts
Detector 2: Semantic Similarity (sentence-transformers+cosine) → 0–25 pts
Detector 3: Delay / Old Pending (Statistical)                  → 0–20 pts
Detector 4: Pattern Flags (Deterministic rule-based heuristics)→ 0–25 pts
      ↓
Risk Fusion → 0–100 Review Priority Score
      ↓
Human-in-the-Loop Review → Verify / Escalate / Clear / Inspect
      ↓
Tamper-Evident Audit Trail (SHA-256 chain hashing)
```

### Risk Score Weights (Default)
| Signal | Weight | Max Points |
|--------|--------|-----------|
| Cost Anomaly | 30% | 30 |
| Similar / Duplicate | 25% | 25 |
| Delay / Old Pending | 20% | 20 |
| Pattern Flags | 25% | 25 |

### Security & Governance
| Feature | Implementation |
|---------|---------------|
| Role-Based Access Control | 4 roles: MoSPI Admin, District Nodal Officer, MP, Public |
| Tamper-Evident Logs | SHA-256 chain hashing |
| Complete Audit Trail | Every action logged with timestamp, user, role |
| Human-in-the-Loop | No automated action — human approves every decision |
| Citizen Feedback | Public issue reporting |
| Public Transparency | Read-only portal |

**SIH 2026 | Problem Statement SIH26102 | Prototype Only**
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Data Available")
        for f in AVAILABLE_FIELDS: st.markdown(f"- {f}")
    with col2:
        st.markdown("### 🔮 Future Data Needed")
        for f in MISSING_FIELDS: st.markdown(f"- {f}")


def main():
    if not st.session_state.logged_in:
        page_login()
        return
        
    with st.spinner("Loading MPLADS data and running core AI models…"):
        base_df, similar_map = run_core_models()
        
    # Dynamic Risk Score Computation based on admin tuning
    # MP/Public won't see these, but they need to exist for the whole app logic
    df = compute_risk_scores(
        base_df, 
        weight_cost=st.session_state.w_cost, 
        weight_duplicate=st.session_state.w_dup,
        weight_delay=st.session_state.w_del,
        weight_rules=st.session_state.w_pat
    )
        
    page, fdf, filters = sidebar_nav(df)
    
    role = st.session_state.role
    if page not in ROLE_PAGES.get(role, []):
        st.error("🚨 Access Denied: Role privileges insufficient for this view.")
        st.stop()
        
    # STRICT DATA LEAK PREVENTION
    if role in ["Member of Parliament", "Public Viewer"]:
        # Strip away all ML risk flags, reasons, and scores entirely so they never enter memory for this render
        safe_cols = [c for c in SAFE_COLUMNS if c in fdf.columns]
        fdf = fdf[safe_cols].copy()
    
    # Apply user-selected filters
    filtered_fdf = apply_filters(fdf, filters)
    
    if len(filtered_fdf) == 0 and page not in ["⚙️ System Tuning & Audits", "💬 Citizen Feedback", "ℹ️ About & Methodology", "🏛️ My Constituency", "🌐 Transparency Portal"]:
        st.warning("No records match current filters. Adjust the sidebar filters.")
        return
        
    if page == "📊 Dashboard": page_dashboard(df, filtered_fdf)
    elif page == "⚙️ System Tuning & Audits": page_mospi_admin_settings(df, filtered_fdf)
    elif page == "🏛️ My Constituency": page_mp_dashboard(filtered_fdf)
    elif page == "🔴 High-Risk Works": page_high_risk(df, filtered_fdf)
    elif page == "🔍 Project Detail": page_project_detail(filtered_fdf, similar_map) # Project detail uses filtered data to restrict access
    elif page == "📈 Risk Analysis": page_risk_analysis(df, filtered_fdf)
    elif page == "📋 Review Workflow": page_review_workflow(df, filtered_fdf, similar_map)
    elif page == "📄 Investigation Report": page_investigation_report(filtered_fdf, similar_map)
    elif page == "🔒 Audit Trail": page_audit_trail()
    elif page == "💬 Citizen Feedback": page_citizen_feedback(fdf) # Pass unfiltered by selection, but role restricted df
    elif page == "🌐 Transparency Portal": page_transparency_portal(df) # Public sees all
    elif page == "ℹ️ About & Methodology": page_about()


if __name__ == "__main__":
    main()
