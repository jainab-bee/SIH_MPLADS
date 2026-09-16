# 🛡️ MPLADS Sentinel

**Explainable AI for Public Works Risk, Audit Prioritization & Early Warning**
_Smart India Hackathon 2026 — Problem Statement SIH26102_

---

## Problem

MPLADS (Members of Parliament Local Area Development Scheme) funds ₹5 crore/year per MP for local development works. With ~19,000+ sanctioned works annually, manually identifying which projects deserve priority review is infeasible.

**Existing system answers**: "What is happening?"
**MPLADS Sentinel answers**: "What needs attention, why, and what should be reviewed next?"

---

## Solution

MPLADS Sentinel is an AI-assisted intelligence layer that:
1. Detects unusually high-cost works using **Isolation Forest** per peer group
2. Finds semantically similar/potentially duplicate works using **sentence-transformers**
3. Identifies old pending works using **date and status analysis**
4. Fuses all signals into a transparent **0–100 Review Priority Score**
5. Explains every flag with evidence cards ("Why Flagged?")

> ⚠️ **This system does NOT determine fraud.** It identifies unusual patterns and prioritizes works for human review.

---

## Input Dataset

- **File**: `Works Sanctioned.csv`
- **Records**: ~19,001 sanctioned MPLADS works (2023–2026)
- **Columns**: Sr. No., Work category, Work (code), State, IDA (district), MP Name, Constituency, Work description, Recommended date, Sanction Date, Sanction Amount (₹), Work Status

---

## ML Techniques

| Detector | Algorithm | Max Score |
|----------|-----------|-----------|
| Cost Anomaly | Isolation Forest (scikit-learn) + RobustScaler | 40 pts |
| Duplicate/Similar Work | all-MiniLM-L6-v2 + cosine similarity | 30 pts |
| Delay / Old Pending | Statistical (days since sanction + status) | 30 pts |

**Risk Score = Cost (40%) + Similarity (30%) + Delay (30%)**

| Score | Risk Level |
|-------|-----------|
| 0–39 | 🟢 LOW |
| 40–69 | 🟡 MEDIUM |
| 70–100 | 🔴 HIGH |

---

## How to Run

```bash
# 1. Clone / navigate to project folder
cd mplads-sentinel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

> **Note**: First run computes sentence-transformer embeddings (~2 min for 19k records).
> Subsequent runs load from cache instantly.

---

## App Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPI cards, charts, state/district analysis |
| 🔴 High-Risk Works | Sortable table of flagged works |
| 🔍 Project Detail | Full explainability — "Why Flagged?" evidence cards |
| 📈 Risk Analysis | Score distributions, scatter plots, trend charts |
| ℹ️ About | Methodology, data availability, future roadmap |

---

## Limitations

- **No expenditure/payment data** in this CSV — cost analysis is sanction-amount only
- **No GPS coordinates** — geographic overlap detection not possible
- **No physical progress %** — delay detection uses dates and status only
- **No expected completion date** — uses 365-day MPLADS guideline as proxy
- Similarity detection flags description-level similarity, not confirmed duplication

---

## Future Scope

- Connect to live eSAKSHI API for real-time data
- Add expenditure analysis when payment data available
- Add GPS coordinates for spatial clustering
- Integrate geo-tagged photo verification
- Add contractor blacklist matching
- Incorporate CAG audit report validation feedback loop

---

_This is a prototype for Smart India Hackathon 2026. All findings are algorithmic signals for human review only._
