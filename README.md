# 🛡️ MPLADS Sentinel

**Explainable AI for Public Works Risk, Audit Prioritization & Early Warning**  
*Smart India Hackathon 2026 — Problem Statement SIH26102*

---

## 📌 Overview

**MPLADS Sentinel** is an AI-assisted intelligence and audit-prioritization platform built on top of MPLADS (*Members of Parliament Local Area Development Scheme*) sanctioned works data.

- **Existing eSAKSHI / Portal answers:** *"What is happening?"*
- **MPLADS Sentinel answers:** *"What needs attention, why, and which works should be audited first?"*

> ⚠️ **Important:** This system does **NOT** autonomously declare fraud. It identifies unusual patterns, calculates multidimensional risk scores, provides explainable AI evidence, and prioritizes works for human-in-the-loop audit review.

---

## ✨ Key Features

| Module / Feature | Description |
|------------------|-------------|
| 📊 **Executive Summary** | High-level KPI cards, state/district filters, risk breakdowns & expenditure trends |
| 🏛️ **My Constituency** | MP/Public Representative portal with constituency performance & work progress tracking |
| 🏢 **Nodal Officer Portal** | District audit queue, local work assignments, and verification tracker |
| 🔴 **High-Risk Audit Prioritization** | AI-ranked work prioritization table with multidimensional anomaly filters |
| 🔍 **Work Deep Dive & Explainability** | Detailed "Why Flagged?" breakdown, score radar, cost comparisons & similar work matches |
| 📋 **Review & Escalation Workflow** | Human-in-the-loop actioning (Clear, Request Info, Flag for Audit, Escalate) |
| 📄 **Investigation Report** | Auto-generated, downloadable formal text/PDF audit dossier with tamper hash |
| 🔗 **Tamper-Evident Audit Trail** | SHA-256 cryptographic hash-chained immutable audit log of all system actions |
| 💬 **Citizen Feedback & Redressal** | Public grievance portal for reporting ground-level work observations with evidence upload |
| 🌐 **Public Transparency Portal** | Read-only open portal displaying anonymized statistics and completed works |
| 🔐 **Role-Based Access Control** | Secure authentication for Admin, Nodal Officer, MP/Representative, and Public Citizen |

---

## 🧠 Risk Scoring Engine & AI Architecture

The overall **Risk Score (0 – 100)** is computed using a 4-pillar Explainable AI framework:

| Detector Pillar | Algorithm / Technique | Max Weight |
|-----------------|----------------------|------------|
| 💰 **Cost Anomaly** | Isolation Forest (`scikit-learn`) + RobustScaler | **30 pts** |
| 👥 **Duplicate / Similar Work** | `all-MiniLM-L6-v2` Sentence Transformer + Cosine Similarity | **25 pts** |
| ⏱️ **Delay / Pending Work** | Statistical delay metrics (days elapsed since recommendation/sanction) | **20 pts** |
| ✂️ **Split Sanctioning / Clustering** | Pattern detection for work splitting below approval thresholds | **25 pts** |

### Risk Level Classification:
- 🔴 **HIGH RISK:** `Score >= 70`
- 🟡 **MEDIUM RISK:** `40 <= Score < 70`
- 🟢 **LOW RISK:** `Score < 40`

---

## 🔑 Demo Login Credentials

| Username | Password | Role | Features & Access |
|----------|----------|------|-------------------|
| `admin` | `admin123` | 🛡️ Admin | Full access to all modules & system settings |
| `nodal_officer` | `nodal123` | 🏢 Nodal Officer / Auditor | Nodal Portal, Audit Prioritization, Review Workflow & Reports |
| `mp_member` | `mp123` | 🏛️ MP / Representative | Constituency Dashboard & Recommended Works Tracker |
| `public` | `pub123` | 🌐 Citizen / Public | Citizen Feedback Portal & Public Transparency Portal |

---

## 📁 Project Structure

```
sih/
├── app.py                        ← Main Streamlit Web Application
├── requirements.txt              ← Python Package Dependencies
├── README.md                     ← Project Documentation
├── Works Sanctioned.csv          ← Dataset file (place in root directory)
├── sentinel.db                   ← SQLite Database (Audit Trail, Evidence, Users)
├── data/
│   ├── __init__.py
│   └── preprocessing.py          ← Data loading, cleaning, & metric calculations
├── ml/
│   ├── __init__.py
│   ├── cost_anomaly.py           ← Isolation Forest model for expenditure anomalies
│   ├── duplicate_detection.py    ← Semantic NLP model for similar work detection
│   ├── delay_detection.py        ← Statistical delay & pending duration metrics
│   └── risk_score.py             ← Multidimensional risk score fusion (0–100)
├── db/
│   ├── __init__.py
│   └── database.py               ← SQLite schema, cryptographic hash chain & user auth
├── utils/
│   ├── __init__.py
│   └── helpers.py                ← Formatting & UI helper functions
└── assets/
    └── embeddings.pkl            ← Cached embeddings (auto-generated on first run)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python `3.10` or higher installed
- Git installed

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/mplads-sentinel.git
cd mplads-sentinel
```

### 2️⃣ Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Dataset Setup
Place your dataset file `Works Sanctioned.csv` inside the project root folder.

> ℹ️ **Dataset Note:** If the CSV file is missing, the application automatically generates realistic fallback sample data so you can test all features immediately.

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

The application will start locally at:  
👉 **`http://localhost:8501`**

---

## ⚡ First Run Notes

- On the **very first launch**, sentence-transformer embeddings are calculated for work descriptions. This takes about **1–2 minutes**.
- Once calculated, embeddings are automatically cached to `assets/embeddings.pkl` for instant future launches.

---

## 🔒 Cryptographic Audit Chain

Every review decision, status update, evidence upload, or citizen report generates a **SHA-256 cryptographic hash block** stored sequentially in `sentinel.db`.  
- Each record links back to the previous record's hash (`previous_hash`).
- Built-in **Chain Integrity Verification** guarantees tamper detection for audit evidence.

---

## 📄 License & Disclaimer

This project is developed as a prototype for **Smart India Hackathon 2026** under Problem Statement `SIH26102`.  
All AI risk classifications and flags serve as decision support for human auditors and should be officially verified prior to taking legal or administrative action.
