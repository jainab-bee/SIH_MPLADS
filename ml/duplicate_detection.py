import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle

SIMILARITY_THRESHOLD_HIGH = 0.88
SIMILARITY_THRESHOLD_MED  = 0.75
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = Path(__file__).resolve().parent.parent / "assets" / "embeddings.pkl"


def _build_text(row):
    parts = []
    if pd.notna(row.get("work_description")) and str(row["work_description"]).strip():
        parts.append(str(row["work_description"]).strip())
    if pd.notna(row.get("sub_category")) and str(row["sub_category"]).strip():
        parts.append(str(row["sub_category"]).strip())
    if pd.notna(row.get("state")) and str(row["state"]).strip():
        parts.append(str(row["state"]).strip())
    return " | ".join(parts) if parts else "unknown work"


def _load_or_compute_embeddings(texts):
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("n") == len(texts):
            return cached["embeddings"]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
    except ImportError:
        st.warning("\u26a0\ufe0f sentence-transformers not installed. Using TF-IDF fallback.")
        embeddings = _tfidf_fallback(texts)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"n": len(texts), "embeddings": embeddings}, f)
    return embeddings


def _tfidf_fallback(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return normalize(X)


def _batched_top_k_similar(embeddings, top_k=TOP_K, batch_size=500):
    N = embeddings.shape[0]
    results = {i: [] for i in range(N)}
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = embeddings[start:end]
        sims = batch @ embeddings.T
        for local_i, global_i in enumerate(range(start, end)):
            row_sims = sims[local_i].copy()
            row_sims[global_i] = -1.0
            above = np.where(row_sims >= SIMILARITY_THRESHOLD_MED)[0]
            if len(above) == 0:
                continue
            top_idxs = above[np.argsort(row_sims[above])[::-1][:top_k]]
            results[global_i] = [(int(idx), float(row_sims[idx])) for idx in top_idxs]
    return results


@st.cache_data(show_spinner="Computing semantic similarity (first run only)\u2026", ttl=3600)
def detect_duplicates(df):
    texts = df.apply(_build_text, axis=1).tolist()
    embeddings = _load_or_compute_embeddings(texts)
    similar_map_raw = _batched_top_k_similar(embeddings, top_k=TOP_K)
    result_df = df.copy()
    result_df["dup_score"] = 0.0
    result_df["dup_flag"] = False
    result_df["dup_reason"] = "No highly similar work found"
    result_df["best_match_idx"] = -1
    result_df["best_match_sim"] = 0.0
    similar_map = {}
    for i, matches in similar_map_raw.items():
        if not matches:
            continue
        best_idx, best_sim = matches[0]
        result_df.at[i, "dup_score"] = round(best_sim, 4)
        result_df.at[i, "best_match_idx"] = best_idx
        result_df.at[i, "best_match_sim"] = round(best_sim, 4)
        row_id = int(result_df.iloc[i]["row_id"])
        similar_map[row_id] = [
            (int(result_df.iloc[m_idx]["row_id"]), round(m_sim, 4))
            for m_idx, m_sim in matches
        ]
        other_desc = df.iloc[best_idx]["work_description"]
        other_sr = df.iloc[best_idx].get("sr_no", str(best_idx))
        if best_sim >= SIMILARITY_THRESHOLD_HIGH:
            result_df.at[i, "dup_flag"] = True
            result_df.at[i, "dup_reason"] = (
                f"Highly similar to Work #{other_sr} "
                f"(similarity {best_sim*100:.0f}%): \"{str(other_desc)[:80]}\u2026\""
            )
        elif best_sim >= SIMILARITY_THRESHOLD_MED:
            result_df.at[i, "dup_reason"] = (
                f"Potentially related to Work #{other_sr} (similarity {best_sim*100:.0f}%)"
            )
    result_df["dup_score_pct"] = (result_df["dup_score"] * 100).round(1)
    return result_df, similar_map
