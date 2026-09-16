"""
MPLADS Sentinel — Duplicate / Similar Work Detector
=====================================================
Uses sentence-transformers embeddings + cosine similarity
to find semantically similar work descriptions.

Performance strategy for 19,000 records:
- Embeddings cached via st.cache_data (computed once)
- FAISS-style numpy batched dot-product (no faiss dependency needed)
- Only top-K candidates per project are stored
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle
import os

# ── Configuration ──────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD_HIGH = 0.88   # Likely duplicate
SIMILARITY_THRESHOLD_MED  = 0.75   # Potentially related

TOP_K = 5                          # Max duplicates to surface per project
MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, 384-dim, good quality

CACHE_PATH = Path(__file__).resolve().parent.parent / "assets" / "embeddings.pkl"

# ── Text construction ──────────────────────────────────────────────────────

def _build_text(row: pd.Series) -> str:
    """Combine fields into a single text for embedding."""
    parts = []
    if pd.notna(row.get("work_description")) and str(row["work_description"]).strip():
        parts.append(str(row["work_description"]).strip())
    if pd.notna(row.get("sub_category")) and str(row["sub_category"]).strip():
        parts.append(str(row["sub_category"]).strip())
    if pd.notna(row.get("state")) and str(row["state"]).strip():
        parts.append(str(row["state"]).strip())
    return " | ".join(parts) if parts else "unknown work"


# ── Embedding computation ─────────────────────────────────────────────────

def _load_or_compute_embeddings(texts: list[str]) -> np.ndarray:
    """
    Load cached embeddings if they exist, otherwise compute and save.
    Returns normalised float32 array of shape (N, 384).
    """
    # Check cache
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("n") == len(texts):
            return cached["embeddings"]

    # Compute embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            texts,
            batch_size=256,
            normalize_embeddings=True,   # L2 normalised → cosine = dot product
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
    except ImportError:
        st.warning(
            "⚠️ sentence-transformers not installed. "
            "Using TF-IDF fallback for similarity."
        )
        embeddings = _tfidf_fallback(texts)

    # Save to cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"n": len(texts), "embeddings": embeddings}, f)

    return embeddings


def _tfidf_fallback(texts: list[str]) -> np.ndarray:
    """TF-IDF based embeddings as fallback if sentence-transformers unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return normalize(X)


# ── Efficient similarity search ────────────────────────────────────────────

def _batched_top_k_similar(
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    batch_size: int = 500,
) -> dict[int, list[tuple[int, float]]]:
    """
    For each row i, find top_k most similar rows (excluding self).
    Returns dict: { row_index → [(other_index, similarity_score), ...] }
    
    Uses batched matrix multiplication — no FAISS required.
    """
    N = embeddings.shape[0]
    results = {i: [] for i in range(N)}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = embeddings[start:end]                    # (batch, dim)
        # Cosine similarity: (batch × N) since embeddings are normalised
        sims = batch @ embeddings.T                      # (batch, N)

        for local_i, global_i in enumerate(range(start, end)):
            row_sims = sims[local_i].copy()
            row_sims[global_i] = -1.0                    # exclude self

            # Only keep those above medium threshold
            above = np.where(row_sims >= SIMILARITY_THRESHOLD_MED)[0]
            if len(above) == 0:
                continue

            # Sort descending and take top_k
            top_idxs = above[np.argsort(row_sims[above])[::-1][:top_k]]
            results[global_i] = [
                (int(idx), float(row_sims[idx])) for idx in top_idxs
            ]

    return results


# ── Main entry point ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing semantic similarity (first run only)…", ttl=3600)
def detect_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      1. df with columns: dup_score, dup_flag, dup_reason, best_match_idx, best_match_sim
      2. similar_map: dict { row_id → [(other_row_id, similarity), ...] }
    """
    texts = df.apply(_build_text, axis=1).tolist()
    embeddings = _load_or_compute_embeddings(texts)

    similar_map_raw = _batched_top_k_similar(embeddings, top_k=TOP_K)

    # Build output columns
    result_df = df.copy()
    result_df["dup_score"] = 0.0
    result_df["dup_flag"] = False
    result_df["dup_reason"] = "No highly similar work found"
    result_df["best_match_idx"] = -1
    result_df["best_match_sim"] = 0.0

    # Convert to row_id-keyed map for external use
    similar_map: dict[int, list[tuple[int, float]]] = {}

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
                f"(similarity {best_sim*100:.0f}%): \"{str(other_desc)[:80]}…\""
            )
        elif best_sim >= SIMILARITY_THRESHOLD_MED:
            result_df.at[i, "dup_reason"] = (
                f"Potentially related to Work #{other_sr} "
                f"(similarity {best_sim*100:.0f}%)"
            )

    # Score as 0–100 for risk fusion
    result_df["dup_score_pct"] = (result_df["dup_score"] * 100).round(1)

    return result_df, similar_map
