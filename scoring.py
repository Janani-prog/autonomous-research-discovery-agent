import math
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import Paper
from graph_score import citation_depth_score


# -----------------------------
# Lightweight TF-IDF Vectorizer
# -----------------------------
_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


def semantic_similarity(paper: Paper, query: str) -> float:
    """
    Computes TF-IDF cosine similarity between paper text and query.
    Lightweight and deployment-safe (no torch / transformers).
    """
    texts = [
        query,
        f"{paper.title} {paper.abstract}"
    ]

    tfidf = _vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return float(sim)


def recency_score(year: int) -> float:
    """
    Exponential decay for older papers.
    """
    age = max(1, 2025 - year)
    return math.exp(-age / 5)


def score_paper(paper: Paper, query: str) -> float:
    """
    Final scoring function combining:
    - semantic relevance (TF-IDF)
    - recency
    - citation depth
    - optional learned ranker
    """
    semantic = semantic_similarity(paper, query)
    recency = recency_score(paper.year)
    depth = citation_depth_score(getattr(paper, "citation_edges", []))

    # Default heuristic score
    heuristic = (
        0.5 * semantic +
        0.3 * recency +
        0.2 * depth
    )

    # Use learned ranker if available
    if ranker is not None:
        features = [[semantic, recency, depth, 0.5]]
        return float(ranker.predict(features)[0])

    return float(heuristic)


# -----------------------------
# Optional Learned Ranker
# -----------------------------
ranker = None

if os.path.exists("ranker.pkl"):
    try:
        ranker = joblib.load("ranker.pkl")
    except Exception:
        ranker = None


def learned_score(features):
    if ranker is None:
        return None
    return float(ranker.predict([features])[0])
