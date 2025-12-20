import math
from sentence_transformers.util import cos_sim
from embeddings import model
from models import Paper
from graph_score import citation_depth_score
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(paper, query):
    """
    Computes TF-IDF cosine similarity between paper text and query.
    """
    texts = [
        query,
        f"{paper.title} {paper.abstract}"
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf = vectorizer.fit_transform(texts)

    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(sim)


def relevance_score(paper: Paper, goal: str) -> float:
    p = model.encode(paper.abstract)
    g = model.encode(goal)
    return float(cos_sim(p, g))


def recency_score(year: int) -> float:
    age = max(1, 2025 - year)
    return math.exp(-age / 5)


def score_paper(paper, query):
    semantic = semantic_similarity(paper, query)
    recency = recency_score(paper.year)
    depth = citation_depth_score(getattr(paper, "citation_edges", []))

    # Fallback heuristic score
    heuristic = (
        0.5 * semantic +
        0.3 * recency +
        0.2 * depth
    )

    # Use learned ranker only if available
    if ranker is not None:
        features = [[semantic, recency, depth, 0.5]]
        return ranker.predict(features)[0]

    return heuristic

ranker = None

if os.path.exists("ranker.pkl"):
    ranker = joblib.load("ranker.pkl")

def learned_score(features):
    return ranker.predict([features])[0]