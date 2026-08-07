"""
Single source of truth for every feature used by paper scoring, at both
training time (train_data.py) and inference time (scoring.py).

The previous ranker had a train/serve skew bug: train_data.py built a feature
vector in one order/meaning, and scoring.py's score_paper() built a
differently-ordered, differently-meaning feature vector at inference. Routing
both through this module - and through RANKER_FEATURE_KEYS specifically - is
what prevents that class of bug from coming back.
"""
import math
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import Paper
from graph_score import citation_depth_score
from coverage import EXPECTED_CONCEPTS

# Features the learned ranker is trained/served on. Deliberately excludes
# "semantic" - that one is query-specific and computed fresh at serving time
# for whatever query is active, so it can't be a fixed training feature.
RANKER_FEATURE_KEYS = ["recency", "citation_depth", "concept_coverage"]


def batch_semantic_similarity(query: str, papers: list[Paper]) -> dict[str, float]:
    """
    Fits ONE TfidfVectorizer across [query] + every paper's text, then reads
    off cosine similarity of each paper to the query from that single fit.

    The old implementation called TfidfVectorizer().fit_transform([query, one_paper])
    separately per paper - refitting IDF statistics on a 2-document "corpus"
    every time, which carries essentially no real document-frequency signal.
    Fitting once across the whole candidate batch gives IDF weights that
    actually reflect which terms are distinctive within this batch.
    """
    if not papers:
        return {}

    texts = [query] + [f"{p.title} {p.abstract}" for p in papers]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)

    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    return {p.id: float(sim) for p, sim in zip(papers, sims)}


def recency_score(year: int, now_year: int | None = None) -> float:
    """Exponential decay for older papers. now_year is injectable for testing."""
    now_year = now_year or datetime.now().year
    age = max(1, now_year - year)
    return math.exp(-age / 5)


def concept_coverage_score(paper: Paper) -> float:
    """
    Fraction of the fixed expected-concept vocabulary (coverage.py) that
    appears as a substring in this paper's title+abstract. Real, deterministic
    per-paper signal - the old code used a hardcoded 0.5 constant for every
    paper regardless of content.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    hits = sum(1 for c in EXPECTED_CONCEPTS if c in text)
    return hits / len(EXPECTED_CONCEPTS)


def compute_features(paper: Paper, semantic: float = 0.0) -> dict:
    """
    The one function that builds a paper's feature dict. Both training
    (train_data.py) and inference (scoring.py) call this - never compute
    these values ad hoc elsewhere, or the train/serve skew bug comes back.
    """
    return {
        "semantic": semantic,
        "recency": recency_score(paper.year),
        "citation_depth": citation_depth_score(getattr(paper, "citation_edges", [])),
        "concept_coverage": concept_coverage_score(paper),
    }


def ranker_feature_vector(features: dict) -> list[float]:
    """Projects a features dict down to the fixed-order vector the ranker consumes."""
    return [features[k] for k in RANKER_FEATURE_KEYS]
