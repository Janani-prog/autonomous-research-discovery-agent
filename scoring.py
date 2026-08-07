import os
import joblib

from models import Paper
from features import compute_features, ranker_feature_vector, batch_semantic_similarity, RANKER_FEATURE_KEYS

_ranker_bundle = None

if os.path.exists("ranker.pkl"):
    try:
        _ranker_bundle = joblib.load("ranker.pkl")
        valid = (
            isinstance(_ranker_bundle, dict)
            and "model" in _ranker_bundle
            and _ranker_bundle.get("feature_keys") == RANKER_FEATURE_KEYS
        )
        if not valid:
            # Stale format from before the leakage fix, or trained against a
            # different feature set - ignore it rather than risk scoring off
            # features it doesn't actually understand.
            _ranker_bundle = None
    except Exception:
        _ranker_bundle = None


def _predicted_impact(features: dict) -> float | None:
    """
    Uses the learned ranker to predict a query-independent "impact" prior
    (see train_data.py for how it's trained), normalized back to [0, 1] using
    the min/max the labels were scaled with at training time.
    """
    if _ranker_bundle is None:
        return None

    vec = [ranker_feature_vector(features)]
    raw = float(_ranker_bundle["model"].predict(vec)[0])

    y_min, y_max = _ranker_bundle["y_min"], _ranker_bundle["y_max"]
    span = max(y_max - y_min, 1e-6)
    normalized = (raw - y_min) / span
    return max(0.0, min(1.0, normalized))


def score_papers(papers: list[Paper], query: str) -> None:
    """
    Scores an entire batch of candidate papers for one subgoal/query in a
    single pass, mutating each paper's .score (and its feature fields) in
    place.

    TF-IDF is fit ONCE across the whole batch here (see
    features.batch_semantic_similarity) rather than per (query, single paper)
    pair - see that function's docstring for why the old per-pair approach
    made the "semantic" score nearly meaningless.

    If a trained ranker is available, its prediction (a learned, query-
    independent "impact" prior - see train_data.py) is blended with the
    real-time query-semantic-similarity, since the ranker itself never sees
    the current query. If no ranker is available, falls back to the
    heuristic weighted sum.
    """
    semantic_by_id = batch_semantic_similarity(query, papers)

    for paper in papers:
        semantic = semantic_by_id.get(paper.id, 0.0)
        features = compute_features(paper, semantic)

        impact = _predicted_impact(features)
        if impact is not None:
            paper.score = 0.6 * semantic + 0.4 * impact
        else:
            paper.score = (
                0.5 * features["semantic"]
                + 0.3 * features["recency"]
                + 0.2 * features["citation_depth"]
            )

        paper.semantic_score = features["semantic"]
        paper.recency_score = features["recency"]
        paper.citation_score = features["citation_depth"]
        paper.concept_coverage = features["concept_coverage"]


def score_paper(paper: Paper, query: str) -> float:
    """Single-paper convenience wrapper. Prefer score_papers() for batches -
    it shares one TF-IDF fit across the whole candidate set instead of
    fitting one per call."""
    score_papers([paper], query)
    return paper.score
