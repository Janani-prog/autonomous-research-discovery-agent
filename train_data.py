import math

from features import compute_features, ranker_feature_vector


def build_training_examples(papers):
    """
    Builds (X, y, y_range) for the impact ranker.

    X: query-independent features [recency, citation_depth, concept_coverage]
       (see features.RANKER_FEATURE_KEYS), built via the exact same
       compute_features()/ranker_feature_vector() functions scoring.py calls
       at inference time. Using the same function in both places - not just
       "the same formula copy-pasted twice" - is what prevents train/serve
       skew from creeping back in if one side changes later.

    y: log1p(citation_count), min-max normalized to [0, 1] across this batch.
       This is a weak-supervision "impact" signal, not a ground-truth
       relevance label (no such label exists here) - the model is learning
       "do recency / citation-graph-position / concept-coverage predict how
       cited a paper ends up being", which is a real, testable relationship.

       Citation count is used ONLY as the label - it is never one of the X
       columns. The previous version of this file set X[0] (semantic_score)
       equal to the y label itself before training, which is textbook target
       leakage: a linear model handed a copy of its own target trivially
       "solves" the problem by learning a ~1.0 coefficient on that column and
       ~0 everywhere else, learning nothing about the other features.
    """
    X = []
    raw_labels = []

    for p in papers:
        features = compute_features(p, semantic=0.0)  # semantic excluded - query-independent ranker
        X.append(ranker_feature_vector(features))
        raw_labels.append(math.log1p(max(p.citation_count, 0)))

    if not raw_labels:
        return [], [], (0.0, 1.0)

    y_min, y_max = min(raw_labels), max(raw_labels)
    span = max(y_max - y_min, 1e-6)
    y = [(v - y_min) / span for v in raw_labels]

    return X, y, (y_min, y_max)
