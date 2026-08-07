"""
Regression tests for the ranker's target-leakage and train/serve-skew bugs.

The original train_once.py set `p.semantic_score = p.score` before training -
i.e. one of the training features was set equal to the label itself. A
LinearRegression handed that trivially "solves" the problem by learning a
coefficient of ~1 on that column, learning nothing real about the other
features. These tests assert the fixed pipeline can't regress back into that
shape.
"""
from models import Paper
from features import compute_features, ranker_feature_vector, RANKER_FEATURE_KEYS
from train_data import build_training_examples


def _paper(id_, year=2023, citations=10, abstract="This model improves evaluation performance on the dataset benchmark."):
    return Paper(id=id_, title=f"Paper {id_}", abstract=abstract, year=year, citation_count=citations)


def test_ranker_features_exclude_semantic():
    # semantic is query-dependent and must never be a fixed training feature -
    # baking it in would mean the ranker is implicitly tied to one query.
    assert "semantic" not in RANKER_FEATURE_KEYS


def test_training_features_never_equal_the_label():
    # Varied citation counts (including a repeated one) and years, so no
    # feature column trivially degenerates to a constant that could
    # coincidentally match a single label value by chance - the real leakage
    # signature is a whole feature *column* being an exact copy of the label
    # column, not a single scalar collision.
    papers = [
        _paper(f"p{i}", year=2018 + (i % 7), citations=(i * 3) % 11)
        for i in range(15)
    ]

    X, y, _ = build_training_examples(papers)

    num_features = len(X[0])
    for col in range(num_features):
        column = [row[col] for row in X]
        assert column != y, (
            f"Feature column {col} is an exact copy of the label column - this "
            "is the leakage bug: a feature must never be a copy of what the "
            "model is trying to predict."
        )


def test_train_and_serve_use_identical_feature_keys():
    """
    train_data.py and scoring.py must build the ranker's feature vector via
    the same RANKER_FEATURE_KEYS ordering - the old bug was that train_data.py
    and scoring.score_paper() each hand-built a features list with different
    column meanings, so a model trained on one ordering was served completely
    different values at inference.
    """
    paper = _paper("p1")

    features_at_train_time = compute_features(paper, semantic=0.0)
    features_at_serve_time = compute_features(paper, semantic=0.42)

    train_vec = ranker_feature_vector(features_at_train_time)
    serve_vec = ranker_feature_vector(features_at_serve_time)

    # Only "semantic" should differ between the two calls (and semantic isn't
    # even part of the ranker's feature vector) - recency/depth/coverage must
    # be computed identically regardless of which caller invoked them.
    assert train_vec == serve_vec


def test_build_training_examples_handles_empty_input():
    X, y, y_range = build_training_examples([])
    assert X == []
    assert y == []
    assert y_range == (0.0, 1.0)
