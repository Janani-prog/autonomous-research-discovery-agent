print("train_data.py LOADED")
def build_training_examples(papers):
    X = []
    y = []

    for p in papers:
        # Use safe defaults (bootstrap phase)
        semantic = getattr(p, "semantic_score", 0.0)
        recency = getattr(p, "recency_score", 0.0)
        citation = getattr(p, "citation_score", 0.0)
        coverage = getattr(p, "concept_coverage", 0.0)

        X.append([semantic, recency, citation, coverage])
        y.append(p.score)

    return X, y

