from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def extract_concepts(papers, top_k=10):
    """
    Extract top concepts from paper abstracts using TF-IDF.
    Returns: dict {concept: count}
    """

    abstracts = [p.abstract for p in papers]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(abstracts)
    feature_names = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1
    concept_scores = list(zip(feature_names, scores))

    # Sort by importance
    concept_scores.sort(key=lambda x: x[1], reverse=True)

    concepts = defaultdict(int)
    for term, score in concept_scores[:top_k]:
        concepts[term] = int(score * 100)

    return dict(concepts)
