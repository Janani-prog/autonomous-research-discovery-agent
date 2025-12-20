from collections import defaultdict


def build_concept_paper_map(papers, top_k=15):
    """
    Build mapping: concept -> list of paper IDs that mention it.
    """
    concept_map = defaultdict(list)

    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()

        for concept in extract_simple_concepts(text, top_k):
            concept_map[concept].append(paper.id)

    return dict(concept_map)


def extract_simple_concepts(text: str, top_k=15):
    """
    Extremely simple noun-like concept extraction.
    Deterministic, no LLM.
    """
    stopwords = {
        "the", "and", "of", "to", "for", "with", "in", "on",
        "a", "an", "by", "is", "are", "this", "that"
    }

    words = [
        w.strip(".,()").lower()
        for w in text.split()
        if len(w) > 4 and w not in stopwords
    ]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]
