from collections import defaultdict

from concepts import extract_concepts


def build_concept_paper_map(papers, top_k=15):
    """
    Identifies the top-k most distinctive concepts (unigrams/bigrams) across
    this paper batch via TF-IDF (concepts.extract_concepts), then maps each
    identified concept back to the specific papers whose title+abstract
    actually contain it.

    Previously this ran its own simple word-length-and-frequency filter per
    paper independently (keep words >4 chars, drop an 18-word stopword list,
    rank by raw frequency) - no real term-importance weighting, and it
    duplicated a second, better TF-IDF n-gram extractor (concepts.py) that
    existed in the repo but was never actually imported anywhere. Using that
    one here means the stronger implementation is the one that's live, and
    there's no redundant dead module left behind.
    """
    if not papers:
        return {}

    concept_scores = extract_concepts(papers, top_k=top_k)  # {term: importance}
    concept_map = defaultdict(list)

    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        for concept in concept_scores:
            if concept in text:
                concept_map[concept].append(paper.id)

    return dict(concept_map)
