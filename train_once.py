"""
Offline bootstrap for the paper-ranking model. Run this once (or whenever you
want to retrain): `python train_once.py`. Produces ranker.pkl, which
scoring.py loads automatically if present.
"""
import logging
import sys

from train_data import build_training_examples
from train_ranker import train_ranker
from retrieval import search_arxiv
from citations import enrich_papers_with_citations

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 10

# Deliberately spans multiple unrelated domains, not just one topic, so the
# ranker learns a general recency/citation-graph/concept-coverage -> impact
# relationship instead of overfitting to one subject area's citation and
# publication-age distribution.
TRAINING_QUERIES = [
    "large language model hallucination mitigation",
    "parameter efficient fine tuning transformers",
    "autonomous vehicle perception and planning",
    "reinforcement learning robotic manipulation",
    "climate change impact modeling",
]


def collect_training_papers():
    papers = []
    seen = set()
    for q in TRAINING_QUERIES:
        for p in search_arxiv(q, max_results=20):
            if p.id not in seen:
                seen.add(p.id)
                papers.append(p)
    return papers


def main():
    logger.info("Collecting training papers across %d queries...", len(TRAINING_QUERIES))
    papers = collect_training_papers()
    logger.info("Collected %d unique papers", len(papers))

    logger.info("Fetching citation data + building citation graph edges...")
    enriched_ids = enrich_papers_with_citations(papers, known_ids={p.id for p in papers})

    # Train only on papers whose citation lookup actually succeeded - a
    # failed/rate-limited lookup reports citation_count=0 by default, which
    # is indistinguishable from a real zero unless filtered out here. Training
    # on unfiltered papers previously produced a completely degenerate label
    # range (every paper's citation_count silently defaulted to 0 whenever
    # Semantic Scholar's rate limit was contended), which would make the
    # ranker learn nothing but "always predict ~0".
    enriched_papers = [p for p in papers if p.id in enriched_ids]
    logger.info("Citation data fetched for %d/%d papers", len(enriched_papers), len(papers))

    X, y, y_range = build_training_examples(enriched_papers)
    logger.info("Training samples: %d (raw log1p(citations) range: %s)", len(X), y_range)

    if len(X) < MIN_TRAINING_SAMPLES:
        # Refusing to call model.fit() on too few (or zero) samples rather
        # than letting sklearn crash with a raw "Expected 2D array, got 1D
        # array" traceback - that's what happened here the first time this
        # ran into a fully rate-limited Semantic Scholar (0/100 papers
        # enriched, so X and y were both empty lists).
        logger.error(
            "Only %d training samples with real citation data (need >= %d). "
            "This usually means Semantic Scholar's API rate-limited every "
            "lookup in this run - try again later, or from a connection that "
            "isn't sharing an already-throttled IP. Not writing ranker.pkl.",
            len(X), MIN_TRAINING_SAMPLES,
        )
        sys.exit(1)

    train_ranker(X, y, y_range)
    logger.info("Ranker trained and saved to ranker.pkl")


if __name__ == "__main__":
    main()
