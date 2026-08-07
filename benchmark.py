"""
Real, reproducible benchmark comparing ARDA's autonomous multi-subgoal,
self-refining search against a simulated single-query "manual" baseline.

This replaces the old README claims of "~70% reduction in manual search
effort" and "~4x increase in topical coverage" - numbers that appeared in the
README/resume with no computation anywhere in the codebase behind them. This
script actually runs both approaches and reports whatever the real numbers
come out to.

METHODOLOGY (documented here so results are falsifiable/reproducible, not
just asserted):

For each objective in EVAL_OBJECTIVES (a fixed, diverse list chosen upfront,
covering multiple unrelated domains - not cherry-picked after seeing results):

  1. AGENT RUN: run_agent(objective) end-to-end, exactly as a real user would
     invoke it. Records agent_queries_issued (1 initial query per subgoal +
     1 more per refinement round actually used) and the concept SET (via
     concepts.extract_concepts) over the union of all retrieved papers.

  2. MANUAL BASELINE: a single arXiv search using the raw objective text as
     the only query, retrieving the SAME number of papers the agent ended up
     with. Equal paper budget isolates the effect of query decomposition +
     refinement specifically, rather than "the agent fetched more papers so
     of course it found more concepts." Same concept extractor, same set.

  3. COMPARE THE SETS, NOT JUST THEIR SIZES. An earlier version of this
     script compared len(agent_concepts) vs len(manual_concepts) - i.e. a
     "coverage ratio". That metric turned out to be structurally unable to
     show a real difference: extract_concepts has its own max_features=500
     ceiling, and any real 40-160+ paper corpus has more than enough
     vocabulary diversity to fill whatever top_k is requested - so both
     sides saturated at *exactly* the same count (first tried top_k=25: both
     sides came back with exactly 25 concepts on every run; raising it to
     top_k=150 just moved the tie to exactly 150 on every run instead).
     Comparing the actual SETS - intersection, and each side's
     concepts-the-other-side-missed-entirely - is sensitive to genuine
     differences in what the two approaches surface, regardless of whether
     the raw counts happen to match.

REPORTED METRICS (averaged across the objective set):
  - jaccard_similarity: |intersection| / |union| of the two concept sets.
    This is the real signal here. A value well below 1.0 means the two
    approaches surface meaningfully different literature, not just
    differently-sized samples of the same literature.
  - agent_only_concepts / manual_only_concepts: how many concepts each side
    surfaced that the other missed. IMPORTANT CAVEAT, found only by actually
    running this and checking the arithmetic: because both sides are
    matched to the same paper budget and both saturate the same
    max_features=500 vocabulary ceiling, |agent_concepts| always equals
    |manual_concepts| in practice - and basic set theory says that whenever
    |A| = |B|, |A\\B| always equals |B\\A| identically (both equal
    |A| - |intersection|). So these two numbers are *mathematically
    guaranteed to be equal* under this design, not a genuine "which side
    covers more" result - they're reported for transparency, but
    jaccard_similarity is the metric that's actually free to vary and is
    the one that means something here.
  - agent_queries_issued: how many distinct search-and-reformulate cycles the
    agent performs autonomously per run, vs. the single query a "manual"
    baseline uses. Honest analogue of the old "~70% manual effort reduction"
    claim, reframed as: how many query iterations does the agent automate
    that a human would otherwise have to think up and run themselves.

All numbers are computed from real calls to the live arXiv API - not
fabricated, not simulated with synthetic data.
"""
import json
import logging
import time
from datetime import datetime

from agent import run_agent
from retrieval import search_arxiv
from concepts import extract_concepts

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "benchmark_results.json"

# Deliberately spans unrelated domains so results aren't a fluke of one
# subject area's literature density or citation patterns. Chosen before
# running this script.
EVAL_OBJECTIVES = [
    "hallucination mitigation in large language models",
    "parameter efficient fine tuning for large language models",
    "autonomous vehicle perception under adverse weather",
    "reinforcement learning for robotic manipulation",
    "graph neural networks for molecular property prediction",
    "climate change impact on crop yields",
]

# Close to extract_concepts' own max_features=500 vocabulary ceiling -
# see the module docstring for why capping this low (25, then 150 in
# earlier attempts) made the whole comparison saturate and hide any real
# difference between the two approaches.
CONCEPT_TOP_K = 400


def _concept_set(papers, top_k: int = CONCEPT_TOP_K) -> set[str]:
    if not papers:
        return set()
    return set(extract_concepts(papers, top_k=top_k).keys())


def run_single_benchmark(objective: str) -> dict:
    state = run_agent(objective)

    all_papers = {}
    queries_issued = 0
    for sg in state.subgoals.values():
        all_papers.update(sg.papers)
        queries_issued += 1 + sg.refinements  # 1 initial query + 1 per refinement round used

    agent_papers = list(all_papers.values())
    agent_concepts = _concept_set(agent_papers)

    manual_papers = search_arxiv(objective, max_results=max(len(agent_papers), 1))
    manual_concepts = _concept_set(manual_papers)

    intersection = agent_concepts & manual_concepts
    union = agent_concepts | manual_concepts
    agent_only = agent_concepts - manual_concepts
    manual_only = manual_concepts - agent_concepts

    jaccard = round(len(intersection) / len(union), 3) if union else None

    return {
        "objective": objective,
        "agent_queries_issued": queries_issued,
        "agent_unique_papers": len(agent_papers),
        "manual_unique_papers": len(manual_papers),
        "agent_concepts_total": len(agent_concepts),
        "manual_concepts_total": len(manual_concepts),
        "shared_concepts": len(intersection),
        "agent_only_concepts": len(agent_only),
        "manual_only_concepts": len(manual_only),
        "jaccard_similarity": jaccard,
        "completion_ratio": round(
            sum(sg.completed for sg in state.subgoals.values()) / max(len(state.subgoals), 1), 3
        ),
    }


def run():
    results = []
    for i, objective in enumerate(EVAL_OBJECTIVES, 1):
        logger.info("[%d/%d] Running benchmark for: %s", i, len(EVAL_OBJECTIVES), objective)

        result = None
        for attempt in range(2):  # one retry - a transient DNS/connection
            # blip during a run this long (arXiv + Semantic Scholar + Groq,
            # dozens of HTTP calls) shouldn't cost the whole objective if it
            # clears up moments later, which is exactly what happened during
            # this benchmark's own initial run (a mid-run network outage
            # dropped 4 of 6 objectives that had nothing wrong with the
            # underlying agent logic).
            try:
                result = run_single_benchmark(objective)
                break
            except Exception as e:
                logger.warning("  attempt %d FAILED: %s", attempt + 1, e)
                if attempt == 0:
                    time.sleep(10)

        if result is None:
            continue

        logger.info(
            "  agent_only=%d  manual_only=%d  shared=%d  jaccard=%s  queries=%d",
            result["agent_only_concepts"], result["manual_only_concepts"],
            result["shared_concepts"], result["jaccard_similarity"], result["agent_queries_issued"],
        )
        results.append(result)
        time.sleep(2)  # breathing room between full agent runs (arXiv + Semantic Scholar + Groq)

    if not results:
        logger.warning("No successful benchmark runs - nothing to report.")
        return

    def _avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_objectives": len(results),
        "num_attempted": len(EVAL_OBJECTIVES),
        "avg_agent_only_concepts": _avg("agent_only_concepts"),
        "avg_manual_only_concepts": _avg("manual_only_concepts"),
        "avg_shared_concepts": _avg("shared_concepts"),
        "avg_jaccard_similarity": _avg("jaccard_similarity"),
        "avg_agent_queries_per_run": _avg("agent_queries_issued"),
        "per_objective": results,
    }

    logger.info("=" * 60)
    logger.info("  Objectives completed: %d/%d", len(results), len(EVAL_OBJECTIVES))
    logger.info("  Avg concepts only the agent found:   %s", summary["avg_agent_only_concepts"])
    logger.info("  Avg concepts only manual search found: %s", summary["avg_manual_only_concepts"])
    logger.info("  Avg shared concepts:                  %s", summary["avg_shared_concepts"])
    logger.info("  Avg Jaccard similarity:               %s", summary["avg_jaccard_similarity"])
    logger.info("  Avg autonomous queries issued per run: %s", summary["avg_agent_queries_per_run"])
    logger.info("=" * 60)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved to %s", RESULTS_PATH)

    return summary


if __name__ == "__main__":
    run()
