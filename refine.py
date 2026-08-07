# Caps how many refined queries one refinement round can fan out to. The
# uncapped version (gaps[:2] x strategies[:2] = up to 4 combinations) was
# harmless when it existed only on paper - agent.py's ANALYZE phase computed
# these queries but never actually routed back to SEARCH to run them (see
# agent.py's fixed refinement-loop bug). Once that loop was fixed to actually
# re-run refined queries, an uncapped 4-way fan-out per round, times up to 2
# refinement rounds, times every subgoal in a run, turns into a real
# combinatorial explosion of full arXiv + Semantic-Scholar-citation-enrichment
# searches. This cap keeps a refinement round's cost close to its original
# single-query intent while still trying more than one angle.
MAX_REFINED_QUERIES = 2


def refine_queries(base: str, gaps: list[str]) -> list[str]:
    strategies = [
        "survey",
        "benchmark",
        "methods",
        "challenges",
        "verification",
    ]

    refined = []
    for g in gaps:
        concept = g.split(":")[-1].strip()
        for s in strategies:
            refined.append(f"{base} {concept} {s}")
            if len(refined) >= MAX_REFINED_QUERIES:
                return refined

    return refined
