def citation_depth_score(edges):
    """
    Simple proxy for influence: more distinct connected papers = higher score.

    `edges` now comes from citations.enrich_papers_with_citations(), which
    builds real edges from each paper's actual Semantic Scholar reference
    list - previously nothing in the codebase ever populated Paper.citation_edges,
    so this function always received an empty list and always returned 0.

    The /50 normalization is calibrated for a typical subgoal batch (~20-25
    papers per query, occasionally more after refinement rounds) - it isn't
    meant to be a properly-scaled score against the full global citation graph.
    """
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)

    return min(len(nodes) / 50, 1.0)
