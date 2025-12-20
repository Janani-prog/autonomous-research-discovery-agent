def citation_depth_score(edges):
    """
    Simple proxy for influence:
    deeper + wider = higher score
    """
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)

    return min(len(nodes) / 50, 1.0)
