def refine_queries(base: str, gaps: list[str]):
    strategies = [
        "survey",
        "benchmark",
        "methods",
        "challenges",
        "verification"
    ]

    refined = []
    for g in gaps[:2]:
        concept = g.split(":")[-1].strip()
        for s in strategies[:2]:
            refined.append(f"{base} {concept} {s}")

    return refined
