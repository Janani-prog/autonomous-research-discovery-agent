def detect_concept_gaps(concept_map: dict):
    expected = [
        "model",
        "dataset",
        "evaluation",
        "method",
        "performance"
    ]

    gaps = []

    for e in expected:
        found = any(e in concept for concept in concept_map.keys())
        if not found:
            gaps.append(f"Concept underrepresented or missing: {e}")

    return gaps
