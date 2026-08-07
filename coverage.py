EXPECTED_CONCEPTS = [
    "model",
    "dataset",
    "evaluation",
    "method",
    "performance",
]


def detect_concept_gaps(concept_map: dict):
    gaps = []

    for e in EXPECTED_CONCEPTS:
        found = any(e in concept for concept in concept_map.keys())
        if not found:
            gaps.append(f"Concept underrepresented or missing: {e}")

    return gaps
