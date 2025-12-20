from claims import extract_claims


def detect_contradictions(papers):
    contradictions = []

    claims = []
    for p in papers:
        for c in extract_claims(p.abstract):
            claims.append((p.id, c.lower()))

    for i in range(len(claims)):
        id_a, c1 = claims[i]
        for j in range(i + 1, len(claims)):
            id_b, c2 = claims[j]

            if "improves" in c1 and "fails" in c2:
                contradictions.append((id_a, id_b, c1, c2))

    return contradictions
