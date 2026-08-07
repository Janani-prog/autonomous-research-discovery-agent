from claims import extract_claims

_STOPWORDS = {
    "the", "and", "of", "to", "for", "with", "in", "on",
    "a", "an", "by", "is", "are", "this", "that", "we", "our",
    "these", "those", "can", "will", "using", "based",
}


def _content_words(sentence: str) -> set[str]:
    return {
        w.strip(".,()")
        for w in sentence.lower().split()
        if len(w) > 3 and w.strip(".,()") not in _STOPWORDS
    }


def detect_contradictions(papers, overlap_threshold: float = 0.25):
    """
    Flags a pair of claims as a possible contradiction only if:
      1. one has an "improves"/"outperforms"-type indicator and the other a
         "fails"/"cannot"-type indicator, AND
      2. the two claim sentences share enough content words (Jaccard overlap
         over words >3 chars, stopwords removed) to plausibly be about the
         same subject.

    The previous version only checked condition 1, across every claim from
    every paper - so e.g. "Method A improves accuracy on benchmark X" from
    one paper and "Method B fails under adversarial noise" from a completely
    unrelated paper would register as a "contradiction" purely from keyword
    co-occurrence, regardless of whether they're even discussing the same
    method or task. Condition 2 is a cheap, still fully offline guard against
    that - it doesn't guarantee semantic agreement, but it filters out the
    clearest false positives where the two claims share no real subject
    matter at all.
    """
    contradictions = []

    claims = []
    for p in papers:
        for c in extract_claims(p.abstract):
            claims.append((p.id, c))

    for i in range(len(claims)):
        id_a, c1 = claims[i]
        c1_l = c1.lower()
        for j in range(i + 1, len(claims)):
            id_b, c2 = claims[j]
            c2_l = c2.lower()

            polarity_conflict = (
                ("improves" in c1_l or "outperforms" in c1_l) and ("fails" in c2_l or "cannot" in c2_l)
            ) or (
                ("improves" in c2_l or "outperforms" in c2_l) and ("fails" in c1_l or "cannot" in c1_l)
            )
            if not polarity_conflict:
                continue

            words_a, words_b = _content_words(c1), _content_words(c2)
            if not words_a or not words_b:
                continue

            overlap = len(words_a & words_b) / len(words_a | words_b)
            if overlap >= overlap_threshold:
                contradictions.append((id_a, id_b, c1, c2))

    return contradictions
