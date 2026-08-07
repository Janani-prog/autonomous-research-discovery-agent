from contradictions import detect_contradictions


class FakePaper:
    def __init__(self, id_, abstract):
        self.id = id_
        self.abstract = abstract


def test_unrelated_claims_are_not_flagged():
    # Same polarity-conflict keywords ("improves" / "fails"), but about
    # completely unrelated subjects - the old version flagged this purely
    # from keyword co-occurrence with no subject-overlap check at all.
    papers = [
        FakePaper("a", "Our pruning method improves inference latency on vision transformers."),
        FakePaper("b", "The retrieval augmented pipeline fails on multi-hop question answering."),
    ]
    assert detect_contradictions(papers) == []


def test_genuinely_conflicting_claims_about_the_same_subject_are_flagged():
    papers = [
        FakePaper("a", "Chain of thought prompting improves reasoning accuracy on arithmetic benchmarks."),
        FakePaper("b", "We show chain of thought prompting fails to improve reasoning accuracy on arithmetic benchmarks."),
    ]
    assert len(detect_contradictions(papers)) >= 1
