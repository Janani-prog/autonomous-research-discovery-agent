from models import Paper
from concept_graph import build_concept_paper_map
from coverage import detect_concept_gaps


def _paper(id_, abstract):
    return Paper(id=id_, title=f"Title {id_}", abstract=abstract, year=2023)


def test_build_concept_paper_map_runs_on_small_batch():
    papers = [
        _paper("a", "This dataset benchmark evaluates model performance on classification method accuracy."),
        _paper("b", "A new method improves evaluation performance on the same dataset benchmark."),
    ]
    concept_map = build_concept_paper_map(papers, top_k=10)

    assert isinstance(concept_map, dict)
    assert all(isinstance(ids, list) for ids in concept_map.values())


def test_build_concept_paper_map_handles_empty_input():
    assert build_concept_paper_map([]) == {}


def test_detect_concept_gaps_flags_missing_expected_concepts():
    gaps = detect_concept_gaps({"unrelated_term": ["a"]})
    assert any("dataset" in g for g in gaps)


def test_detect_concept_gaps_empty_when_all_present():
    concept_map = {
        "model": ["a"], "dataset": ["a"], "evaluation": ["a"],
        "method": ["a"], "performance": ["a"],
    }
    assert detect_concept_gaps(concept_map) == []
