"""
Regression test for the most significant bug found in this rebuild: the
ANALYZE phase computed refined_queries and incremented sg.refinements
whenever a subgoal had gaps, but the phase-transition logic only ever chose
between INQUIRE and TERMINATE next - no code path routed back to SEARCH. The
"iterative self-refinement loop" central to this project's whole premise
could never actually execute more than one search round; refined queries
were computed and silently discarded every time.

This test proves the fix without hitting any real API: it fakes
search_arxiv to always return a paper missing every expected concept, so
ANALYZE keeps detecting gaps and keeps queuing a refinement every round -
then asserts search_arxiv was actually called multiple times (proving the
loop re-enters SEARCH) and that it stops after exactly 2 refinement rounds
(proving the cap in agent.py's `if sg.refinements < 2` is still respected,
i.e. this isn't now an infinite loop).
"""
import agent
from models import Paper


def test_analyze_routes_back_to_search_when_refinement_is_queued(monkeypatch):
    call_count = {"n": 0}

    def fake_search_arxiv(query, max_results=25):
        call_count["n"] += 1
        # Deliberately missing every expected concept
        # (model/dataset/evaluation/method/performance) on every round, so
        # ANALYZE keeps detecting gaps and keeps queuing refinements up to
        # its 2-round cap - this isolates the thing this test actually
        # checks (does the loop re-enter SEARCH at all, and does it stop)
        # from the separate, unrelated question of exactly which concepts a
        # tiny synthetic TF-IDF corpus happens to rank into its top-k.
        return [Paper(id=f"p{call_count['n']}", title=f"T{call_count['n']}", abstract="irrelevant filler text about nothing notable", year=2023)]

    def fake_generate_subgoals(objective):
        return {"TESTGOAL": {"name": "testgoal", "description": f"Test subgoal for {objective}"}}

    def fake_enrich(papers, known_ids):
        # Must match enrich_papers_with_citations' real contract: return the
        # set of successfully-enriched ids, so agent.py's SCORE phase marks
        # them citation_checked (see agent.py + citations.py).
        for p in papers:
            p.citation_count = 0
            p.citation_edges = []
        return {p.id for p in papers}

    monkeypatch.setattr(agent, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(agent, "generate_subgoals", fake_generate_subgoals)
    monkeypatch.setattr(agent, "enrich_papers_with_citations", fake_enrich)

    state = agent.run_agent("test objective")

    # 1 initial query, then up to 4 refined queries per refinement round
    # (refine_queries fans one gap-detection out to gaps[:2] x strategies[:2]),
    # capped at 2 refinement rounds by agent.py's `if sg.refinements < 2` -
    # i.e. 1 + 4 + 4 = 9 with this test's fixed 5-gap scenario. The exact
    # count is a product of refine_queries' fan-out (not this test's
    # concern); what matters is that it's well past 1 (proving ANALYZE
    # actually routed back to SEARCH) and finite (proving the 2-round cap
    # was still respected, i.e. this isn't now an infinite loop).
    assert call_count["n"] > 1, (
        "search_arxiv was only called once - ANALYZE detected a gap and queued "
        "a refinement, but the state machine never routed back to SEARCH to "
        "actually run it. This is the refinement-loop bug this test guards."
    )

    testgoal = state.subgoals["TESTGOAL"]
    assert testgoal.completed is False  # gaps never resolved by design in this test
    assert testgoal.refinements == 2  # hit its cap, not left uncapped or unbounded
    assert len(testgoal.papers) == call_count["n"]  # one new paper accumulated per query issued
