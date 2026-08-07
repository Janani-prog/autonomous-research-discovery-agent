from models import ResearchState, Phase, Subgoal
from planner import generate_subgoals
from retrieval import search_arxiv
from scoring import score_papers
from coverage import detect_concept_gaps
from refine import refine_queries
from inquiry import maybe_ask_question
from concept_graph import build_concept_paper_map
from contradictions import detect_contradictions
from citations import enrich_papers_with_citations
from cross_analysis import SAFETY_KEYWORDS


def run_agent(objective: str):
    state = ResearchState(objective=objective)

    while state.phase != Phase.TERMINATE:
        print(f"\n--- Phase: {state.phase} ---")

        # =====================
        # 1. PLAN
        # =====================
        if state.phase == Phase.PLAN:
            subgoals_raw = generate_subgoals(objective)

            for k, v in subgoals_raw.items():
                state.subgoals[k] = Subgoal(
                    name=v["name"],
                    description=v["description"]
                )

            state.phase = Phase.SEARCH

        # =====================
        # 2. SEARCH (retrieval only)
        # =====================
        elif state.phase == Phase.SEARCH:
            for sg in state.subgoals.values():
                if sg.completed:
                    continue

                # A subgoal that's already been searched at least once, and
                # has no freshly-computed refined_queries waiting from the
                # last ANALYZE pass, has nothing new to search for this round
                # (it's either exhausted its refinement budget, or its last
                # round's queries were already consumed below).
                if sg.papers and sg.refined_queries is None:
                    continue

                queries = sg.refined_queries or [
                f"{sg.name.replace('_', ' ')} {objective}"
                ]

                for q in queries:
                    try:
                        papers = search_arxiv(q)
                        for p in papers:
                            sg.papers[p.id] = p
                    except Exception:
                        continue

                sg.refined_queries = None  # consumed - don't repeat next round

            state.phase = Phase.SCORE

        # =====================
        # 3. SCORE (ranking + enrichment)
        # =====================
        elif state.phase == Phase.SCORE:
            for sg in state.subgoals.values():
                papers = list(sg.papers.values())
                if not papers:
                    continue

                to_enrich = [p for p in papers if not p.citation_checked]
                if to_enrich:
                    try:
                        known_ids = {p.id for p in papers}
                        enriched_ids = enrich_papers_with_citations(to_enrich, known_ids)
                    except Exception:
                        enriched_ids = set()
                    # Only mark successfully-enriched papers as checked - a
                    # rate-limited/failed lookup should stay eligible for a
                    # retry on a later pass rather than being permanently
                    # recorded as "checked, zero citations" from a failure.
                    for p in to_enrich:
                        if p.id in enriched_ids:
                            p.citation_checked = True

                score_papers(papers, sg.description)

            state.phase = Phase.ANALYZE

        # =====================
        # 4. ANALYZE
        # =====================
        elif state.phase == Phase.ANALYZE:
            unresolved = 0
            queued_for_refinement = False

            for sg in state.subgoals.values():
                if not sg.papers:
                    unresolved += 1
                    continue

                ranked = sorted(
                    sg.papers.values(),
                    key=lambda p: p.score,
                    reverse=True
                )

                sg.concept_map = build_concept_paper_map(ranked)
                sg.gaps = detect_concept_gaps(sg.concept_map)

                # Matched by keyword rather than exact name "safety" - subgoal
                # names are now LLM-generated per objective (see planner.py),
                # so they won't reliably be the literal string "safety".
                is_safety_subgoal = any(
                    k in sg.name.lower() or k in sg.description.lower() for k in SAFETY_KEYWORDS
                )
                if is_safety_subgoal:
                    contradictions = detect_contradictions(sg.papers.values())
                    if contradictions:
                        sg.gaps.append("Conflicting safety claims detected")

                if not sg.gaps:
                    sg.completed = True
                    continue

                if sg.refinements < 2:
                    sg.refined_queries = refine_queries(
                        sg.description,
                        sg.gaps
                    )
                    sg.refinements += 1
                    unresolved += 1
                    queued_for_refinement = True
                else:
                    unresolved += 1

            if unresolved == 0:
                state.phase = Phase.TERMINATE
            elif queued_for_refinement:
                # At least one subgoal just got fresh refined_queries and
                # still has refinement budget left - actually go run another
                # SEARCH round with them. Without this branch, refined_queries
                # would be computed here and then simply discarded, since the
                # only other options are INQUIRE/TERMINATE, neither of which
                # ever route back to SEARCH - which is exactly what the
                # previous version of this function did: it advertised an
                # "iterative self-refinement loop" that could never actually
                # execute more than one search round.
                state.phase = Phase.SEARCH
            else:
                # Every remaining unresolved subgoal has exhausted its
                # refinement budget (refinements >= 2) - nothing left to
                # search for, so fall through to inquiry/termination.
                state.confidence += 0.2
                if state.confidence < 0.6:
                    state.phase = Phase.INQUIRE
                else:
                    state.phase = Phase.TERMINATE

        # =====================
        # 5. INQUIRE
        # =====================
        elif state.phase == Phase.INQUIRE:
            state.inquiry = maybe_ask_question(state)
            state.phase = Phase.TERMINATE

    return state