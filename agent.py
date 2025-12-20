from models import ResearchState, Phase, Subgoal
from planner import generate_subgoals
from retrieval import search_arxiv
from scoring import score_paper
from coverage import detect_concept_gaps
from refine import refine_queries
from inquiry import maybe_ask_question
from concept_graph import build_concept_paper_map
from contradictions import detect_contradictions
from citations import fetch_citation_count


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

            state.phase = Phase.SCORE

        # =====================
        # 3. SCORE (ranking + enrichment)
        # =====================
        elif state.phase == Phase.SCORE:
            for sg in state.subgoals.values():
                for p in sg.papers.values():
                    if p.citation_count == 0:
                        try:
                            p.citation_count = fetch_citation_count(p.id)
                        except Exception:
                            p.citation_count = 0

                    p.score = score_paper(p, sg.description)

            state.phase = Phase.ANALYZE

        # =====================
        # 4. ANALYZE
        # =====================
        elif state.phase == Phase.ANALYZE:
            unresolved = 0

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

                if sg.name.lower() == "safety":
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
                else:
                    unresolved += 1

            if unresolved == 0:
                state.phase = Phase.TERMINATE
            else:
                state.confidence += 0.2
                if state.confidence < 0.6:
                    state.phase = Phase.INQUIRE
                else:
                    state.phase = Phase.TERMINATE

        # =====================
        # 5. INQUIRE
        # =====================
        elif state.phase == Phase.INQUIRE:
            maybe_ask_question(state)
            state.phase = Phase.TERMINATE

    return state
