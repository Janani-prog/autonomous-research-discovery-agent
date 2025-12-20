from agent import run_agent
from cross_analysis import analyze_cross_subgoal_gaps
from export import export_markdown
from metrics import compute_metrics

state = run_agent("hallucination mitigation in large language models")

print("\n===== FINAL REPORT =====")

for name, sg in state.subgoals.items():
    print(f"\n--- Subgoal: {name.upper()} ---")
    print("Description:", sg.description)
    print("Completed:", sg.completed)

    print("\nTop Papers:")
    for p in sorted(sg.papers.values(), key=lambda p: p.score, reverse=True)[:3]:
        print(f"- {p.year} | {p.title}")

    if sg.gaps:
        print("\nRemaining Gaps:")
        for g in sg.gaps:
            print("-", g)
print("\nConcept Coverage:")
for concept, papers in list(sg.concept_map.items())[:5]:
    print(f"- {concept}: {len(papers)} papers")


print("\n=== CROSS-SUBGOAL INSIGHTS ===")
cross_gaps = analyze_cross_subgoal_gaps(state.subgoals)

for g in cross_gaps:
    print("-", g)

export_markdown(state)
print("\n📄 Report written to report.md")

print("\n=== AGENT METRICS ===")
metrics = compute_metrics(state)
for k, v in metrics.items():
    print(f"{k}: {v}")
