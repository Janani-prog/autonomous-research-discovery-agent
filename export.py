def export_markdown(state, path="report.md"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"## Objective\n{state.objective}\n\n")

        for name, sg in state.subgoals.items():
            f.write(f"## {name.upper()}\n")
            f.write(f"{sg.description}\n\n")

            f.write("### Top Papers\n")
            for p in sorted(sg.papers.values(), key=lambda p: p.score, reverse=True)[:3]:
                f.write(f"- {p.year} — {p.title}\n")

            if sg.gaps:
                f.write("\n### Gaps\n")
                for g in sg.gaps:
                    f.write(f"- {g}\n")

            f.write("\n")
