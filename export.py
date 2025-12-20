def export_markdown(state, path="report.md"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"## Objective\n{state.objective}\n\n")

        for name, sg in state.subgoals.items():
            f.write(f"## {name.upper()}\n")
            f.write(f"{sg.description}\n\n")

            f.write("### Top Papers\n")
            for p in sorted(sg.papers.values(), key=lambda p: p.score, reverse=True)[:3]:
                line = f"- {p.year} | "
                if p.url:
                    line += f"[{p.title}]({p.url})"
                else:
                    line += p.title

                extras = []

                if getattr(p, "pdf_url", None):
                    extras.append(f"[PDF]({p.pdf_url})")

                if getattr(p, "citation_count", 0) > 0:
                    extras.append(f"Citations: {p.citation_count}")

                if extras:
                    line += " — " + " | ".join(extras)

                f.write(line + "\n")

            if sg.gaps:
                f.write("\n### Gaps\n")
                for g in sg.gaps:
                    f.write(f"- {g}\n")

            f.write("\n")
