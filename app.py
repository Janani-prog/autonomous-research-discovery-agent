import gradio as gr
from agent import run_agent

def run(objective):
    state = run_agent(objective)

    output = []
    output.append(f"Objective: {state.objective}\n")
    output.append(f"Confidence: {round(state.confidence, 2)}\n")

    for name, sg in state.subgoals.items():
        output.append(f"\n## {name}")
        output.append(sg.description.strip())
        output.append(f"Completed: {sg.completed}")

        if sg.gaps:
            output.append("Gaps:")
            for g in sg.gaps:
                output.append(f"- {g}")

        output.append("Top Papers:")
        for p in sorted(sg.papers.values(), key=lambda x: x.score, reverse=True)[:3]:
            output.append(
                f"- {p.year} | {p.title}\n  {p.url}"
            )

    return "\n".join(output)

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(
        label="High-level research objective",
        placeholder="e.g. hallucination mitigation in large language models",
        lines=2
    ),
    outputs=gr.Markdown(label="Research Analysis"),
    title="Autonomous Research Discovery Agent",
    description=(
        "Decomposes research objectives, retrieves papers, "
        "ranks relevance, detects gaps, and surfaces insights."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
