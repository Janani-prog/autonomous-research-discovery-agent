import gradio as gr
from agent import run_agent
import time

GITHUB_URL = "https://github.com/Janani-prog/autonomous-research-discovery-agent"

def run_with_progress(objective, progress=gr.Progress()):
    progress(0.05, desc="Planning research strategy")
    time.sleep(0.3)

    progress(0.15, desc="Decomposing objective into subgoals")
    time.sleep(0.3)

    progress(0.30, desc="Retrieving relevant papers from arXiv")
    time.sleep(0.3)

    progress(0.50, desc="Scoring & ranking papers by relevance")
    time.sleep(0.3)

    progress(0.70, desc="Analyzing concepts & detecting research gaps")
    time.sleep(0.3)

    progress(0.90, desc="Aggregating insights across subgoals")
    time.sleep(0.3)

    state = run_agent(objective)

    progress(1.0, desc="Research synthesis complete")

    # ---------- FORMAT OUTPUT ----------
    output = []
    output.append(f"### Objective\n{state.objective.strip()}")
    output.append(f"### Agent Confidence\n**{round(state.confidence, 2)}**\n")

    for name, sg in state.subgoals.items():
        output.append(f"---\n## {name.replace('_', ' ')}")
        output.append(sg.description.strip())
        output.append(f"**Coverage Status:** {'Yes' if sg.completed else 'Partial'}")

        if sg.gaps:
            output.append("\n**Identified Gaps:**")
            for g in sg.gaps:
                output.append(f"- {g}")

        output.append("\n**Top Papers:**")
        for p in sorted(sg.papers.values(), key=lambda p: p.score, reverse=True)[:3]:
            output.append(
                f"- **{p.year}** | [{p.title}]({p.url}) "
                f"(score: {round(p.score, 3)})"
            )

    return "\n".join(output)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Autonomous Research Discovery Agent
        Decomposes open-ended research objectives into subgoals, retrieves and ranks academic literature,
        detects conceptual gaps, and surfaces research insights **autonomously**.
        """
    )

    with gr.Row():
        objective = gr.Textbox(
            label="High-level research objective",
            placeholder="e.g. hallucination mitigation in large language models",
            lines=2
        )

    run_btn = gr.Button("Run Agent")
    output = gr.Markdown()

    gr.Markdown(
        f"""
        🔗 **Explore full methodology & code:** {GITHUB_URL}
        """
    )

    run_btn.click(
        fn=run_with_progress,
        inputs=objective,
        outputs=output
    )

demo.queue().launch()
