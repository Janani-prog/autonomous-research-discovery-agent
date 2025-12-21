from fastapi import FastAPI
from agent import run_agent

app = FastAPI(
    title="Autonomous Research Discovery Agent",
    description="Self-directed research agent with gap analysis and inquiry control",
    version="1.0.0"
)

@app.post("/run")
def run_research(payload: dict):
    objective = payload.get("objective")
    if not objective:
        return {"error": "Missing 'objective' in request body"}

    state = run_agent(objective)

    response = {
        "objective": state.objective,
        "confidence": state.confidence,
        "inquiry": state.inquiry,
        "subgoals": {}
    }

    for name, sg in state.subgoals.items():
        response["subgoals"][name] = {
            "description": sg.description,
            "completed": sg.completed,
            "gaps": sg.gaps,
            "top_papers": [
                {
                    "title": p.title,
                    "year": p.year,
                    "url": p.url,
                    "pdf": p.pdf_url,
                    "citations": p.citation_count,
                    "score": round(p.score, 3)
                }
                for p in sorted(
                    sg.papers.values(),
                    key=lambda p: p.score,
                    reverse=True
                )[:3]
            ]
        }

    return response
