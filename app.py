from flask import Flask, request, jsonify, render_template
from agent import run_agent
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(silent=True) or {}
    objective = data.get("objective")

    if not objective:
        return jsonify({"error": "Missing objective"}), 400

    state = run_agent(objective)

    response = {
        "objective": state.objective,
        "confidence": state.confidence,
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

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
