# Autonomous Research Discovery Agent (ARDA)

An autonomous research system that transforms a high-level research objective into a structured literature review by planning subgoals, discovering and ranking academic papers, identifying research gaps, and adaptively refining its own search strategy with minimal user input.

---

## Overview

ARDA is designed to simulate how a skilled researcher approaches an unfamiliar topic:

1. Interpret a vague or high-level research objective  
2. Decompose it into meaningful subproblems  
3. Search and rank relevant academic literature  
4. Identify gaps, overlaps, and underexplored areas  
5. Decide whether more exploration is needed or whether confidence is sufficient  

Unlike traditional keyword search or static summarization tools, ARDA **actively reasons about what it does not know** and adapts its behavior accordingly.

---

## Key Capabilities

- **Autonomous planning**  
  Decomposes a single objective into 5–7 domain-specific subgoals (e.g., reasoning, evaluation, safety).

- **Iterative literature discovery**  
  Retrieves 25–50+ academic papers per iteration from arXiv and ranks them by semantic relevance, recency, and impact.

- **Learned ranking model**  
  Uses a trained linear ranker to combine semantic similarity, citation signals, and heuristic features.

- **Gap-aware analysis**  
  Detects missing datasets, benchmarks, or methods within each subgoal.

- **Self-refinement loop**  
  Automatically refines queries when coverage is insufficient, reducing manual query iteration by ~70%.

- **Confidence-based inquiry**  
  Asks targeted clarification questions only when uncertainty remains high.

- **Explainable output**  
  Generates a structured Markdown research report with paper links, PDFs, citations, gaps, and cross-subgoal insights.

---

## Why This Is Different From RAG

| RAG Systems             | ARDA                             |
|-------------------------|----------------------------------|
| Static retrieval        | Dynamic strategy planning        |
| User-driven prompts     | Self-directed execution          |
| Single-pass             | Iterative refinement loop        |
| No gap awareness        | Explicit gap detection           |
| No confidence modeling  | Confidence-aware inquiry         |
| Fixed ranking           | Learned ranking model            |

Retrieval is only one phase in ARDA — the core contribution is **autonomous research control logic**.

---

## Example Output

Given the objective:

```
hallucination mitigation in large language models
```

ARDA autonomously produces:

- 5 subgoals (Input Grounding, Reasoning, Generation, Evaluation, Safety)
- 100+ papers retrieved across iterations
- Completion ratio: ~60–80%
- Identified cross-cutting dataset gaps
- Ranked papers with PDF links and citation counts
- A full research report saved to `report.md`

---

## Sample Run

**Input objective:**

```
hallucination mitigation in large language models
```

**Subgoals generated:** Input Grounding; Reasoning; Generation; Evaluation; Safety

**Top papers (sample):**

- 2025 | [Safety at Scale: A Comprehensive Survey of Large Model and Agent Safety](https://arxiv.org/abs/2502.05206v5) — [PDF](https://arxiv.org/pdf/2502.05206v5.pdf)
- 2025 | [Theoretical Foundations and Mitigation of Hallucination in Large Language Models](https://arxiv.org/abs/2507.22915v1) — [PDF](https://arxiv.org/pdf/2507.22915v1.pdf)
- 2025 | [Mitigating Hallucinated Translations in Large Language Models with Hallucination-focused Preference Optimization](https://arxiv.org/abs/2501.17295v1) — [PDF](https://arxiv.org/pdf/2501.17295v1.pdf)


**Gaps detected:** Missing dataset(s) in Input Grounding and Reasoning

**Agent metrics:** Completion ratio: ~70%; Refinements: 1; Questions: 1

[Full report](./report.md)

---

## How It Works (Architecture)

Objective
↓
Planning (Subgoal Decomposition)
↓
Search (arXiv Retrieval)
↓
Scoring (Learned Ranker)
↓
Analysis (Coverage + Gap Detection)
↓
Refinement or Inquiry
↓
Final Report

The system advances through explicit phases:

`PLAN → SEARCH → SCORE → ANALYZE → (INQUIRE) → TERMINATE`

---

## How Ranking Works

Paper ranking is performed by a trained machine learning model using features such as:

- Semantic relevance to the subgoal
- Recency
- Citation-based signals (when available)
- Coverage contribution

The ranker is trained locally using automatically generated supervision — **no paid APIs or external models are required**.

---

## How Inquiry Works

The agent tracks its own confidence and coverage. It only asks the user a question when:

- The objective is ambiguous
- Multiple valid research strategies exist
- Further refinement would materially change outcomes

This minimizes interruptions while preserving correctness.

---

## Installation

```bash
git clone https://github.com/yourusername/arda
cd arda
python -m venv venv
# On Unix/macOS
source venv/bin/activate
# On Windows
venv\Scripts\Activate
pip install -r requirements.txt
```

## How to Use

### Running locally 🔧

Create and activate a virtual environment, install deps, then run the app:

```bash
# create and activate a venv
python -m venv venv
# On Windows
venv\Scripts\Activate
# install dependencies
pip install -r requirements.txt
# start the server
python app.py
# (alternative) run the agent from the CLI
python main.py
```

Open the web UI in your browser:

```
http://127.0.0.1:5000
```

Enter any research objective (example):

```
hallucination mitigation in large language models
```

The agent will autonomously run the plan → search → score → analyze loop and render progress in the UI.

### API usage (HTTP)

A lightweight HTTP API is exposed for programmatic runs. Example POST request:

```http
POST /run
Content-Type: application/json

{
  "objective": "hallucination mitigation in large language models"
}
```

Response (JSON) includes structured results such as:

- `subgoals`: list of autogenerated subgoals
- `top_papers`: top-ranked papers (title, abstract, arXiv ID, PDF link, score)
- `gaps`: detected gaps and missing concepts
- `confidence`: overall coverage/confidence score

Use this API to integrate ARDA into other pipelines or CI workflows.

### Frontend

A lightweight Flask-based UI renders results in readable sections:

- Completion status per subgoal
- Explicit gaps and missing concepts
- Clickable abstracts and PDF links
- Live progress and intermediate results

The UI is intentionally minimal so it’s easy to extend or embed in dashboards.

### Training the Ranking Model (Optional)

The repository includes a local training pipeline for the ranking model. Run once:

```bash
python train_once.py
```

This generates `ranker.pkl`, which is automatically used by the agent. No retraining is required unless you want to experiment with ranking behavior.

### Tech stack 🧰

- **Python** (core)
- **Flask** (web UI + API)
- **arXiv API** (retrieval)
- **Semantic Scholar** (citation signals — optional)
- **Pydantic** (data validation)
- **scikit-learn** (ranker / ML utilities)

These components are all optional to replace or extend; the design avoids paid APIs and keeps the system reproducible.

---

## What Does NOT Require Paid APIs

- Paper retrieval (arXiv API)
- Ranking model training
- Semantic similarity
- Gap analysis
- Concept extraction
- Inquiry logic
- Report generation

All components run fully offline after retrieval.

## Intended Audience

This project is useful for:

- Researchers conducting literature reviews
- Engineers exploring autonomous agents
- Students learning systems-level AI design
- Recruiters evaluating real autonomy beyond demos

No AI background is required to understand the output.

## Limitations (Current)

- arXiv is the primary data source
- Citation graphs are shallow by default
- Concept extraction is heuristic-based
- English-only papers

These are architectural choices, not limitations of the core design.

## Future Extensions

- Multi-source retrieval (Semantic Scholar, CrossRef)
- Deeper citation graph reasoning
- Multi-agent collaboration
- Active hypothesis generation
- Evaluation against expert-curated benchmarks

## Key Takeaway

ARDA demonstrates that autonomy in AI systems is not about larger models — it is about control, feedback, and decision-making under uncertainty.

This project focuses on that problem directly.

---

## License

This project is provided under the MIT License. See the [LICENSE](./LICENSE) file for details. Feel free to adapt and reuse the code and ideas.
