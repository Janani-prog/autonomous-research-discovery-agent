# Autonomous Research Discovery Agent (ARDA)

## Overview

ARDA is a self-directed research agent that autonomously plans, executes, evaluates, and refines literature reviews starting from a single high-level research objective.

Unlike traditional search tools or RAG systems, ARDA does not simply retrieve and summarize papers. It actively decides *what to search*, *how to rank results*, *what knowledge is missing*, and *when to ask the user clarifying questions*, demonstrating true autonomous research behavior.

The system is domain-agnostic and adapts its strategy dynamically based on internal confidence, coverage metrics, and gap analysis — without requiring prompt engineering or manual iteration.

---

## What This Project Does

Given a single research objective (for example:
> *“hallucination mitigation in large language models”*),

the system automatically:

1. **Plans** a multi-subgoal research strategy
2. **Searches** academic sources (arXiv) without manual queries
3. **Ranks** papers using a trained machine learning model
4. **Builds concept coverage maps** across subgoals
5. **Detects research gaps and contradictions**
6. **Refines queries autonomously** when coverage is insufficient
7. **Asks targeted clarifying questions only when necessary**
8. **Terminates automatically** when confidence and coverage are sufficient
9. **Generates a structured research report**

No step requires the user to manually guide the process beyond the initial goal.

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

## Example Output (Real Run)

**Objective:**
```
hallucination mitigation in large language models
```

**Agent Metrics:**

- Completion ratio: **0.8**
- Refinements used: **1**
- Questions asked: **1**

**Observed Behavior:**

- Generated 5 research subgoals automatically
- Retrieved and ranked 50+ papers
- Detected unresolved gaps only in perception-related areas
- Asked a single clarifying question to refine focus
- Terminated early due to sufficient coverage

This demonstrates reduced user intervention and increased autonomy compared to earlier runs (completion ratio ~0.2).

---

## System Architecture (Conceptual)

1. Objective
2. Planning (Subgoals)
3. Search (arXiv)
4. Learned Ranking Model
5. Concept Coverage Mapping
6. Gap & Contradiction Analysis
7. Refinement or Inquiry
8. Final Research Report

The agent decides which transition to take based on internal state, not hardcoded rules.

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

### Run the Agent

```bash
python main.py
```

When prompted, enter any research objective, for example:

```
hallucination mitigation in large language models
```

The agent will handle everything else.

### Change the Research Topic

No code changes are required. Simply run the agent again with a different objective, for example:

```bash
safety verification of machine learning systems
```

or

```bash
robustness of multimodal foundation models
```

The system adapts automatically.

### Training the Ranking Model (Optional)

The repository includes a local training pipeline for the ranking model. Run once:

```bash
python train_once.py
```

This generates `ranker.pkl`, which is automatically used by the agent.

No retraining is required unless you want to experiment with ranking behavior.

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
