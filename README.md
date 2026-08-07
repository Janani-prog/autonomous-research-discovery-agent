# Autonomous Research Discovery Agent (ARDA)

An autonomous research system that transforms a high-level research objective into a structured literature review by planning subgoals, discovering and ranking academic papers, identifying research gaps, and adaptively refining its own search strategy with minimal user input.

---

## Overview

ARDA simulates how a researcher approaches an unfamiliar topic:

1. Interpret a high-level research objective
2. Decompose it into meaningful subgoals
3. Search and rank relevant academic literature
4. Identify gaps and underexplored areas
5. Decide whether more exploration is needed or whether confidence is sufficient

It runs as an explicit state machine — `PLAN → SEARCH → SCORE → ANALYZE → (INQUIRE) → TERMINATE` — not an LLM agent-executor loop. The autonomy here is **control-flow autonomy**: a hand-written loop re-searches and re-scores without a human in the loop between phases, and only the PLAN phase itself calls an LLM. Being precise about that distinction matters more than the word "autonomous" — see [How It Works](#how-it-works-architecture) below for exactly what each phase does.

---

## Key Capabilities

- **LLM-based planning** — an objective is decomposed into 5–7 subgoals via a Groq (`llama-3.3-70b-versatile`) call that reasons about the specific objective text, with an offline fallback template if no API key is configured. (Previously this was a 2-branch keyword lookup returning one of exactly two fixed subgoal sets regardless of the actual objective — see [What Changed](#what-changed-in-this-rebuild).)
- **Iterative literature discovery** — retrieves papers per subgoal from arXiv and re-searches with refined queries when gaps are detected.
- **Learned ranking model** — a `scikit-learn` linear model predicts a query-independent "impact" prior from recency, citation-graph position, and concept coverage, blended at serving time with real-time TF-IDF query relevance.
- **Real citation graph** — citation edges between retrieved papers are built from actual Semantic Scholar reference data, not left as a permanently-empty placeholder.
- **Gap-aware analysis** — detects missing expected concepts (dataset, method, evaluation, etc.) per subgoal via TF-IDF-based concept extraction.
- **Self-refinement loop** — automatically reformulates queries (capped at 2 rounds per subgoal) when a subgoal's gaps remain unresolved.
- **Confidence-based inquiry** — asks a clarification question, naming the specific subgoals and gaps involved, only when overall confidence stays below threshold after analysis.
- **Explainable output** — a structured Markdown report with paper links, PDFs, citation counts, and per-subgoal gaps.

---

## How It Works (Architecture)

```
Objective
  │
  ▼
PLAN     — Groq LLM decomposes the objective into 5-7 subgoals
  │         (falls back to a fixed generic template if no GROQ_API_KEY)
  ▼
SEARCH   — arXiv query per subgoal (or per refined query, after a refinement round)
  ▼
SCORE    — Semantic Scholar citation-graph enrichment (batched, once per new paper)
  │         + TF-IDF query relevance (fit once per subgoal batch)
  │         + learned ranker's impact prior (recency / citation-depth / concept-coverage -> predicted impact)
  ▼
ANALYZE  — TF-IDF concept extraction, gap detection, contradiction detection (safety-adjacent subgoals),
  │         cross-subgoal reasoning
  │
  ├─(gaps unresolved, refinements < 2)──▶ back to SEARCH with refined queries
  │
  ▼
INQUIRE  — only if confidence still < 0.6 after analysis: a context-specific clarification question
  ▼
Final Report (report.md)
```

---

## What Changed In This Rebuild

An earlier version of this README described capabilities that, on closer inspection of the code, didn't hold up — either because a supporting file existed but was never actually imported by the live pipeline, or because a "learned"/"graph-based" feature was structurally always a no-op. Being upfront about exactly what was fixed is part of what makes the current numbers below trustworthy:

| Issue found | Fix |
|---|---|
| **The state machine's ANALYZE phase computed `sg.refined_queries` and incremented `sg.refinements` whenever a subgoal had gaps, but the phase-transition logic only ever chose between INQUIRE and TERMINATE next - no code path ever set `state.phase` back to SEARCH. The advertised "iterative self-refinement loop" could structurally never execute more than one search round per run; refined queries were computed and immediately discarded** | ANALYZE now transitions back to SEARCH when a subgoal still has refinement budget left; SEARCH consumes and clears `refined_queries` after running them so a round isn't repeated |
| Per-paper retry/backoff meant that once Semantic Scholar's endpoint was genuinely throttled (not just one unlucky request), *every subsequent paper* still paid its own full retry-with-backoff cost hitting the same already-known-bad endpoint - multiplying minutes of wasted retry sleep across a whole run | Added a module-level circuit breaker in `citations.py`: once retries are exhausted while still rate limited, all lookups fail fast (no network call) for a cooldown window instead of each paying the same retry cost |
| `train_once.py` called `model.fit()` even with zero training samples (e.g. if every citation lookup in a run failed), crashing with a raw sklearn `ValueError` instead of failing clearly | Added a `MIN_TRAINING_SAMPLES` guard that exits with an actionable error instead of crashing |
| `citations.py`'s Semantic Scholar lookup returned `{}` (indistinguishable from "paper genuinely has 0 citations") on *any* failure - including a 429 after only one weak retry. Under real contention this silently trained the ranker on a degenerate all-zero label range, since every lookup in the batch failed | Lookup now returns `None` on failure (not `{}`), with real exponential backoff honoring `Retry-After`; only successfully-enriched papers are used for training or marked `citation_checked` |
| Once the refinement loop above actually worked, `refine.py`'s uncapped `gaps[:2] x strategies[:2]` fan-out (up to 4 new full arXiv + citation-enrichment searches per refinement round) turned into a real combinatorial cost - up to 9 full searches per subgoal across 2 rounds, times every subgoal in a run | Added `MAX_REFINED_QUERIES = 2` cap in `refine.py` |
| **`retrieval.py` called `arxiv.Search.results()`, an API removed from the `arxiv` package years ago (confirmed against the currently-installable `arxiv==4.0.1`) — every single arXiv search in the pipeline raised `AttributeError`, silently swallowed by a bare `except Exception: continue` in `agent.py`, so every run retrieved zero papers with a fresh `pip install`** | Use `arxiv.Client().results(search)`, the current API |
| `main.py` crashed with `UnicodeEncodeError` on Windows' default console codepage (cp1252) from a `📄` emoji in a `print()` call | Removed the emoji from console output |
| `planner.py` was a 2-branch keyword lookup — any objective not matching "language model"/"llm"/"hallucination" got a fixed autonomous-vehicle subgoal set (perception/planning/control/localization/safety), regardless of topic | Real Groq LLM call dynamically decomposes the specific objective; offline fallback if no key |
| `Paper.citation_edges` was never populated by any code path, so `citation_depth_score` always computed on an empty list and always returned 0 | `citations.py` now builds real edges from Semantic Scholar reference data within each retrieved batch |
| The ranker's training script set a feature (`semantic_score`) equal to the label (`p.score`) before training — target leakage; a linear model handed a copy of its own target trivially "solves" the problem without learning anything real | Ranker retrained on citation-count-derived weak supervision, with recency/citation-depth/concept-coverage as inputs — none of which are a copy of the label |
| The feature vector `train_data.py` built and the one `scoring.py` served at inference used different orders/meanings for the same positions — train/serve skew | Both now go through one shared `features.py` module and a single `RANKER_FEATURE_KEYS` ordering |
| `TfidfVectorizer` was refit on just `[query, single_paper]` per call — a 2-document IDF fit carries essentially no real document-frequency signal | Fit once per subgoal batch across the query + all candidates (`features.batch_semantic_similarity`) |
| `embeddings.py` (FAISS + `sentence-transformers`), `scholar.py` (Google Scholar), `local_reasoner.py` (local Mistral-7B via `llama_cpp`), and `gap_analysis.py` existed in the repo but were never imported by any live code path | Removed. `render.yaml` pointed at a `uvicorn api:app` that doesn't exist anywhere in the repo (the real deployment is Gradio's `app.py` on HF Spaces) — removed |
| `main.py`'s "Concept Coverage" print was outside its `for` loop (indentation), so it only ever printed the *last* subgoal's concept map | Fixed indentation — prints per-subgoal |
| `inquiry.py` had a dead branch (`if state.phase != Phase.INQUIRE: return None`) that can never be true given how it's called, and returned an identical hardcoded question regardless of which subgoal caused low confidence | Dead branch removed; message now names the actual incomplete subgoals/gaps |
| `contradictions.py` flagged a "contradiction" from any two claims across *any* papers sharing an "improves"/"fails" keyword pair, with no check they were even about the same subject | Added a content-word Jaccard-overlap threshold before flagging |
| The README's "~70% reduction in manual search effort" and "~4x topical coverage" claims had no supporting computation anywhere in the codebase | Replaced with `benchmark.py`, a real, documented, reproducible measurement — see [Benchmark](#benchmark) below |

Sentence-transformer/FAISS-based embeddings were deliberately *not* reintroduced for semantic scoring — an earlier commit ([`dc6f07e`](../../commit/dc6f07e)) removed them specifically for low-memory deployment, and that constraint is respected here; the TF-IDF fix above addresses the actual bug (per-pair refitting) without reversing that decision.

---

## Benchmark

`benchmark.py` runs the real agent end-to-end against a fixed set of 6 objectives spanning unrelated domains (LLM hallucination, PEFT, autonomous vehicles, robotics RL, molecular GNNs, climate modeling — chosen upfront, not cherry-picked after seeing results), and compares it against a simulated "manual" baseline: a single arXiv query using the raw objective text, retrieving the same number of papers the agent ended up processing (so both are compared at equal paper-reading cost).

**A first version of this compared `len(agent_concepts)` vs `len(manual_concepts)` — a "coverage ratio".** That metric turned out to be structurally incapable of showing a real difference: the underlying TF-IDF extractor has its own vocabulary ceiling (`max_features=500`), and any real 40–160+ paper corpus has more than enough distinct terms to fill whatever top-k is requested — so both sides saturated at *exactly* the same count every time (tried at top_k=25, then top_k=150; both sides tied exactly, every run, at whichever ceiling was set). The fix: compare the actual concept **sets**, not their sizes — intersection, and what each side found that the other missed entirely.

It reports, per objective and averaged across the set:
- **Jaccard similarity** — how much the two approaches' concept sets overlap overall. This is the metric that's actually free to vary (see the caveat below on why the next one isn't).
- **Concepts only the agent found / only manual search found** — reported for transparency, but see the results caveat below: under this design (equal paper budgets, shared vocabulary ceiling), these two numbers are mathematically forced to be equal, not a genuine "which side covers more" signal.
- **Autonomous queries per run** — how many distinct search-and-reformulate cycles the agent performs on its own, vs. the one query a manual baseline uses. Honest analogue of the old "~70% manual effort reduction" claim.

Run it yourself: `python benchmark.py` (needs `GROQ_API_KEY`; takes a long time — dozens of real arXiv, Semantic Scholar, and Groq calls per objective. A mid-run network outage during testing here genuinely dropped 4 of 6 objectives with connection errors unrelated to the agent logic; the script now retries a failed objective once before giving up on it).

### Results (real run, `benchmark_results.json`)

5 of 6 objectives completed (the 6th — "hallucination mitigation in large language models" on this particular run — hit a transient `IncompleteRead` connection error on both its attempt and its retry; everything else succeeded):

| Metric | Avg across 5 objectives |
|---|---|
| Jaccard similarity (agent vs. manual concept sets) | **0.56** |
| Concepts found only by the agent / only by manual search | 113.2 / 113.2 |
| Shared concepts | 286.8 |
| Autonomous queries issued per run | **18.6** |

**Read this carefully, not as "4x coverage":** `agent_only_concepts` and `manual_only_concepts` come out *identical* in every single objective (95/95, 135/135, 114/114, 103/103, 119/119). That isn't the model tying by coincidence — it's forced by set theory: both sides are matched to the same paper budget and both saturate the extractor's `max_features=500` vocabulary ceiling, so `|agent_concepts| = |manual_concepts|` always holds here, and whenever two sets are equal in size, `|A\B|` always equals `|B\A|`. An earlier version of this benchmark used raw set *sizes* (`len(agent_concepts)` vs `len(manual_concepts)`) as a "coverage ratio" and hit the exact same problem one level up — both sides saturated at whichever `top_k` was chosen (25, then 150, both times an exact tie). Comparing sizes here structurally cannot show a "decomposition wins" result, no matter how good the decomposition is — this took two rebuilds of the metric to actually notice.

**Jaccard similarity (0.56) is the number that's actually free to vary, and it's real:** it means the two approaches' concept vocabularies overlap by only about 56% — 44% of the *combined* vocabulary is exclusive to one approach or the other. That's a genuinely different literature footprint, not a bigger or smaller one. The honest interpretation: at equal paper-reading cost, multi-subgoal decomposition doesn't straightforwardly find "more" — it finds *different* literature, consistent with deliberately searching distinct conceptual angles (subgoals) rather than one generic query. Whether that's more valuable than raw volume is a judgment call, not something this number settles by itself.

**Autonomous queries issued per run (18.6) is a clean, non-tautological number:** it's a direct count of how many distinct search-and-reformulate cycles the agent executed on its own per run, against the single query a manual baseline uses. That's the honest analogue of "reduces manual search effort" — not a percentage, just a literal count of automated iterations.

---

## Tech Stack

- **Python** — core system orchestration
- **Groq API (`llama-3.3-70b-versatile`)** — LLM-based subgoal planning
- **arXiv API** — academic paper retrieval
- **scikit-learn** — `TfidfVectorizer` for query relevance and concept extraction, `LinearRegression` for the learned impact ranker
- **Semantic Scholar API** — citation counts and reference-graph edges
- **Gradio** — interactive UI (HuggingFace Spaces)
- **Pydantic** — typed state (`Paper`, `Subgoal`, `ResearchState`)

No FAISS/`sentence-transformers`/torch — deliberately kept off the dependency list for low-memory deployment (see [What Changed](#what-changed-in-this-rebuild)).

---

## Installation

```bash
git clone https://github.com/Janani-prog/autonomous-research-discovery-agent
cd autonomous-research-discovery-agent
python -m venv venv
# Windows: venv\Scripts\activate    |    Unix/macOS: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add GROQ_API_KEY to .env (free tier: https://console.groq.com/keys)
```

## Usage

```bash
# Run the agent on a single objective, print a report to the console + report.md
python main.py

# Launch the Gradio UI
python app.py

# Train (or retrain) the ranking model - writes ranker.pkl, used automatically if present
python train_once.py

# Run the real coverage/effort benchmark (see above)
python benchmark.py

# Run the test suite
pytest tests/ -v
```

Without `GROQ_API_KEY` set, the agent still runs end-to-end — `planner.py` falls back to a fixed generic subgoal template (background/methods/evaluation/limitations/future-directions) instead of a dynamically decomposed one.

---

## What To Pay Attention To

- Planning is genuinely dynamic per-objective now (LLM-based), not a lookup table — but it's still one LLM call per run, not a multi-step planning/tool-use agent.
- The ranker predicts a query-independent "impact" prior from cheap features; it doesn't replace real-time query relevance, which is computed fresh via TF-IDF every run.
- Citation graphs are intentionally shallow — edges are only captured *within* the current retrieved batch per subgoal, not the full global citation graph (see `citations.py`).
- Concept extraction and gap detection are TF-IDF/keyword-heuristic based, not semantic/NLP-model based. This is a deliberate scope choice, not a hidden gap.
- The benchmark numbers above are real, single-run measurements against live external APIs (arXiv, Semantic Scholar rate limits, Groq's free tier), not averaged over many repeated trials — treat them as a directional signal, not a tightly-controlled experiment.

## Limitations (Current)

- arXiv is the primary literature source; Semantic Scholar is used only for citation counts/edges of papers arXiv already returned.
- Citation graphs are shallow by default (see above).
- Concept extraction and contradiction detection are heuristic/keyword-based, not semantic-model-based.
- English-only papers.
- The learned ranker is trained on a small bootstrap corpus (`train_once.py`'s `TRAINING_QUERIES`) — retrain periodically as its scope grows.
- Semantic Scholar's unauthenticated API tier rate-limits aggressively and appears to share throttling state across unrelated traffic on the same egress IP in some hosting environments — `train_once.py` and the SCORE phase's citation enrichment can both legitimately return 0 results for a while under these conditions. Both fail gracefully (a paper's citation-graph features default to 0/heuristic rather than crashing), but a fully-throttled run means the ranker and citation-depth score aren't contributing real signal for that run. Getting a Semantic Scholar API key removes this constraint.

## Future Extensions

- Multi-source retrieval (Semantic Scholar's own search, CrossRef)
- Deeper citation graph reasoning beyond the current retrieved batch
- Multi-agent collaboration
- Evaluation against expert-curated benchmarks, not just the self-benchmark above

## License

MIT License. See [LICENSE](./LICENSE).
