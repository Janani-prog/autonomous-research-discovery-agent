from train_data import build_training_examples
from train_ranker import train_ranker
from retrieval import search_arxiv
from scoring import score_paper


def collect_training_papers():
    """
    Collect a small corpus of papers to bootstrap the ranking model.
    """
    queries = [
        "self-driving car perception",
        "autonomous vehicle planning",
        "autonomous driving safety",
    ]

    papers = []
    seen = set()

    for q in queries:
        for p in search_arxiv(q)[:15]:
            if p.id not in seen:
                seen.add(p.id)
                papers.append(p)

    return papers


# 1️⃣ Collect papers
papers = collect_training_papers()

# 2️⃣ Compute heuristic scores + features
for p in papers:
    # Base heuristic score (what we currently use)
    p.score = score_paper(p, "autonomous driving")

    # Feature components (very important)
    p.semantic_score = p.score          # proxy
    p.recency_score = 1 / (2025 - p.year + 1)
    p.citation_score = 0.5              # placeholder if graph not ready
    p.concept_coverage = 0.5             # placeholder

# 3️⃣ Build training data
X, y = build_training_examples(papers)

print(f"Training samples: {len(X)}")

# 4️⃣ Train and save model
train_ranker(X, y)

print("✓ Ranker trained and saved to ranker.pkl")
