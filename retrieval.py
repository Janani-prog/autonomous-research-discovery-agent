import arxiv
from models import Paper


def search_arxiv(query: str, max_results: int = 25):
    """
    Broad recall-first arXiv search.
    Returns a list of Paper objects.
    """

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []

    for r in search.results():
        try:
            entry_id = r.entry_id.split("/")[-1]

            paper = Paper(
                id=entry_id,
                title=r.title.strip(),
                abstract=r.summary.strip(),
                year=r.published.year,
                url=f"https://arxiv.org/abs/{entry_id}",
                pdf_url=f"https://arxiv.org/pdf/{entry_id}.pdf",
                citation_count=0  # filled later via Semantic Scholar
            )

            results.append(paper)

        except Exception:
            continue

    return results
