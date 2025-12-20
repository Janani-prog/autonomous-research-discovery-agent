import arxiv
from models import Paper


def search_arxiv(query: str, max_results: int = 25):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for r in search.results():
        paper = Paper(
            id=r.entry_id,
            title=r.title,
            abstract=r.summary,
            year=r.published.year,
        )
        results.append(paper)

    return results
