import arxiv
from models import Paper

# arxiv>=2.0 moved iteration from Search.results() onto a Client - the old
# `for r in search.results()` call raises AttributeError on any currently
# installable version of the package (confirmed against arxiv==4.0.1, where
# this silently made every single arXiv search in the whole pipeline return
# zero papers, since agent.py's SEARCH phase wraps this call in a bare
# `except Exception: continue`). One client is reused across calls per the
# library's own recommendation, rather than constructed per search.
_client = arxiv.Client()


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

    for r in _client.results(search):
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
