import requests

SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1"

def fetch_citations(paper_id, max_depth=1):
    """
    Returns citation graph up to depth N
    """
    visited = set()
    edges = []

    def dfs(pid, depth):
        if depth > max_depth or pid in visited:
            return
        visited.add(pid)

        url = f"{SEMANTIC_SCHOLAR}/paper/{pid}/citations"
        params = {"fields": "paperId"}
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return

        for c in r.json().get("data", []):
            cited = c["citingPaper"]["paperId"]
            edges.append((cited, pid))
            dfs(cited, depth + 1)

    dfs(paper_id, 0)
    return edges
