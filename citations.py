import requests

def fetch_citation_count(arxiv_id: str) -> int:
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {
        "fields": "citationCount"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return 0

        data = r.json()
        return data.get("citationCount", 0)

    except Exception:
        return 0
