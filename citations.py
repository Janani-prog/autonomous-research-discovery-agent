import logging
import time

import requests

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}"

# Semantic Scholar's unauthenticated tier allows roughly 3 requests/sec, but
# under real contention (e.g. two of this pipeline's scripts running at once)
# it 429s much sooner than that suggests - a single 3s retry is nowhere near
# enough backoff for a genuinely-throttled window. This gives it real
# exponential backoff and honors Retry-After when the API sends one.
_REQUEST_INTERVAL_S = 0.4
_MAX_ATTEMPTS = 3
_INITIAL_BACKOFF_S = 3.0
_CIRCUIT_COOLDOWN_S = 30.0

# Module-level circuit breaker: once a lookup exhausts its retries while
# still rate limited, we know the *endpoint* is throttled, not just this one
# request - retrying that hard for every subsequent paper in the batch (and
# every paper in every later batch) burns minutes of pure backoff sleep for
# no benefit, since it's virtually certain to hit the same 429. While the
# circuit is open, lookups fail immediately with no network call at all;
# it closes on its own after _CIRCUIT_COOLDOWN_S and the next call re-probes
# for real.
_circuit_open_until = 0.0


def _fetch_paper_data(arxiv_id: str) -> dict | None:
    """
    Returns the Semantic Scholar record for this arXiv id, or None if the
    lookup could not be completed (rate limited after retries, network
    error, no matching record, or the circuit breaker above is currently
    open).

    Returning None instead of {} on failure matters: {"citationCount": 0} is
    what an actual, successfully-fetched paper with zero citations looks
    like. Conflating "we couldn't check" with "this paper genuinely has zero
    citations" silently contaminates anything trained on citation_count with
    false zeros - which is exactly what happened during initial testing here,
    when a concurrent process training the ranker hit sustained 429s and
    every single paper in that batch came back reporting 0 citations.
    """
    global _circuit_open_until

    if time.monotonic() < _circuit_open_until:
        return None

    url = S2_API.format(arxiv_id)
    params = {"fields": "citationCount,references.externalIds"}
    backoff = _INITIAL_BACKOFF_S

    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = requests.get(url, params=params, timeout=10)
        except requests.RequestException as e:
            logger.debug("Semantic Scholar request error for %s: %s", arxiv_id, e)
            return None

        if r.status_code == 200:
            return r.json()

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else backoff

            if attempt < _MAX_ATTEMPTS - 1:
                logger.debug("Semantic Scholar rate limited on %s, waiting %.1fs", arxiv_id, wait_s)
                time.sleep(wait_s)
                backoff *= 3
                continue

            logger.warning(
                "Semantic Scholar still rate limited after %d attempts - opening "
                "circuit breaker for %.0fs (no more citation lookups until then).",
                _MAX_ATTEMPTS, max(wait_s, _CIRCUIT_COOLDOWN_S),
            )
            _circuit_open_until = time.monotonic() + max(wait_s, _CIRCUIT_COOLDOWN_S)
            return None

        return None

    return None


def fetch_citation_count(arxiv_id: str) -> int:
    """Kept for backwards compatibility / standalone use. Prefer
    enrich_papers_with_citations() when scoring a batch - it also builds
    citation_edges in the same pass instead of a second round of calls, and
    it lets callers distinguish a failed lookup from a real zero."""
    data = _fetch_paper_data(arxiv_id)
    if data is None:
        return 0
    return data.get("citationCount", 0) or 0


def enrich_papers_with_citations(papers: list, known_ids: set[str]) -> set[str]:
    """
    Populates citation_count for each paper in `papers`, AND builds real
    citation_edges among `known_ids`: an edge (paper.id, ref_id) is added
    whenever paper's Semantic Scholar reference list includes another paper
    that is also in `known_ids` (typically: the rest of the current
    subgoal's retrieved batch).

    This is what makes graph_score.citation_depth_score meaningful. Before
    this, Paper.citation_edges was never populated by anything in the
    codebase, so citation_depth_score always received an empty list and
    always returned 0 - the "graph-based heuristic" ran, but on an
    always-empty input.

    Deliberately shallow: only edges *within* the current batch are
    captured (a paper's references to papers outside `known_ids` are not
    fetched), since fetching the full global citation graph for every
    paper would multiply the number of API calls far beyond what an
    unauthenticated Semantic Scholar client can sustain. This matches the
    README's stated "citation graphs are shallow by default" limitation.

    Returns the set of paper ids that were actually successfully enriched.
    Callers should only mark those as citation_checked/consumed - a paper
    whose lookup failed (rate limited, network error) should stay eligible
    for a retry on a later pass, rather than being permanently recorded as
    "checked, zero citations" based on a failure.
    """
    enriched_ids = set()

    for paper in papers:
        data = _fetch_paper_data(paper.id)
        if data is None:
            # No point pacing requests to an endpoint that just failed or is
            # circuit-broken - the politeness delay only matters between
            # genuinely successful calls.
            continue

        paper.citation_count = data.get("citationCount", 0) or 0

        edges = []
        for ref in data.get("references", []) or []:
            ref_arxiv_id = (ref.get("externalIds") or {}).get("ArXiv")
            if ref_arxiv_id and ref_arxiv_id in known_ids and ref_arxiv_id != paper.id:
                edges.append((paper.id, ref_arxiv_id))
        paper.citation_edges = edges

        enriched_ids.add(paper.id)
        time.sleep(_REQUEST_INTERVAL_S)

    return enriched_ids
