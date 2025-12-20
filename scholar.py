from scholarly import scholarly

def search_scholar(query):
    results = []
    for r in scholarly.search_pubs(query):
        results.append({
            "title": r.bib["title"],
            "year": r.bib.get("year"),
            "citations": r.citedby
        })
        if len(results) >= 10:
            break
    return results
