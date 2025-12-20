def extract_claims(text: str):
    """
    Very conservative claim extraction.
    """
    indicators = ["outperforms", "improves", "fails", "cannot", "significantly"]
    claims = []

    for sent in text.split("."):
        for i in indicators:
            if i in sent.lower():
                claims.append(sent.strip())
                break

    return claims
