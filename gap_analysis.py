def detect_gaps(papers: list, goal: str):
    if len(papers) < 10:
        return [f"Very little research found for: {goal}"]

    recent = [p for p in papers if p.year >= 2022]
    if not recent:
        return [f"No recent work found on: {goal}"]

    return []
