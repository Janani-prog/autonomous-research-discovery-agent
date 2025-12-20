def analyze_cross_subgoal_gaps(subgoals: dict):
    """
    Reason across subgoals to find systemic gaps.
    """
    cross_gaps = []

    completed = {k for k, sg in subgoals.items() if sg.completed}
    incomplete = {k for k, sg in subgoals.items() if not sg.completed}

    # Rule 1: Downstream depends on upstream
    if "perception" in incomplete and "planning" in incomplete:
        cross_gaps.append(
            "Planning relies on perception, but perception has unresolved gaps."
        )

    # Rule 2: Safety depends on everything
    if "safety" in incomplete and completed:
        cross_gaps.append(
            f"Safety remains incomplete while other subsystems ({', '.join(completed)}) are complete."
        )

    # Rule 3: Dataset imbalance
    dataset_missing = [
        name for name, sg in subgoals.items()
        if any("dataset" in g for g in sg.gaps)
    ]

    if len(dataset_missing) >= 2:
        cross_gaps.append(
            f"Dataset gaps appear across multiple subsystems: {', '.join(dataset_missing)}"
        )

    return cross_gaps
