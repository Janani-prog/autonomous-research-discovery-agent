SAFETY_KEYWORDS = ("safety", "robust", "risk", "verification", "ethic")


def analyze_cross_subgoal_gaps(subgoals: dict):
    """
    Reasons across subgoals to find systemic gaps that no single subgoal's
    own analysis would catch.

    Two of the three rules here used to hardcode exact subgoal names
    ("perception", "planning", "safety") that only ever existed because the
    old planner returned one of exactly two fixed subgoal sets. Now that
    planner.py dynamically generates subgoal names per objective via an LLM,
    an exact-name check like `"perception" in incomplete` would silently
    never fire again for almost any real objective. Rewritten to work off
    keyword/structural signals that hold regardless of what the subgoals
    happen to be named.
    """
    cross_gaps = []

    completed = {name: sg for name, sg in subgoals.items() if sg.completed}
    incomplete = {name: sg for name, sg in subgoals.items() if not sg.completed}

    # Rule 1: safety-adjacent subgoals (identified by keyword, not exact
    # name) lagging behind while everything else is resolved is worth
    # calling out specifically - a review that's otherwise complete but
    # hasn't resolved its safety angle is a different risk profile than one
    # that's just generally incomplete.
    safety_incomplete = [
        name for name, sg in incomplete.items()
        if any(k in sg.name.lower() or k in sg.description.lower() for k in SAFETY_KEYWORDS)
    ]
    if safety_incomplete and completed:
        cross_gaps.append(
            f"Safety-related subgoal(s) ({', '.join(safety_incomplete)}) remain incomplete "
            f"while other subgoals ({', '.join(completed)}) are complete."
        )

    # Rule 2: a majority of subgoals still unresolved after their refinement
    # budget signals a systemic coverage gap in the underlying literature,
    # not just one isolated weak subgoal.
    if subgoals and len(incomplete) > len(subgoals) / 2:
        cross_gaps.append(
            f"{len(incomplete)}/{len(subgoals)} subgoals remain incomplete after refinement - "
            "this objective may be broader than the current literature supports, or need "
            "more refinement rounds."
        )

    # Rule 3: dataset gaps appearing independently in multiple subgoals is a
    # stronger signal than any single subgoal's own gap list suggests alone.
    dataset_missing = [
        name for name, sg in subgoals.items()
        if any("dataset" in g.lower() for g in sg.gaps)
    ]
    if len(dataset_missing) >= 2:
        cross_gaps.append(
            f"Dataset gaps appear across multiple subgoals: {', '.join(dataset_missing)}"
        )

    return cross_gaps
