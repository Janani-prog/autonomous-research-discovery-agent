def maybe_ask_question(state):
    """
    Only fires when overall confidence is still low after analysis (this
    function is only ever called from the INQUIRE phase branch in agent.py,
    so a "state.phase != INQUIRE" guard - which the previous version had -
    can never actually be true there; it was dead code, removed here).

    The clarification message and which subgoals it references are derived
    from the actual incomplete subgoals and their detected gaps, rather than
    a single fully-generic question returned unconditionally whenever
    confidence was low. The three suggested directions stay fixed by design
    (they're meant to be reusable framings regardless of topic), but which
    subgoals/gaps the message names is now real, per-run data.
    """
    if state.confidence >= 0.6:
        return None

    incomplete = [sg for sg in state.subgoals.values() if not sg.completed]
    if not incomplete:
        return None

    subgoal_names = [sg.name.replace("_", " ") for sg in incomplete]
    gap_topics = sorted({g.split(":")[-1].strip() for sg in incomplete for g in sg.gaps})

    message = (
        f"I'm not fully confident yet - {', '.join(subgoal_names)} "
        f"still {'has' if len(subgoal_names) == 1 else 'have'} unresolved gaps"
    )
    if gap_topics:
        message += f" around {', '.join(gap_topics)}"
    message += "."

    return {
        "type": "clarification",
        "message": message,
        "question": "Which direction should I prioritize?",
        "options": {
            "theory": "Foundational understanding and mechanisms",
            "applications": "Practical systems and implementations",
            "recent breakthroughs": "Latest papers and methods",
        },
    }
