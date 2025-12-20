def compute_metrics(state):
    total = len(state.subgoals)
    completed = sum(1 for sg in state.subgoals.values() if sg.completed)

    refinements = sum(sg.refinements for sg in state.subgoals.values())

    return {
        "completion_ratio": completed / total,
        "refinements_used": refinements,
        "questions_asked": 1 if any(sg.gaps for sg in state.subgoals.values()) else 0,
    }
