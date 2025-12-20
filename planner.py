def generate_subgoals(objective: str):
    """
    Deterministic subgoal decomposition.
    No LLMs.
    """

    templates = {
        "perception": "Perception systems for autonomous driving",
        "planning": "Planning and decision-making for self-driving cars",
        "control": "Vehicle control and actuation",
        "localization": "Localization and mapping methods",
        "safety": "Safety, verification, and robustness"
    }

    subgoals = {}

    for name, desc in templates.items():
        subgoals[name] = {
            "name": name,
            "description": f"{desc} related to {objective}"
        }

    return subgoals
