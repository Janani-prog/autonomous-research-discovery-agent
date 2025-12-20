def generate_subgoals(objective: str):
    obj = objective.lower()

    # --- LLM / NLP DOMAIN ---
    if "language model" in obj or "llm" in obj or "hallucination" in obj:
        return {
            "INPUT_GROUNDING": {
                "name": "input_grounding",
                "description": "Input grounding and context alignment related to " + objective
            },
            "REASONING": {
                "name": "reasoning",
                "description": "Reasoning and inference mechanisms related to " + objective
            },
            "GENERATION": {
                "name": "generation",
                "description": "Text generation behavior and decoding strategies related to " + objective
            },
            "EVALUATION": {
                "name": "evaluation",
                "description": "Evaluation methods and benchmarks related to " + objective
            },
            "SAFETY": {
                "name": "safety",
                "description": "Safety, robustness, and verification related to " + objective
            }
        }

    # --- AUTONOMOUS SYSTEMS DOMAIN (default) ---
    return {
        "PERCEPTION": {
            "name": "perception",
            "description": "Perception systems related to " + objective
        },
        "PLANNING": {
            "name": "planning",
            "description": "Planning and decision-making related to " + objective
        },
        "CONTROL": {
            "name": "control",
            "description": "Control and actuation related to " + objective
        },
        "LOCALIZATION": {
            "name": "localization",
            "description": "Localization and mapping related to " + objective
        },
        "SAFETY": {
            "name": "safety",
            "description": "Safety, verification, and robustness related to " + objective
        }
    }
