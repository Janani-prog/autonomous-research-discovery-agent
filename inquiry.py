from models import Phase

def maybe_ask_question(state):
    if state.confidence >= 0.6:
        return None

    if state.phase != Phase.INQUIRE:
        return None

    return {
        "type": "clarification",
        "message": "I’m not fully confident I understand your goal.",
        "question": "Which direction should I prioritize?",
        "options": {
            "theory": "Foundational understanding and mechanisms",
            "applications": "Practical systems and implementations",
            "recent breakthroughs": "Latest papers and methods"
        }
    }
