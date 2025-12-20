from models import Phase


def maybe_ask_question(state):
    """
    Ask a clarification question only when the agent is genuinely uncertain.
    This function updates state and returns the user's choice (or None).
    """

    # Guard: do not ask repeatedly
    if state.confidence >= 0.6:
        return None

    # Guard: only ask during INQUIRE phase
    if state.phase != Phase.INQUIRE:
        return None

    print("\nI’m not fully confident I understand your goal.")
    print("Should I focus more on:")
    print("1) Theory")
    print("2) Applications")
    print("3) Recent breakthroughs")
    print("Reply with a number.")

    choice = input(">> ").strip()

    # Map choice to intent
    intent_map = {
        "1": "theory",
        "2": "applications",
        "3": "recent breakthroughs"
    }

    if choice not in intent_map:
        print("Invalid input. Proceeding with current strategy.")
        state.confidence += 0.1
        return None

    # Update state based on user input
    state.user_intent = intent_map[choice]
    state.confidence += 0.3

    print(f"\n✔ Noted preference: focus on {state.user_intent}")

    return state.user_intent
