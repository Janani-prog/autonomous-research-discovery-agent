import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

_PLANNING_PROMPT = """You are a research strategist helping plan a literature review.

Decompose the following high-level research objective into 5 to 7 distinct, \
non-overlapping subgoals that a thorough literature review on this topic would \
need to cover. Each subgoal should be a specific angle of investigation \
(e.g. a mechanism, an evaluation approach, a safety concern, an application \
area) - not a generic restatement of the objective.

OBJECTIVE: {objective}

Respond with ONLY a JSON array, no other text, in exactly this shape:
[
  {{"name": "short_snake_case_id", "description": "one sentence describing what this subgoal covers, referencing the objective"}}
]"""


def _fallback_subgoals(objective: str) -> dict:
    """
    Offline fallback used when no GROQ_API_KEY is configured, or the API call
    fails for any reason (network, rate limit, malformed response). Keeps the
    agent runnable with zero external dependencies, at the cost of a fixed
    generic subgoal set rather than one dynamically decomposed for this
    specific objective.
    """
    generic = [
        "background_and_motivation",
        "methods",
        "evaluation",
        "limitations",
        "future_directions",
    ]
    return {
        name.upper(): {
            "name": name,
            "description": f"{name.replace('_', ' ').title()} related to {objective}",
        }
        for name in generic
    }


def _parse_llm_subgoals(raw: str) -> dict:
    start, end = raw.find("["), raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON array found in LLM response")

    items = json.loads(raw[start:end])
    if not items:
        raise ValueError("LLM returned zero subgoals")

    subgoals = {}
    for item in items:
        name = item["name"].strip().lower().replace(" ", "_")
        subgoals[name.upper()] = {
            "name": name,
            "description": item["description"].strip(),
        }
    return subgoals


def generate_subgoals(objective: str) -> dict:
    """
    Dynamically decomposes any research objective into subgoals via an LLM
    call (Groq, llama-3.3-70b-versatile - free tier).

    The previous version of this function was a 2-branch keyword lookup: it
    checked whether the objective contained "language model"/"llm"/
    "hallucination" and returned one of exactly two fixed 5-subgoal sets
    regardless of what the actual objective was - an objective about, say,
    climate policy would silently get the autonomous-vehicle subgoal set
    (perception/planning/control/localization/safety). This version actually
    reasons about the specific objective text.

    Falls back to a fixed generic template (see _fallback_subgoals) if no
    GROQ_API_KEY is set or the call fails, so the repo still runs without a
    key - just with less specific subgoals.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set - using offline generic subgoal template")
        return _fallback_subgoals(objective)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _PLANNING_PROMPT.format(objective=objective)}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_llm_subgoals(raw)

    except Exception as e:
        logger.warning("LLM planning failed (%s) - falling back to offline template", e)
        return _fallback_subgoals(objective)
