import os

from planner import generate_subgoals, _fallback_subgoals


def test_falls_back_offline_with_no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    subgoals = generate_subgoals("quantum error correction for scalable qubits")

    assert subgoals == _fallback_subgoals("quantum error correction for scalable qubits")
    assert len(subgoals) == 5


def test_fallback_subgoals_reference_the_objective():
    subgoals = _fallback_subgoals("some novel objective text")
    for entry in subgoals.values():
        assert "some novel objective text" in entry["description"]
