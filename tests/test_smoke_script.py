"""The live smoke-test script, run offline against a fake OpenRouter.

This tests the script's logic (checks, report, recordings, key handling), not
OpenRouter itself: that is what running it with --live is for.
"""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue, SecretStr, TypeAdapter

import smoke_openrouter as smoke
from sre_agent.harness import ROUTE_ANTHROPIC

FAKE_KEY = "sk-or-v1-test-not-a-real-key-0123456789"
SERVED = {"anthropic": "Anthropic", "amazon-bedrock": "Amazon Bedrock"}
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

TOOLS_TURN: dict[str, JsonValue] = {
    "role": "assistant",
    "content": None,
    "reasoning_details": [
        {"type": "reasoning.text", "text": "Look both up.", "signature": "c2lnbmF0dXJl"}
    ],
    "tool_calls": [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_service_status", "arguments": '{"service": "payments-api"}'},
        },
        {
            "id": "toolu_2",
            "type": "function",
            "function": {"name": "get_owner", "arguments": '{"service": "payments-api"}'},
        },
    ],
}
FINAL_TURN: dict[str, JsonValue] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "toolu_3",
            "type": "function",
            "function": {
                "name": "final_answer",
                "arguments": '{"service": "payments-api", "status": "degraded", '
                '"owner": "team-payments"}',
            },
        }
    ],
}


def _fake_openrouter(script: list[str]) -> httpx.MockTransport:
    """Answers requests in order: 'tools', 'final', 'reject' (400) or 'no_provider' (404)."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" in request.headers
        body = _JSON.validate_json(request.content)
        only = smoke.dig(body, "provider", "only", 0)
        assert isinstance(only, str)
        kind = script.pop(0)
        if kind == "reject":
            return httpx.Response(400, json={"error": {"message": "invalid signature"}})
        if kind == "no_provider":
            return httpx.Response(404, json={"error": {"message": "No endpoints found"}})
        message = TOOLS_TURN if kind == "tools" else FINAL_TURN
        return httpx.Response(
            200,
            json={
                "model": ROUTE_ANTHROPIC.model,
                "provider": SERVED[only],
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "native_finish_reason": "tool_use",
                        "message": message,
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 50,
                    "cost": 0.001,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )

    return httpx.MockTransport(handler)


def _outcomes(checks: list[smoke.Check]) -> dict[str, str]:
    return {f"{c.route} {c.name}": c.outcome for c in checks}


def test_dry_run_sends_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke.main([]) == 2
    assert "Dry run: nothing was sent" in capsys.readouterr().out


def test_all_checks_pass_when_routes_behave(tmp_path: Path) -> None:
    script = ["tools", "final", "reject", "tools", "final", "reject", "reject"]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path
    )
    assert not script  # every scripted response was used: 7 requests
    outcomes = _outcomes(checks)
    for route in ("r1-anthropic", "r2-bedrock-zdr"):
        for name in ("run", "pin", "strict_args", "usage", "edited_history"):
            assert outcomes[f"{route} {name}"] == "PASS", f"{route} {name}"
    assert outcomes["cross cross_route"] == "PASS"


def test_silent_acceptance_is_a_finding(tmp_path: Path) -> None:
    # The edited-history request comes back 200 instead of 400.
    script = ["tools", "final", "final", "tools", "final", "reject", "final"]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path
    )
    outcomes = _outcomes(checks)
    assert outcomes["r1-anthropic edited_history"] == "FINDING"
    assert outcomes["cross cross_route"] == "FINDING"


def test_no_eligible_provider_on_r2_is_a_finding_not_a_failure(tmp_path: Path) -> None:
    script = ["tools", "final", "reject", "no_provider", "reject"]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path
    )
    outcomes = _outcomes(checks)
    assert outcomes["r2-bedrock-zdr run"] == "FINDING"
    assert "r2-bedrock-zdr edited_history" not in outcomes


def test_recordings_are_saved_without_the_key(tmp_path: Path) -> None:
    script = ["tools", "final", "reject", "tools", "final", "reject", "reject"]
    smoke.run_smoke(transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert "r1-anthropic/run-01.json" in files
    assert "r1-anthropic/edited_history-01.json" in files
    assert "cross/cross_route-01.json" in files
    assert "report.json" in files
    for path in tmp_path.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert FAKE_KEY not in text
        assert "authorization" not in text.lower()
    first = json.loads((tmp_path / "r1-anthropic" / "run-01.json").read_text(encoding="utf-8"))
    assert first["status"] == 200
    assert first["request"]["provider"]["only"] == ["anthropic"]


def test_a_recording_containing_the_key_is_refused(tmp_path: Path) -> None:
    leak = smoke.Exchange("r1/run", {"note": FAKE_KEY}, 200, {})
    with pytest.raises(RuntimeError, match="refusing to write"):
        smoke.save_recordings([leak], [], tmp_path, SecretStr(FAKE_KEY))
    assert not list(tmp_path.iterdir())  # nothing written at all
