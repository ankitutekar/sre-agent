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


CHAT = ("r1-anthropic", "r2-bedrock-zdr")
MESSAGES_ROUTES = ("r1m-anthropic", "r2m-bedrock-zdr")

M_THINKING: dict[str, JsonValue] = {
    "type": "thinking",
    "thinking": "Look both up.",
    "signature": "c2lnbmF0dXJl",
}


def _m_tool_use(id_: str, name: str, args: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


M_TOOLS: list[JsonValue] = [
    _m_tool_use("toolu_1", "get_service_status", {"service": "payments-api"}),
    _m_tool_use("toolu_2", "get_owner", {"service": "payments-api"}),
]
M_FINAL: list[JsonValue] = [
    _m_tool_use(
        "toolu_3",
        "final_answer",
        {"service": "payments-api", "status": "degraded", "owner": "team-payments"},
    )
]


def _messages_response(kind: str, served: str) -> dict[str, JsonValue]:
    contents: dict[str, list[JsonValue]] = {
        "tools": [M_THINKING, *M_TOOLS],
        "plain": M_TOOLS,
        "final": M_FINAL,
    }
    content = contents[kind]
    return {
        "type": "message",
        "role": "assistant",
        "model": ROUTE_ANTHROPIC.model,
        "provider": served,
        "content": content,
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 500,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cost": 0.001,
        },
    }


def _fake_openrouter(script: list[str]) -> httpx.MockTransport:
    """Answers requests in order: 'tools', 'plain' (no reasoning), 'final', 'reject' (400)
    or 'no_provider' (404), in the wire format of the endpoint that was called."""

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
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json=_messages_response(kind, SERVED[only]))
        if kind == "plain":  # a tool turn without reasoning (adaptive thinking skipped it)
            message = {k: v for k, v in TOOLS_TURN.items() if k != "reasoning_details"}
        else:
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
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
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
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    outcomes = _outcomes(checks)
    assert outcomes["r1-anthropic edited_history"] == "FINDING"
    assert outcomes["cross cross_route"] == "FINDING"


def test_no_eligible_provider_on_r2_is_a_finding_not_a_failure(tmp_path: Path) -> None:
    script = ["tools", "final", "reject", "no_provider", "reject"]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    outcomes = _outcomes(checks)
    assert outcomes["r2-bedrock-zdr run"] == "FINDING"
    assert "r2-bedrock-zdr edited_history" not in outcomes


def test_recordings_are_saved_without_the_key(tmp_path: Path) -> None:
    script = ["tools", "final", "reject", "tools", "final", "reject", "reject"]
    smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
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


def test_probe_runs_when_the_main_run_has_no_reasoning(tmp_path: Path) -> None:
    script = [
        *("plain", "final", "tools", "final", "reject"),  # R1: run, probe, edited history
        *("tools", "final", "reject"),  # R2: run has reasoning, so no probe
        "reject",  # cross route
    ]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    assert not script
    outcomes = _outcomes(checks)
    assert outcomes["r1-anthropic probe"] == "INFO"
    assert "r2-bedrock-zdr probe" not in outcomes
    assert outcomes["r1-anthropic edited_history"] == "PASS"
    assert outcomes["cross cross_route"] == "PASS"
    probe = json.loads((tmp_path / "r1-anthropic" / "probe-01.json").read_text(encoding="utf-8"))
    assert probe["request"]["reasoning"] == {"effort": "max", "enabled": True}
    assert probe["request"]["max_tokens"] == 8_000


def test_no_reasoning_after_probing_is_a_finding(tmp_path: Path) -> None:
    script = [
        *("plain", "final"),  # R1 run
        *("plain", "final"),  # the probe, still no reasoning
        *("tools", "final", "reject"),  # R2
    ]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    assert not script
    outcomes = _outcomes(checks)
    assert outcomes["r1-anthropic probe"] == "FINDING"
    assert outcomes["r1-anthropic edited_history"] == "FINDING"
    assert outcomes["cross cross_route"] == "FINDING"


def test_default_route_without_a_provider_is_a_failure(tmp_path: Path) -> None:
    # What the first live run showed: an account-level ZDR setting removed Anthropic.
    script = ["no_provider", "tools", "final", "reject"]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    outcomes = _outcomes(checks)
    assert outcomes["r1-anthropic run"] == "FAIL"
    assert outcomes["r2-bedrock-zdr run"] == "PASS"
    assert outcomes["cross cross_route"] == "FINDING"


def test_stale_recordings_are_removed(tmp_path: Path) -> None:
    stale = tmp_path / "r1-anthropic" / "probe-07.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    smoke.save_recordings([], [], tmp_path, SecretStr(FAKE_KEY))
    assert not stale.exists()
    assert (tmp_path / "report.json").exists()


def test_account_identifiers_are_scrubbed(tmp_path: Path) -> None:
    response: JsonValue = {"error": {"message": "x"}, "user_id": "user_3Kabc"}
    smoke.save_recordings(
        [smoke.Exchange("r1/run", {}, 401, response)], [], tmp_path, SecretStr(FAKE_KEY)
    )
    text = (tmp_path / "r1" / "run-01.json").read_text(encoding="utf-8")
    assert "user_3Kabc" not in text
    assert '"message": "x"' in text


def test_messages_routes_check_thinking_and_cross_route(tmp_path: Path) -> None:
    script = [
        *("tools", "final", "reject"),  # r1m: run with thinking, edited history rejected
        *("tools", "final", "reject"),  # r2m
        "reject",  # cross route, from r1m to Bedrock over the Messages endpoint
    ]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script),
        api_key=SecretStr(FAKE_KEY),
        out=tmp_path,
        routes=MESSAGES_ROUTES,
    )
    assert not script
    outcomes = _outcomes(checks)
    for route in MESSAGES_ROUTES:
        for name in ("run", "pin", "strict_args", "usage", "edited_history"):
            assert outcomes[f"{route} {name}"] == "PASS", f"{route} {name}"
    assert outcomes["cross cross_route"] == "PASS"
    cross = next(c for c in checks if c.name == "cross_route")
    assert "from r1m-anthropic" in cross.detail

    edited = json.loads(
        (tmp_path / "r1m-anthropic" / "edited_history-01.json").read_text(encoding="utf-8")
    )
    assistant = edited["request"]["messages"][1]
    assert assistant["content"][0]["thinking"] == "Look both up. (edited)"
    cross_request = json.loads(
        (tmp_path / "cross" / "cross_route-01.json").read_text(encoding="utf-8")
    )["request"]
    assert cross_request["provider"]["only"] == ["amazon-bedrock"]
    assert cross_request["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_cross_route_prefers_the_messages_conversation(tmp_path: Path) -> None:
    script = [
        *("tools", "final", "reject"),  # r1 (chat), with reasoning
        *("tools", "final", "reject"),  # r1m (messages), with thinking
        "reject",  # cross
    ]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script),
        api_key=SecretStr(FAKE_KEY),
        out=tmp_path,
        routes=("r1-anthropic", "r1m-anthropic"),
    )
    cross = next(c for c in checks if c.name == "cross_route")
    assert "from r1m-anthropic" in cross.detail


def test_signed_but_empty_thinking_is_tampered_by_signature() -> None:
    raw: dict[str, JsonValue] = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}],
    }
    tampered = smoke._tamper(raw)  # pyright: ignore[reportPrivateUsage]
    assert tampered is not None
    edited, what = tampered
    assert what == "reasoning signature changed"
    assert edited != raw
    assert raw["content"] == [{"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}]


def test_the_probe_runs_on_anthropic_routes_only(tmp_path: Path) -> None:
    script = [
        *("tools", "final", "reject"),  # r1: reasoning, edited history
        *("plain", "final"),  # r2: no reasoning, and no probe on Bedrock
        "reject",  # cross route
    ]
    checks = smoke.run_smoke(
        transport=_fake_openrouter(script), api_key=SecretStr(FAKE_KEY), out=tmp_path, routes=CHAT
    )
    assert not script
    outcomes = _outcomes(checks)
    assert outcomes["r2-bedrock-zdr probe"] == "INFO"
    assert outcomes["r2-bedrock-zdr edited_history"] == "FINDING"
    assert outcomes["cross cross_route"] == "PASS"
