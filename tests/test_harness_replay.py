"""Replay of real OpenRouter responses, recorded by the live smoke test (step 1.3, slice C).

The replay transport serves the recorded responses in order, and checks that every
request the code sends today is identical to the recorded request. That proves,
offline and on every change, that history goes back byte for byte (append-only), and
that request bodies have not drifted since the last live run.

If these tests fail after a deliberate change to what goes over the wire (tool
schemas, request fields), re-run the smoke test live and commit the new recordings.

Recordings: tests/recordings/openrouter/, written by scripts/smoke_openrouter.py on
2026-11-21 (sixth live run). They hold bodies only: no headers, no key, no account id.
"""

import re
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, JsonValue, SecretStr, TypeAdapter

import smoke_openrouter as smoke
from sre_agent.harness import (
    ROUTE_ANTHROPIC,
    ROUTE_BEDROCK_ZDR,
    Message,
    ModelHTTPError,
    ModelTurn,
    OpenRouterBackend,
    OpenRouterClient,
    OpenRouterMessagesClient,
    Route,
    RunResult,
    StopReason,
    SystemMessage,
    UserMessage,
    Wire,
    offered_tools,
    run_agent,
)
from toy_agent import EXPECTED, PROBE_TASK, SYSTEM, TASK, Triage, toy_tools

RECORDINGS = Path(__file__).parent / "recordings" / "openrouter"
FAKE_KEY = "sk-or-v1-test-not-a-real-key-0123456789"
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class Recording(BaseModel):
    label: str
    request: JsonValue
    status: int
    response: JsonValue


def _load(label: str) -> list[Recording]:
    """The recordings for one label (e.g. "r1-anthropic/run"), in request order."""
    route, name = label.split("/")
    paths = sorted((RECORDINGS / route).glob(f"{name}-*.json"))
    assert paths, f"no recordings for {label}: re-run the smoke test live"
    return [Recording.model_validate_json(p.read_bytes()) for p in paths]


class ReplayTransport(httpx.BaseTransport):
    """Serves recorded responses in order; fails if a request differs from the recording."""

    def __init__(self, recordings: list[Recording]) -> None:
        self._queue = list(recordings)

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        assert self._queue, "more requests than recorded"
        recording = self._queue.pop(0)
        sent = _JSON.validate_json(request.content)
        assert sent == recording.request, f"request differs from the recording ({recording.label})"
        return httpx.Response(recording.status, json=recording.response)


def _client(route: Route, wire: Wire, transport: ReplayTransport) -> OpenRouterBackend:
    backend = OpenRouterClient if wire == "chat_completions" else OpenRouterMessagesClient
    http = httpx.Client(transport=transport)
    return backend(api_key=SecretStr(FAKE_KEY), route=route, http=http)


ROUTES = [
    ("r1-anthropic", ROUTE_ANTHROPIC, "chat_completions"),
    ("r2-bedrock-zdr", ROUTE_BEDROCK_ZDR, "chat_completions"),
    ("r1m-anthropic", ROUTE_ANTHROPIC, "messages"),
    ("r2m-bedrock-zdr", ROUTE_BEDROCK_ZDR, "messages"),
]
ANTHROPIC = [("r1-anthropic", "chat_completions"), ("r1m-anthropic", "messages")]


def _probe(name: str, wire: Wire) -> tuple[RunResult[Triage], Route]:
    """Replay the effort-max probe; returns its conversation and route."""
    route = ROUTE_ANTHROPIC.model_copy(update={"effort": smoke.PROBE_EFFORT})
    transport = ReplayTransport(_load(f"{name}/probe"))
    result = run_agent(
        model=_client(route, wire, transport),
        tools=toy_tools(),
        system=SYSTEM,
        task=PROBE_TASK,
        answer_model=Triage,
        limits=smoke.PROBE_LIMITS,
    )
    assert transport.remaining == 0
    return result, route


def _replay_once(route: Route, wire: Wire, label: str, messages: list[Message]) -> ModelTurn:
    transport = ReplayTransport(_load(label))
    client = _client(route, wire, transport)
    turn = client.complete(
        messages, offered_tools(toy_tools(), Triage), max_tokens=smoke.REPLAY_MAX_TOKENS
    )
    assert transport.remaining == 0
    return turn


@pytest.mark.parametrize(("name", "route", "wire"), ROUTES)
def test_toy_agent_replays_on_every_route(name: str, route: Route, wire: Wire) -> None:
    transport = ReplayTransport(_load(f"{name}/run"))
    result = run_agent(
        model=_client(route, wire, transport),
        tools=toy_tools(),
        system=SYSTEM,
        task=TASK,
        answer_model=Triage,
        limits=smoke.LIMITS,
    )
    assert result.ok, result.error
    assert result.answer == EXPECTED
    assert transport.remaining == 0  # every recorded exchange used, each request identical
    replies = [m for m in result.conversation if isinstance(m, ModelTurn)]
    served = "Anthropic" if route is ROUTE_ANTHROPIC else "Amazon Bedrock"
    assert {r.provider for r in replies} == {served}
    assert result.usage.cost is not None and result.usage.cost > 0


@pytest.mark.parametrize(("name", "wire"), ANTHROPIC)
def test_cut_off_thinking_is_kept_signed(name: str, wire: Wire) -> None:
    result, _ = _probe(name, wire)
    assert result.stop_reason is StopReason.LENGTH  # effort max spent the whole cap thinking
    prefix = smoke.reasoning_prefix(result.conversation)
    assert prefix is not None
    assert not prefix.turn.tool_calls
    items = [i for i in smoke.reasoning_items(prefix.turn.raw_message) if isinstance(i, dict)]
    assert items and all(isinstance(i.get("signature"), str) and i["signature"] for i in items)
    assert result.usage.reasoning_tokens == smoke.PROBE_LIMITS.max_tokens_per_request


@pytest.mark.parametrize(("name", "wire"), ANTHROPIC)
def test_control_and_edited_history_replay(name: str, wire: Wire) -> None:
    result, route = _probe(name, wire)
    prefix = smoke.reasoning_prefix(result.conversation)
    assert prefix is not None
    history = [*prefix.before, prefix.turn, *prefix.after]
    _replay_once(route, wire, f"{name}/control", history)

    tampered = smoke.tamper(prefix.turn.raw_message)
    assert tampered is not None
    edited, _ = tampered
    turn = prefix.turn.model_copy(update={"raw_message": edited})
    # Recorded finding (2026-11-21): the provider ACCEPTED an edited thinking summary.
    # If this starts raising, the provider began checking it: update ADR 0008.
    _replay_once(route, wire, f"{name}/edited_history", [*prefix.before, turn, *prefix.after])


def test_cross_route_fails_loudly() -> None:
    result, route = _probe("r1m-anthropic", "messages")
    prefix = smoke.reasoning_prefix(result.conversation)
    assert prefix is not None
    target = ROUTE_BEDROCK_ZDR.model_copy(update={"effort": route.effort})
    with pytest.raises(ModelHTTPError) as caught:
        _replay_once(
            target, "messages", "cross/cross_route", [*prefix.before, prefix.turn, *prefix.after]
        )
    assert caught.value.status_code == 400
    assert "encrypted reasoning" in str(caught.value)


# --- A permanent guard on what the recordings may contain ---

_KEY_LIKE = re.compile(r"sk-or-[A-Za-z0-9-]{8,}|Bearer\s+\S+", re.IGNORECASE)


def test_recordings_hold_no_secrets_or_account_ids() -> None:
    files = sorted(RECORDINGS.rglob("*.json"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not _KEY_LIKE.search(text), f"key-like string in {path.name}"
        assert '"user_id"' not in text, f"account id in {path.name}"


def test_a_changed_request_is_caught() -> None:
    # The identity check itself: one extra token of output room is a different request.
    transport = ReplayTransport(_load("r1-anthropic/run"))
    client = _client(ROUTE_ANTHROPIC, "chat_completions", transport)
    history: list[Message] = [SystemMessage(text=SYSTEM), UserMessage(text=TASK)]
    specs = offered_tools(toy_tools(), Triage)
    with pytest.raises(AssertionError, match="request differs"):
        client.complete(history, specs, max_tokens=smoke.LIMITS.max_tokens_per_request + 1)
