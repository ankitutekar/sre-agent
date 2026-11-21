"""Messages backend: Anthropic-format bodies, thinking replay, stop reasons, wire separation.

No network: httpx.MockTransport. Responses are hand-written in the shape OpenRouter's
Messages endpoint documents; recorded ones replace them after the live smoke test.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import JsonValue, SecretStr, TypeAdapter

from harness_fakes import Triage, get_owner, toy_tools, turn
from sre_agent.harness import (
    ROUTE_ANTHROPIC,
    ModelError,
    OpenRouterClient,
    OpenRouterMessagesClient,
    PinViolationError,
    StopReason,
    SystemMessage,
    ToolResult,
    UserMessage,
    run_agent,
)
from toy_agent import EXPECTED, SYSTEM, TASK

FAKE_KEY = "sk-or-v1-test-not-a-real-key-0123456789"
MESSAGES = [SystemMessage(text="sys"), UserMessage(text="task")]
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

THINKING: dict[str, JsonValue] = {
    "type": "thinking",
    "thinking": "Check status and owner.",
    "signature": "c2lnbmF0dXJl",
}


def _tool_use(id_: str, name: str, service: str = "payments-api") -> dict[str, JsonValue]:
    return {"type": "tool_use", "id": id_, "name": name, "input": {"service": service}}


def _response(
    content: list[JsonValue],
    *,
    stop: str = "tool_use",
    provider: str = "Anthropic",
    stop_details: JsonValue = None,
) -> dict[str, JsonValue]:
    return {
        "id": "gen-1",
        "type": "message",
        "role": "assistant",
        "model": "anthropic/claude-sonnet-5.5",
        "provider": provider,
        "content": content,
        "stop_reason": stop,
        "stop_details": stop_details,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
            "cost": 0.0007,
        },
    }


Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> OpenRouterMessagesClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenRouterMessagesClient(api_key=SecretStr(FAKE_KEY), route=ROUTE_ANTHROPIC, http=http)


def _ok(body: dict[str, JsonValue]) -> Handler:
    return lambda request: httpx.Response(200, json=body)


TWO_CALLS = _response(
    [THINKING, _tool_use("toolu_1", "get_service_status"), _tool_use("toolu_2", "get_owner")]
)


# --- Request body ---


def test_body_uses_the_anthropic_format_and_the_pin() -> None:
    body = _client(_ok(TWO_CALLS)).build_body(MESSAGES, [get_owner.spec()], max_tokens=4000)
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "task"}]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["tool_choice"] == {"type": "auto"}
    assert body["tools"] == [
        {
            "name": "get_owner",
            "description": "The team that owns a service.",
            "input_schema": get_owner.spec().parameters,
            "strict": True,
        }
    ]
    assert body["provider"] == {
        "only": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert "reasoning" not in body  # that is the Chat Completions field


def test_requests_go_to_the_messages_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TWO_CALLS)

    _client(handler).complete(MESSAGES, [], max_tokens=10)
    assert seen[0].url.path == "/api/v1/messages"
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


def test_thinking_is_replayed_and_results_are_grouped() -> None:
    client = _client(_ok(TWO_CALLS))
    reply = client.complete(MESSAGES, [], max_tokens=10)
    results = [
        ToolResult(tool_call_id="toolu_1", name="get_service_status", content="degraded"),
        ToolResult(tool_call_id="toolu_2", name="get_owner", content="nope", is_error=True),
    ]
    body = client.build_body([*MESSAGES, reply, *results], [], max_tokens=10)
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {"role": "assistant", "content": TWO_CALLS["content"]}
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "degraded",
                "is_error": False,
            },
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "nope", "is_error": True},
        ],
    }
    assert len(messages) == 3  # both results in one user message


def test_a_late_system_message_is_refused() -> None:
    with pytest.raises(ValueError, match="system message"):
        _client(_ok(TWO_CALLS)).build_body(
            [*MESSAGES, SystemMessage(text="late")], [], max_tokens=10
        )


# --- Response parsing ---


def test_response_is_parsed() -> None:
    reply = _client(_ok(TWO_CALLS)).complete(MESSAGES, [], max_tokens=10)
    assert reply.wire == "messages"
    assert reply.finish_reason == "tool_calls"
    assert reply.native_finish_reason == "tool_use"
    assert [(c.id, c.name) for c in reply.tool_calls] == [
        ("toolu_1", "get_service_status"),
        ("toolu_2", "get_owner"),
    ]
    assert json.loads(reply.tool_calls[0].arguments) == {"service": "payments-api"}
    assert reply.text is None
    assert reply.usage.prompt_tokens == 150  # input + cache write + cache read
    assert reply.usage.cached_tokens == 30
    assert reply.usage.cache_write_tokens == 20
    assert reply.usage.cost == pytest.approx(0.0007)


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "pause_turn"),  # unhandled: the loop stops on it
    ],
)
def test_stop_reasons_are_mapped(stop: str, finish: str) -> None:
    body = _response([{"type": "text", "text": "x"}], stop=stop)
    reply = _client(_ok(body)).complete(MESSAGES, [], max_tokens=10)
    assert reply.finish_reason == finish
    assert reply.native_finish_reason == stop


def test_refusal_carries_its_explanation() -> None:
    body = _response(
        [], stop="refusal", stop_details={"category": "cyber", "explanation": "Declined."}
    )
    reply = _client(_ok(body)).complete(MESSAGES, [], max_tokens=10)
    assert reply.refusal == "Declined."


def test_pin_violation_fails_closed() -> None:
    with pytest.raises(PinViolationError):
        _client(_ok(_response([], provider="Google"))).complete(MESSAGES, [], max_tokens=10)


# --- Wire formats never mix ---


def test_a_chat_completions_turn_is_refused_by_the_messages_backend() -> None:
    chat_turn = turn(calls=[("get_owner", '{"service": "a"}')])  # wire="chat_completions"
    with pytest.raises(ModelError, match="chat_completions turn"):
        _client(_ok(TWO_CALLS)).complete([*MESSAGES, chat_turn], [], max_tokens=10)


def test_a_messages_turn_is_refused_by_the_chat_completions_backend() -> None:
    messages_turn = _client(_ok(TWO_CALLS)).complete(MESSAGES, [], max_tokens=10)
    chat = OpenRouterClient(
        api_key=SecretStr(FAKE_KEY),
        route=ROUTE_ANTHROPIC,
        http=httpx.Client(transport=httpx.MockTransport(_ok({}))),
    )
    with pytest.raises(ModelError, match="messages turn"):
        chat.complete([*MESSAGES, messages_turn], [], max_tokens=10)


# --- The whole loop over the Messages backend ---


def test_toy_agent_completes_over_the_messages_backend() -> None:
    final = _response(
        [
            THINKING,
            {
                "type": "tool_use",
                "id": "toolu_3",
                "name": "final_answer",
                "input": EXPECTED.model_dump(),
            },
        ]
    )
    replies = [TWO_CALLS, final]
    requests: list[JsonValue] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(_JSON.validate_json(request.content))
        return httpx.Response(200, json=replies.pop(0))

    result = run_agent(
        model=_client(handler),
        tools=toy_tools(),
        system=SYSTEM,
        task=TASK,
        answer_model=Triage,
    )
    assert result.ok, result.error
    assert result.stop_reason is StopReason.COMPLETED
    assert result.answer == EXPECTED
    second = requests[1]
    assert isinstance(second, dict)
    history = second["messages"]
    assert isinstance(history, list)
    assert history[1] == {"role": "assistant", "content": TWO_CALLS["content"]}  # thinking kept
    assert history[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "degraded",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "team-payments",
                "is_error": False,
            },
        ],
    }
