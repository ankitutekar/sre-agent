"""OpenRouter backend: request body, response parsing, pin checks, errors and key handling.

No network: httpx.MockTransport hands each request to a function that returns a
canned response. Responses here are hand-written; recorded ones arrive with the
smoke test (step 1.3, slice D).
"""

import json
import logging
from collections.abc import Callable

import httpx
import pytest
from pydantic import JsonValue, SecretStr

from harness_fakes import get_owner
from sre_agent.harness import (
    API_KEY_ENV,
    ROUTE_ANTHROPIC,
    ROUTE_BEDROCK_ZDR,
    MissingAPIKeyError,
    ModelError,
    ModelHTTPError,
    ModelResponseError,
    OpenRouterClient,
    PinViolationError,
    Route,
    SystemMessage,
    ToolResult,
    UserMessage,
)

FAKE_KEY = "sk-or-v1-test-not-a-real-key-0123456789"
MESSAGES = [SystemMessage(text="sys"), UserMessage(text="task")]
REASONING: list[JsonValue] = [
    {
        "type": "reasoning.text",
        "text": "Check status first.",
        "signature": "c2lnbmF0dXJl",
        "id": "r1",
    }
]


def _response(
    *,
    model: str = "anthropic/claude-sonnet-5.5",
    provider: str | None = "Anthropic",
    finish: str = "tool_calls",
    native: str = "tool_use",
    usage: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    message: dict[str, JsonValue] = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "reasoning_details": REASONING,
        "tool_calls": [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "get_owner", "arguments": '{"service": "payments-api"}'},
            }
        ],
    }
    body: dict[str, JsonValue] = {
        "id": "gen-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "native_finish_reason": native,
                "message": message,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "cost": 0.00054,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }
    if provider is not None:
        body["provider"] = provider
    return body


Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, route: Route = ROUTE_ANTHROPIC) -> OpenRouterClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenRouterClient(api_key=SecretStr(FAKE_KEY), route=route, http=http)


def _ok(body: dict[str, JsonValue]) -> Handler:
    return lambda request: httpx.Response(200, json=body)


# --- Request body ---


def test_body_pins_the_provider_and_denies_data_collection() -> None:
    body = _client(_ok(_response())).build_body(MESSAGES, [get_owner.spec()], max_tokens=4000)
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert body["provider"] == {
        "only": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert body["reasoning"] == {"effort": "medium", "enabled": True}
    assert body["max_tokens"] == 4000
    assert body["tool_choice"] == "auto"
    assert "models" not in body  # never a fallback model list


def test_thinking_can_be_left_to_the_default() -> None:
    route = ROUTE_ANTHROPIC.model_copy(update={"thinking": False})
    body = _client(_ok(_response()), route).build_body(MESSAGES, [], max_tokens=10)
    assert body["reasoning"] == {"effort": "medium"}


def test_zdr_route_asks_for_zero_data_retention_on_bedrock() -> None:
    body = _client(_ok(_response()), ROUTE_BEDROCK_ZDR).build_body(MESSAGES, [], max_tokens=10)
    provider = body["provider"]
    assert isinstance(provider, dict)
    assert provider["only"] == ["amazon-bedrock"]
    assert provider["zdr"] is True
    assert "tools" not in body and "tool_choice" not in body


def test_tools_are_sent_strict() -> None:
    body = _client(_ok(_response())).build_body(MESSAGES, [get_owner.spec()], max_tokens=10)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_owner",
                "description": "The team that owns a service.",
                "parameters": get_owner.spec().parameters,
                "strict": True,
            },
        }
    ]


def test_assistant_turn_is_replayed_byte_for_byte() -> None:
    client = _client(_ok(_response()))
    reply = client.complete(MESSAGES, [], max_tokens=10)
    result = ToolResult(tool_call_id="toolu_01", name="get_owner", content="team-payments")
    body = client.build_body([*MESSAGES, reply, result], [], max_tokens=10)
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[2] == _response()["choices"][0]["message"]  # type: ignore[index]
    assert messages[3] == {"role": "tool", "tool_call_id": "toolu_01", "content": "team-payments"}


# --- Response parsing and the pin ---


def test_response_is_parsed() -> None:
    reply = _client(_ok(_response())).complete(MESSAGES, [], max_tokens=10)
    assert reply.finish_reason == "tool_calls"
    assert reply.native_finish_reason == "tool_use"
    assert [(c.id, c.name) for c in reply.tool_calls] == [("toolu_01", "get_owner")]
    assert reply.raw_message["reasoning_details"] == REASONING
    assert reply.usage.total_tokens == 150
    assert reply.usage.reasoning_tokens == 12
    assert reply.usage.cost == pytest.approx(0.00054)
    assert reply.provider == "Anthropic"


@pytest.mark.parametrize(
    "body",
    [
        _response(provider="Amazon Bedrock"),
        _response(provider=None),
        _response(model="anthropic/claude-haiku-5.5"),
    ],
)
def test_pin_violation_fails_closed(body: dict[str, JsonValue]) -> None:
    with pytest.raises(PinViolationError):
        _client(_ok(body)).complete(MESSAGES, [], max_tokens=10)


def test_missing_usage_is_an_error() -> None:
    body = _response()
    del body["usage"]
    with pytest.raises(ModelResponseError, match="no usage"):
        _client(_ok(body)).complete(MESSAGES, [], max_tokens=10)


def test_error_inside_a_200_is_an_error() -> None:
    body: dict[str, JsonValue] = {"error": {"code": 502, "message": "upstream failed"}}
    with pytest.raises(ModelResponseError, match="upstream failed"):
        _client(_ok(body)).complete(MESSAGES, [], max_tokens=10)


# --- HTTP and transport errors ---


@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 503])
def test_http_errors_are_typed(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": status, "message": "nope"}})

    with pytest.raises(ModelHTTPError) as caught:
        _client(handler).complete(MESSAGES, [], max_tokens=10)
    assert caught.value.status_code == status
    assert str(caught.value) == f"HTTP {status}: nope"


def test_timeout_is_a_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelError, match="timed out"):
        _client(handler).complete(MESSAGES, [], max_tokens=10)


# --- The API key ---


def test_key_comes_from_the_environment_only() -> None:
    with pytest.raises(MissingAPIKeyError, match=API_KEY_ENV):
        OpenRouterClient.from_env(route=ROUTE_ANTHROPIC, environ={})
    client = OpenRouterClient.from_env(route=ROUTE_ANTHROPIC, environ={API_KEY_ENV: FAKE_KEY})
    assert FAKE_KEY not in repr(client)


def test_key_is_sent_only_in_the_auth_header(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    caplog.set_level(logging.DEBUG)  # httpx and httpcore log at DEBUG
    with pytest.raises(ModelHTTPError) as caught:
        _client(handler).complete(MESSAGES, [], max_tokens=10)

    request = seen[0]
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in request.content.decode()
    assert FAKE_KEY not in str(request.url)
    assert FAKE_KEY not in str(caught.value)
    assert FAKE_KEY not in caplog.text
    assert json.loads(request.content)["model"] == "anthropic/claude-sonnet-5.5"
