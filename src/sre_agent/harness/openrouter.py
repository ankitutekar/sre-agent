"""ModelClient backend for OpenRouter's Chat Completions API (ADR 0008).

Plain HTTP with httpx and our own typed models for the fields we use. Non-streaming,
no retries: a failure stops the run so we see it.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal, Self, assert_never

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, SecretStr, ValidationError

from sre_agent.harness.messages import (
    Message,
    ModelTurn,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    UserMessage,
    Wire,
)
from sre_agent.harness.model_client import (
    MissingAPIKeyError,
    ModelError,
    ModelHTTPError,
    ModelResponseError,
    PinViolationError,
)

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
SONNET_5_5 = "anthropic/claude-sonnet-5.5"
_MAX_ERROR_DETAIL = 300

type Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Route(BaseModel):
    """Which model, through which provider, under which data rules.

    `providers` are OpenRouter provider slugs, sent as `provider.only`.
    `served_as` are the names OpenRouter reports in the response's `provider`
    field; a response from anyone else is rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    providers: tuple[str, ...]
    served_as: tuple[str, ...]
    zdr: bool = False
    effort: Effort = "medium"
    # Ask for adaptive thinking explicitly (OpenRouter `reasoning.enabled`). With effort
    # alone, the first smoke test saw no reasoning at all on Bedrock (2026-11-20).
    thinking: bool = True
    # Messages endpoint only: "summarized" returns a readable summary of the thinking,
    # "omitted" returns empty (still signed) thinking blocks. Thinking happens either way.
    thinking_display: Literal["summarized", "omitted"] = "summarized"


# Default route R1: Anthropic's own endpoint, standard retention (accepted risk).
ROUTE_ANTHROPIC = Route(model=SONNET_5_5, providers=("anthropic",), served_as=("Anthropic",))
# Alternative route R2: zero data retention, Amazon Bedrock only (served_as confirmed
# by the step 1.3 smoke test, 2026-11-20).
ROUTE_BEDROCK_ZDR = Route(
    model=SONNET_5_5, providers=("amazon-bedrock",), served_as=("Amazon Bedrock",), zdr=True
)


# --- Wire models: the parts of OpenRouter's response we read. Extra fields are ignored.


class WireModel(BaseModel):
    """Base for parsing provider responses: fields we do not use are ignored."""

    model_config = ConfigDict(extra="ignore")


class _WireFunction(WireModel):
    name: str
    arguments: str


class _WireToolCall(WireModel):
    id: str
    function: _WireFunction


class _WireMessage(WireModel):
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[_WireToolCall] | None = None


class _WireChoice(WireModel):
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    message: dict[str, JsonValue]
    error: dict[str, JsonValue] | None = None


class _WirePromptDetails(WireModel):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None


class _WireCompletionDetails(WireModel):
    reasoning_tokens: int | None = None


class _WireUsage(WireModel):
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _WirePromptDetails | None = None
    completion_tokens_details: _WireCompletionDetails | None = None
    cost: float | None = None


class _WireResponse(WireModel):
    model: str
    provider: str | None = None
    choices: list[_WireChoice]
    usage: _WireUsage | None = None


class _WireErrorInfo(WireModel):
    message: str = ""


class _WireErrorBody(WireModel):
    error: _WireErrorInfo


class OpenRouterBackend(ABC):
    """What both OpenRouter backends share: the key, the HTTP call, the provider pin.

    Subclasses set `path` and `wire`, and build and parse their own wire format.
    ABC with @abstractmethod means Python refuses to create a subclass instance that
    forgot one of those methods.
    """

    path: ClassVar[str]
    wire: ClassVar[Wire]

    def __init__(
        self,
        *,
        api_key: SecretStr,
        route: Route,
        http: httpx.Client | None = None,
        timeout_s: float = 60.0,
        base_url: str = BASE_URL,
    ) -> None:
        if not api_key.get_secret_value():
            raise MissingAPIKeyError("the API key is empty")
        self.route = route
        self._api_key = api_key
        # Passing `http` in lets tests use httpx.MockTransport: no network, no key.
        self._http = http if http is not None else httpx.Client()
        self._timeout_s = timeout_s
        self._url = f"{base_url}{self.path}"

    @classmethod
    def from_env(
        cls,
        *,
        route: Route,
        http: httpx.Client | None = None,
        timeout_s: float = 60.0,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        """Build a client with the key from OPENROUTER_API_KEY (the only place it may come from)."""
        env = os.environ if environ is None else environ
        key = env.get(API_KEY_ENV, "")
        if not key:
            raise MissingAPIKeyError(f"{API_KEY_ENV} is not set")
        return cls(api_key=SecretStr(key), route=route, http=http, timeout_s=timeout_s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(route={self.route!r})"  # never the key

    @abstractmethod
    def build_body(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, max_tokens: int
    ) -> dict[str, JsonValue]:
        """The JSON request body. Public so tests (and later the gateway) can inspect it."""

    @abstractmethod
    def parse_response(self, content: bytes) -> ModelTurn:
        """Turn a 200 response body into a ModelTurn, checking the pin and the usage."""

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, max_tokens: int
    ) -> ModelTurn:
        for message in messages:
            if isinstance(message, ModelTurn) and message.wire != self.wire:
                raise ModelError(
                    f"history holds a {message.wire} turn; this backend speaks {self.wire}"
                )
        body = self.build_body(messages, tools, max_tokens=max_tokens)
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        try:
            response = self._http.post(
                self._url, json=body, headers=headers, timeout=self._timeout_s
            )
        except httpx.TimeoutException as exc:
            raise ModelError(f"request timed out after {self._timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"request failed ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise ModelHTTPError(response.status_code, error_detail(response.content))
        return self.parse_response(response.content)

    def provider_preferences(self) -> dict[str, JsonValue]:
        """The pin: only these providers, no fallbacks, no data collection."""
        provider: dict[str, JsonValue] = {
            "only": [*self.route.providers],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
        if self.route.zdr:
            provider["zdr"] = True
        return provider

    def check_pin(self, model: str, provider: str | None) -> None:
        if model != self.route.model:
            raise PinViolationError(f"served by model {model!r}, pinned to {self.route.model!r}")
        if provider not in self.route.served_as:
            raise PinViolationError(
                f"served by provider {provider!r}, pinned to {self.route.served_as}"
            )


class OpenRouterClient(OpenRouterBackend):
    """Sends conversations to OpenRouter Chat Completions on one pinned route."""

    path = "/chat/completions"
    wire = "chat_completions"

    def build_body(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, max_tokens: int
    ) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {
            "model": self.route.model,
            "messages": [_render_message(m) for m in messages],
            "max_tokens": max_tokens,
            "reasoning": _reasoning(self.route),
            "provider": self.provider_preferences(),
        }
        if tools:
            body["tools"] = [_render_tool(t) for t in tools]
            body["tool_choice"] = "auto"  # forced tool_choice is a 400 on Claude 5.5
        return body

    def parse_response(self, content: bytes) -> ModelTurn:
        """Turn a 200 response body into a ModelTurn, checking the pin and the usage."""
        try:
            wire = _WireResponse.model_validate_json(content)
        except ValidationError as exc:
            raise ModelResponseError(f"unexpected response shape: {error_detail(content)}") from exc
        if len(wire.choices) != 1:
            raise ModelResponseError(f"expected 1 choice, got {len(wire.choices)}")
        choice = wire.choices[0]
        if choice.error is not None:
            raise ModelResponseError(f"provider error in response: {choice.error}")
        self.check_pin(wire.model, wire.provider)
        if wire.usage is None:
            raise ModelResponseError("response has no usage, so the token budget cannot hold")
        try:
            message = _WireMessage.model_validate(choice.message)
        except ValidationError as exc:
            raise ModelResponseError("unexpected message shape") from exc
        return ModelTurn(
            text=message.content,
            refusal=message.refusal,
            tool_calls=tuple(
                ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
                for c in message.tool_calls or ()
            ),
            finish_reason=choice.finish_reason,
            native_finish_reason=choice.native_finish_reason,
            usage=_usage(wire.usage),
            model=wire.model,
            provider=wire.provider,
            wire=self.wire,
            raw_message=choice.message,
        )


def _render_message(message: Message) -> dict[str, JsonValue]:
    match message:
        case SystemMessage():
            return {"role": "system", "content": message.text}
        case UserMessage():
            return {"role": "user", "content": message.text}
        case ModelTurn():
            return message.raw_message  # replayed exactly as received
        case ToolResult():
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        case _:
            assert_never(message)  # pyright errors here if a Message type is not handled


def _reasoning(route: Route) -> dict[str, JsonValue]:
    reasoning: dict[str, JsonValue] = {"effort": route.effort}
    if route.thinking:
        reasoning["enabled"] = True
    return reasoning


def _render_tool(tool: ToolSpec) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        },
    }


def _usage(wire: _WireUsage) -> Usage:
    prompt = wire.prompt_tokens_details or _WirePromptDetails()
    completion = wire.completion_tokens_details or _WireCompletionDetails()
    return Usage(
        prompt_tokens=wire.prompt_tokens,
        completion_tokens=wire.completion_tokens,
        cached_tokens=prompt.cached_tokens or 0,
        cache_write_tokens=prompt.cache_write_tokens or 0,
        reasoning_tokens=completion.reasoning_tokens or 0,
        cost=wire.cost,
    )


def error_detail(content: bytes) -> str:
    """A short, printable reason from an error body (OpenRouter's `error.message` if present)."""
    try:
        detail = _WireErrorBody.model_validate_json(content).error.message
    except ValidationError:
        detail = content.decode("utf-8", errors="replace")
    return detail[:_MAX_ERROR_DETAIL]
