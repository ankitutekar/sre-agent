"""ModelClient backend for OpenRouter's Anthropic-compatible Messages API (ADR 0008 fallback).

Added in step 1.3 slice E: through Chat Completions, Claude Sonnet 5.5 returned no
reasoning at all, so thinking could be neither seen nor checked on replay. Here the
assistant's content blocks, including signed `thinking` blocks, come back as they
are and are sent back unchanged.

Differences from Chat Completions that matter to the loop:
- the system prompt is a top-level field, not a message;
- tool results go back as `tool_result` blocks, all results of one turn in a single
  user message (Anthropic's rule);
- `stop_reason` is mapped onto the loop's finish reasons, and kept as the native one.
"""

import json
from collections.abc import Sequence
from typing import assert_never

from pydantic import JsonValue, ValidationError

from sre_agent.harness.messages import (
    Message,
    ModelTurn,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    UserMessage,
)
from sre_agent.harness.model_client import ModelResponseError
from sre_agent.harness.openrouter import OpenRouterBackend, WireModel, error_detail

# Anthropic stop_reason -> the loop's finish_reason. Anything else (pause_turn, ...) is
# passed through unchanged, and the loop stops on it as unhandled.
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class _MStopDetails(WireModel):
    category: str | None = None
    explanation: str | None = None


class _MUsage(WireModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cost: float | None = None


class _MResponse(WireModel):
    model: str
    provider: str | None = None
    content: list[dict[str, JsonValue]]
    stop_reason: str | None = None
    stop_details: _MStopDetails | None = None
    usage: _MUsage | None = None


class _MToolUse(WireModel):
    id: str
    name: str
    input: dict[str, JsonValue]


class OpenRouterMessagesClient(OpenRouterBackend):
    """Sends conversations to OpenRouter's Messages endpoint on one pinned route."""

    path = "/messages"
    wire = "messages"

    def build_body(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, max_tokens: int
    ) -> dict[str, JsonValue]:
        system, rest = _split_system(messages)
        body: dict[str, JsonValue] = {
            "model": self.route.model,
            "max_tokens": max_tokens,
            "messages": _render_messages(rest),
            "output_config": {"effort": self.route.effort},
            "provider": self.provider_preferences(),
        }
        if system:
            body["system"] = system
        if self.route.thinking:
            body["thinking"] = {"type": "adaptive", "display": self.route.thinking_display}
        if tools:
            body["tools"] = [_render_tool(t) for t in tools]
            body["tool_choice"] = {"type": "auto"}  # forced tool_choice is a 400 on Claude 5.5
        return body

    def parse_response(self, content: bytes) -> ModelTurn:
        try:
            wire = _MResponse.model_validate_json(content)
        except ValidationError as exc:
            raise ModelResponseError(f"unexpected response shape: {error_detail(content)}") from exc
        self.check_pin(wire.model, wire.provider)
        if wire.usage is None:
            raise ModelResponseError("response has no usage, so the token budget cannot hold")

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in wire.content:
            kind = block.get("type")
            if kind == "text":
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif kind == "tool_use":
                try:
                    use = _MToolUse.model_validate(block)
                except ValidationError as exc:
                    raise ModelResponseError("unexpected tool_use block") from exc
                # The loop and tools work with JSON text, as on Chat Completions.
                calls.append(ToolCall(id=use.id, name=use.name, arguments=json.dumps(use.input)))
            # thinking / redacted_thinking: kept in raw_message, not shown to the loop.

        stop = wire.stop_reason
        refusal = None
        if stop == "refusal":
            details = wire.stop_details or _MStopDetails()
            refusal = details.explanation or f"refused (category {details.category!r})"
        return ModelTurn(
            text="".join(texts) or None,
            refusal=refusal,
            tool_calls=tuple(calls),
            finish_reason=_FINISH.get(stop, stop) if stop is not None else None,
            native_finish_reason=stop,
            usage=_usage(wire.usage),
            model=wire.model,
            provider=wire.provider,
            wire=self.wire,
            raw_message={"role": "assistant", "content": [*wire.content]},
        )


def _split_system(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    """The leading system messages become the top-level `system` field."""
    i = 0
    parts: list[str] = []
    while i < len(messages):
        message = messages[i]
        if not isinstance(message, SystemMessage):
            break
        parts.append(message.text)
        i += 1
    rest = list(messages[i:])
    if any(isinstance(m, SystemMessage) for m in rest):
        raise ValueError("a system message after the conversation started is not supported")
    return "\n\n".join(parts), rest


def _render_messages(messages: Sequence[Message]) -> list[JsonValue]:
    out: list[JsonValue] = []
    pending: list[JsonValue] = []  # tool_result blocks of the current turn

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": [*pending]})
            pending.clear()

    for message in messages:
        match message:
            case ToolResult():
                pending.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                        "is_error": message.is_error,
                    }
                )
            case UserMessage():
                flush()
                out.append({"role": "user", "content": message.text})
            case ModelTurn():
                flush()
                out.append(message.raw_message)  # replayed exactly as received
            case SystemMessage():
                raise ValueError("system messages are sent as the top-level field")
            case _:
                assert_never(message)
    flush()
    return out


def _render_tool(tool: ToolSpec) -> dict[str, JsonValue]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
        "strict": True,
    }


def _usage(wire: _MUsage) -> Usage:
    # Anthropic's input_tokens excludes cached tokens; the prompt total includes them.
    cache_write = wire.cache_creation_input_tokens or 0
    cache_read = wire.cache_read_input_tokens or 0
    return Usage(
        prompt_tokens=wire.input_tokens + cache_write + cache_read,
        completion_tokens=wire.output_tokens,
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost=wire.cost,
    )
