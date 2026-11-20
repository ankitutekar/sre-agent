"""Tools: how they are defined, and the interface the loop uses to call them.

The loop only ever calls tools through a ToolClient. Today that is LocalToolClient,
which runs Python functions in-process; later the gateway sits behind the same
interface and decides which calls are allowed (Phase 3).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from sre_agent.harness.messages import ToolCall, ToolResult, ToolSpec

_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolArgs(BaseModel):
    """Base for tool argument models: unknown fields are rejected, values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolError(Exception):
    """Raised by a tool for a failure the model may see, and may be able to fix.

    Any other exception is treated as a bug: the model only learns its type, not its
    message, so internal details do not leak into the conversation.
    """


class RunnableTool(Protocol):
    """What LocalToolClient needs from a tool (Tool below is the usual one)."""

    @property
    def name(self) -> str: ...

    def spec(self) -> ToolSpec: ...

    def invoke(self, arguments: str) -> str: ...


@dataclass(frozen=True)
class Tool[A: ToolArgs]:
    """A tool: its arguments are a ToolArgs model, its function returns text or a model.

    `Tool[A: ToolArgs]` is Python 3.12 generic syntax: A stands for the tool's own
    argument class, so `fn` is type-checked against it.
    """

    name: str
    description: str
    args_model: type[A]
    fn: Callable[[A], str | BaseModel]

    def __post_init__(self) -> None:
        if not _TOOL_NAME.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}: use 1-64 of [a-zA-Z0-9_-]")

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    def invoke(self, arguments: str) -> str:
        """Validate the model's JSON arguments, then run. Raises ValidationError if invalid."""
        args = self.args_model.model_validate_json(arguments)
        out = self.fn(args)
        return out if isinstance(out, str) else out.model_dump_json()


class ToolClient(Protocol):
    """Anything that can list tools and run one call. Must not raise for a bad call."""

    def list_tools(self) -> Sequence[ToolSpec]: ...

    def call(self, call: ToolCall) -> ToolResult: ...


class LocalToolClient:
    """Runs tools in-process. Every failure becomes an error result the model can read."""

    def __init__(self, tools: Sequence[RunnableTool]) -> None:
        self._tools: dict[str, RunnableTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def list_tools(self) -> Sequence[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def call(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return error_result(call, f"unknown tool {call.name!r}")
        try:
            content = tool.invoke(call.arguments)
        except ValidationError as exc:
            return error_result(call, f"invalid arguments: {validation_summary(exc)}")
        except ToolError as exc:
            return error_result(call, str(exc))
        except Exception as exc:  # a bug in the tool: report the type only
            return error_result(call, f"tool failed ({type(exc).__name__})")
        return ToolResult(tool_call_id=call.id, name=call.name, content=content)


def error_result(call: ToolCall, reason: str) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id, name=call.name, content=f"error: {reason}", is_error=True
    )


def validation_summary(exc: ValidationError) -> str:
    """Short, model-readable list of what was wrong, without echoing the input back."""
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
        for err in exc.errors(include_input=False, include_url=False)
    ]
    return "; ".join(parts)
