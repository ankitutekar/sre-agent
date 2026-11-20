"""The agent harness: a small tool-use loop over a pinned model route (plan step 1.3, ADR 0008).

Import from here; the submodules are implementation detail.
"""

from sre_agent.harness.loop import FINAL_ANSWER_TOOL, Limits, RunResult, StopReason, run_agent
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
from sre_agent.harness.model_client import (
    MissingAPIKeyError,
    ModelClient,
    ModelError,
    ModelHTTPError,
    ModelResponseError,
    PinViolationError,
)
from sre_agent.harness.openrouter import (
    API_KEY_ENV,
    ROUTE_ANTHROPIC,
    ROUTE_BEDROCK_ZDR,
    OpenRouterClient,
    Route,
)
from sre_agent.harness.tools import LocalToolClient, Tool, ToolArgs, ToolClient, ToolError
from sre_agent.harness.trace import (
    ModelCallEvent,
    NoOpTrace,
    RecordingTrace,
    ToolCallEvent,
    TraceHook,
)

__all__ = [
    "API_KEY_ENV",
    "FINAL_ANSWER_TOOL",
    "ROUTE_ANTHROPIC",
    "ROUTE_BEDROCK_ZDR",
    "Limits",
    "LocalToolClient",
    "Message",
    "MissingAPIKeyError",
    "ModelCallEvent",
    "ModelClient",
    "ModelError",
    "ModelHTTPError",
    "ModelResponseError",
    "ModelTurn",
    "NoOpTrace",
    "OpenRouterClient",
    "PinViolationError",
    "RecordingTrace",
    "Route",
    "RunResult",
    "StopReason",
    "SystemMessage",
    "Tool",
    "ToolArgs",
    "ToolCall",
    "ToolCallEvent",
    "ToolClient",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "TraceHook",
    "Usage",
    "UserMessage",
    "run_agent",
]
