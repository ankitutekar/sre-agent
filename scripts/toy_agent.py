"""The toy agent for step 1.3: two read-only tools and a structured answer. Data is made up.

Shared by the live smoke test and the offline tests, so recorded responses replay
against exactly the tool definitions that produced them.
"""

from pydantic import BaseModel

from sre_agent.harness import LocalToolClient, Tool, ToolArgs, ToolError

SYSTEM = (
    "You are a triage assistant. Use the tools to look things up; do not guess. "
    "When you know the answer, call final_answer."
)
TASK = "What is the current status of payments-api, and which team owns it?"
# A variant that invites reasoning, for the smoke test's edited-history check: Sonnet 5.5
# thinks adaptively and often skips thinking on TASK. Same expected answer.
PROBE_TASK = (
    "Two services may be affected: payments-api and orders-api. Check both, work out "
    "which one needs attention and why, then report that service and its owning team."
)

_STATUS = {"payments-api": "degraded", "orders-api": "healthy"}
_OWNERS = {"payments-api": "team-payments", "orders-api": "team-orders"}


class ServiceArgs(ToolArgs):
    service: str


class Triage(BaseModel):
    """The toy agent's final answer."""

    service: str
    status: str
    owner: str


EXPECTED = Triage(service="payments-api", status="degraded", owner="team-payments")


def _status(args: ServiceArgs) -> str:
    if args.service not in _STATUS:
        raise ToolError(f"no such service {args.service!r}")
    return _STATUS[args.service]


def _owner(args: ServiceArgs) -> str:
    if args.service not in _OWNERS:
        raise ToolError(f"no such service {args.service!r}")
    return _OWNERS[args.service]


get_service_status = Tool(
    name="get_service_status",
    description="Current health of a service: healthy, degraded or down.",
    args_model=ServiceArgs,
    fn=_status,
)
get_owner = Tool(
    name="get_owner",
    description="The team that owns a service.",
    args_model=ServiceArgs,
    fn=_owner,
)


def toy_tools() -> LocalToolClient:
    return LocalToolClient([get_service_status, get_owner])
