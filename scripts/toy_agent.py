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
# A task that needs reasoning before the first tool call, for the smoke test's
# edited-history and cross-route checks. Sonnet 5.5 thinks adaptively and did not think
# at all on TASK, or on a two-service variant, even at effort high (2026-11-20).
# Same expected answer: payments-api is the degraded service whose dependency is healthy.
PROBE_TASK = (
    "Users report checkout errors. Dependency map: checkout-web calls orders-api and "
    "payments-api; payments-api calls ledger-db; orders-api calls inventory-db. A "
    "degraded service is the root cause only if every service it calls is healthy. "
    "Plan which services to check and in what order before calling any tool, use as few "
    "tool calls as you can, then report the root-cause service, its status and its owner."
)

_STATUS = {
    "checkout-web": "degraded",
    "orders-api": "healthy",
    "payments-api": "degraded",
    "ledger-db": "healthy",
    "inventory-db": "healthy",
}
_OWNERS = {
    "checkout-web": "team-web",
    "orders-api": "team-orders",
    "payments-api": "team-payments",
    "ledger-db": "team-ledger",
    "inventory-db": "team-inventory",
}


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
