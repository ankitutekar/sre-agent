"""Live smoke test of the model path (plan step 1.3, slice D; ADR 0008).

Run by the maintainer, in their own terminal, with OPENROUTER_API_KEY set:

    uv run python scripts/smoke_openrouter.py --live

Without --live it only explains what it would do: no request, no key needed.
About 10-15 requests on Claude Sonnet 5.5, a few cents in total.

For each route (R1 Anthropic, R2 Amazon Bedrock with ZDR) it checks:
  run             the toy agent finishes a two-tool task
  pin             every reply came from the pinned model and provider
  strict_args     the model's tool arguments always passed validation
  usage           cost and cache-token counts are present
  finish_reasons  finish_reason / native_finish_reason pairs (recorded)
  probe           only if the run had no reasoning: the PROBE_TASK at effort high
  edited_history  a turn sent back with edited reasoning: rejected, or silently accepted?
And once:
  cross_route     a conversation started on R1, continued on R2: must fail loudly

Outcomes: PASS, FINDING (unexpected but informative, e.g. "no eligible provider"
on R2), FAIL (the harness or the route is broken), INFO (recorded only).

Request and response bodies are saved to tests/recordings/openrouter/ for the
offline tests. Headers are never saved, so the key cannot be in them; every file
is also checked for the key before it is written.
"""

import argparse
import copy
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from sre_agent.harness import (
    API_KEY_ENV,
    ROUTE_ANTHROPIC,
    ROUTE_BEDROCK_ZDR,
    Limits,
    Message,
    ModelError,
    ModelTurn,
    OpenRouterClient,
    Route,
    RunResult,
    ToolResult,
    offered_tools,
    run_agent,
)
from toy_agent import EXPECTED, PROBE_TASK, SYSTEM, TASK, Triage, toy_tools

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "recordings" / "openrouter"
LIMITS = Limits(max_turns=6, max_total_tokens=30_000)
PROBE_ATTEMPTS = 2
ROUTES: dict[str, Route] = {"r1-anthropic": ROUTE_ANTHROPIC, "r2-bedrock-zdr": ROUTE_BEDROCK_ZDR}

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

type Outcome = Literal["PASS", "FINDING", "FAIL", "INFO"]


@dataclass(frozen=True)
class Check:
    route: str
    name: str
    outcome: Outcome
    detail: str


@dataclass(frozen=True)
class Exchange:
    label: str
    request: JsonValue
    status: int
    response: JsonValue


class RecordingTransport(httpx.BaseTransport):
    """Wraps the real transport and keeps each request and response body (never headers)."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner
        self.label = "unlabelled"
        self.exchanges: list[Exchange] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        response.read()
        self.exchanges.append(
            Exchange(
                self.label, _parse(request.content), response.status_code, _parse(response.content)
            )
        )
        return response


@dataclass
class Smoke:
    transport: RecordingTransport
    api_key: SecretStr
    checks: list[Check] = field(default_factory=lambda: [])

    def client(self, route: Route) -> OpenRouterClient:
        http = httpx.Client(transport=self.transport)
        return OpenRouterClient(api_key=self.api_key, route=route, http=http)

    def record(self, route: str, name: str, outcome: Outcome, detail: str) -> None:
        self.checks.append(Check(route, name, outcome, detail))

    def run_route(self, name: str, route: Route) -> RunResult[Triage]:
        self.transport.label = f"{name}/run"
        first_exchange = len(self.transport.exchanges)
        result = run_agent(
            model=self.client(route),
            tools=toy_tools(),
            system=SYSTEM,
            task=TASK,
            answer_model=Triage,
            limits=LIMITS,
        )
        exchanges = self.transport.exchanges[first_exchange:]
        replies = [m for m in result.conversation if isinstance(m, ModelTurn)]

        if result.ok and result.answer == EXPECTED:
            called = sorted({c.name for r in replies for c in r.tool_calls} - {"final_answer"})
            self.record(name, "run", "PASS", f"{result.turns} turns, tools {called}")
        elif result.ok:
            self.record(name, "run", "FAIL", f"wrong answer {result.answer!r}")
        else:
            # "No eligible provider" is a finding on the ZDR route; the default route must work.
            no_provider = route.zdr and _no_provider(result.error)
            outcome: Outcome = "FINDING" if no_provider else "FAIL"
            self.record(name, "run", outcome, f"{result.stop_reason}: {result.error}")

        served = sorted({f"{r.model} via {r.provider}" for r in replies})
        if replies:
            self.record(name, "pin", "PASS", ", ".join(served))
        elif result.error and "served by" in result.error:
            self.record(name, "pin", "FAIL", result.error)

        invalid = [
            m.content
            for m in result.conversation
            if isinstance(m, ToolResult) and "invalid arguments" in m.content
        ]
        if replies:
            self.record(
                name,
                "strict_args",
                "FINDING" if invalid else "PASS",
                "; ".join(invalid) if invalid else "all tool arguments valid",
            )
            self.record(name, "usage", *_usage_check(exchanges))
            pairs = [f"{r.finish_reason}/{r.native_finish_reason}" for r in replies]
            self.record(name, "finish_reasons", "INFO", ", ".join(pairs))
            cost = result.usage.cost
            self.record(
                name,
                "cost",
                "INFO",
                f"{result.usage.total_tokens} tokens, "
                + (f"${cost:.5f}" if cost is not None else "cost not reported"),
            )
        return result

    def reasoning_source(
        self, name: str, route: Route, result: RunResult[Triage]
    ) -> tuple[RunResult[Triage], Route] | None:
        """A finished run with a reasoning turn, and the route that produced it.

        Sonnet 5.5 thinks adaptively, so the main run may have none. Then run PROBE_TASK
        at effort high, up to PROBE_ATTEMPTS times.
        """
        if _prefix_after_first_tool_turn(result.conversation) is not None:
            return result, route
        probe_route = route.model_copy(update={"effort": "high"})
        for attempt in range(1, PROBE_ATTEMPTS + 1):
            self.transport.label = f"{name}/probe"
            probe = run_agent(
                model=self.client(probe_route),
                tools=toy_tools(),
                system=SYSTEM,
                task=PROBE_TASK,
                answer_model=Triage,
                limits=LIMITS,
            )
            if probe.ok and _prefix_after_first_tool_turn(probe.conversation) is not None:
                self.record(
                    name,
                    "probe",
                    "INFO",
                    f"reasoning on attempt {attempt} at effort high "
                    f"({probe.usage.reasoning_tokens} reasoning tokens)",
                )
                return probe, probe_route
        self.record(
            name, "probe", "FINDING", f"no reasoning in {PROBE_ATTEMPTS} attempts at effort high"
        )
        return None

    def edited_history(self, name: str, route: Route, result: RunResult[Triage]) -> None:
        prefix = _prefix_after_first_tool_turn(result.conversation)
        if prefix is None:
            self.record(name, "edited_history", "FINDING", "skipped: no turn with reasoning")
            return
        tampered = _tamper(prefix.turn.raw_message)
        if tampered is None:
            self.record(
                name, "edited_history", "FINDING", "skipped: reasoning has no editable text"
            )
            return
        edited, what = tampered
        messages: list[Message] = [
            *prefix.before,
            prefix.turn.model_copy(update={"raw_message": edited}),
            *prefix.results,
        ]
        self.transport.label = f"{name}/edited_history"
        try:
            self.client(route).complete(
                messages,
                offered_tools(toy_tools(), Triage),
                max_tokens=LIMITS.max_tokens_per_request,
            )
        except ModelError as exc:
            self.record(name, "edited_history", "PASS", f"rejected loudly ({what}): {exc}")
            return
        self.record(name, "edited_history", "FINDING", f"accepted silently ({what})")

    def cross_route(self, r1_result: RunResult[Triage], r1_route: Route) -> None:
        prefix = _prefix_after_first_tool_turn(r1_result.conversation)
        if prefix is None:
            self.record("cross", "cross_route", "FINDING", "skipped: R1 produced no reasoning turn")
            return
        # Same effort as the R1 conversation, so the only difference is the provider.
        target = ROUTE_BEDROCK_ZDR.model_copy(update={"effort": r1_route.effort})
        self.transport.label = "cross/cross_route"
        try:
            self.client(target).complete(
                [*prefix.before, prefix.turn, *prefix.results],
                offered_tools(toy_tools(), Triage),
                max_tokens=LIMITS.max_tokens_per_request,
            )
        except ModelError as exc:
            self.record("cross", "cross_route", "PASS", f"failed loudly: {exc}")
            return
        self.record("cross", "cross_route", "FINDING", "R2 accepted R1's reasoning without error")


def run_smoke(*, transport: httpx.BaseTransport, api_key: SecretStr, out: Path) -> list[Check]:
    smoke = Smoke(RecordingTransport(transport), api_key)
    sources: dict[str, tuple[RunResult[Triage], Route]] = {}
    for name, route in ROUTES.items():
        result = smoke.run_route(name, route)
        if not result.ok:
            continue
        source = smoke.reasoning_source(name, route, result)
        if source is None:
            smoke.record(name, "edited_history", "FINDING", "skipped: no turn with reasoning")
            continue
        sources[name] = source
        smoke.edited_history(name, source[1], source[0])
    if "r1-anthropic" in sources:
        smoke.cross_route(*sources["r1-anthropic"])
    else:
        smoke.record("cross", "cross_route", "FINDING", "skipped: no R1 run with reasoning")
    save_recordings(smoke.transport.exchanges, smoke.checks, out, api_key)
    return smoke.checks


def save_recordings(
    exchanges: Sequence[Exchange], checks: Sequence[Check], out: Path, api_key: SecretStr
) -> None:
    """Write one JSON file per exchange plus report.json. Refuses any file containing the key."""
    files: dict[Path, str] = {}
    counters: dict[str, int] = {}
    for ex in exchanges:
        n = counters[ex.label] = counters.get(ex.label, 0) + 1
        body = {
            "label": ex.label,
            "request": ex.request,
            "status": ex.status,
            "response": ex.response,
        }
        files[out / f"{ex.label}-{n:02}.json"] = json.dumps(body, indent=2) + "\n"
    report = [vars(c) for c in checks]
    files[out / "report.json"] = json.dumps(report, indent=2) + "\n"

    secret = api_key.get_secret_value()
    leaking = [str(path) for path, text in files.items() if secret in text]
    if leaking:  # should be impossible: headers are never recorded
        raise RuntimeError(f"refusing to write recordings: the API key appears in {leaking}")
    # Remove the previous run's recordings, so no stale file outlives a changed run.
    for old in out.rglob("*.json") if out.exists() else ():
        old.unlink()
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def print_report(checks: Sequence[Check], out: Path) -> None:
    width = max(len(f"{c.route} {c.name}") for c in checks)
    for c in checks:
        print(f"{c.outcome:<8} {f'{c.route} {c.name}':<{width}}  {c.detail}")
    counts = {o: sum(c.outcome == o for c in checks) for o in ("PASS", "FINDING", "FAIL")}
    print(f"\n{counts['PASS']} pass, {counts['FINDING']} findings, {counts['FAIL']} fail")
    print(f"recordings: {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--live", action="store_true", help="really call OpenRouter (costs money)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to save recordings")
    args = parser.parse_args(argv)
    out: Path = args.out

    if not args.live:
        print(__doc__)
        print("Dry run: nothing was sent. Add --live to call OpenRouter.")
        return 2
    key = os.environ.get(API_KEY_ENV, "")
    if not key:
        print(f"{API_KEY_ENV} is not set in this terminal.", file=sys.stderr)
        return 2

    checks = run_smoke(transport=httpx.HTTPTransport(), api_key=SecretStr(key), out=out)
    print_report(checks, out)
    return 1 if any(c.outcome == "FAIL" for c in checks) else 0


# --- helpers ---


def _parse(content: bytes) -> JsonValue:
    try:
        return _JSON.validate_json(content)
    except ValidationError:
        return content.decode("utf-8", errors="replace")


def dig(value: JsonValue, *path: str | int) -> JsonValue:
    """value[path[0]][path[1]]..., or None if any step is missing."""
    for key in path:
        if isinstance(key, str) and isinstance(value, dict):
            value = value.get(key)
        elif isinstance(key, int) and isinstance(value, list) and -len(value) <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _usage_check(exchanges: Sequence[Exchange]) -> tuple[Outcome, str]:
    ok = [e for e in exchanges if e.status == 200]
    missing: list[str] = []
    for field_path in (("cost",), ("prompt_tokens_details", "cached_tokens")):
        if any(dig(e.response, "usage", *field_path) is None for e in ok):
            missing.append(".".join(field_path))
    if not ok:
        return "INFO", "no successful responses"
    if missing:
        return "FINDING", f"missing in some responses: {', '.join(missing)}"
    return "PASS", f"cost and cached_tokens in all {len(ok)} responses"


@dataclass(frozen=True)
class Prefix:
    """A conversation cut just after the first tool-calling turn that carries reasoning."""

    before: tuple[Message, ...]
    turn: ModelTurn
    results: tuple[Message, ...]  # the turn's tool results


def _prefix_after_first_tool_turn(conversation: Sequence[Message]) -> Prefix | None:
    for i, message in enumerate(conversation):
        if isinstance(message, ModelTurn) and message.tool_calls and _has_reasoning(message):
            end = i + 1 + len(message.tool_calls)
            return Prefix(tuple(conversation[:i]), message, tuple(conversation[i + 1 : end]))
    return None


def _has_reasoning(turn: ModelTurn) -> bool:
    details = turn.raw_message.get("reasoning_details")
    return isinstance(details, list) and len(details) > 0


def _tamper(raw: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], str] | None:
    """A copy of the message with its reasoning edited, and what was edited."""
    edited = copy.deepcopy(raw)
    details = edited.get("reasoning_details")
    if not isinstance(details, list):
        return None
    for item in details:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text:
                item["text"] = text + " (edited)"
                return edited, "reasoning text changed"
    for item in details:
        if isinstance(item, dict):
            signature = item.get("signature")
            if isinstance(signature, str) and signature:
                item["signature"] = signature[:-4] + "AAAA"
                return edited, "reasoning signature changed"
    return None


def _no_provider(error: str | None) -> bool:
    text = (error or "").lower()
    markers = ("no endpoints", "endpoints out of", "no allowed providers", "eligible")
    return any(marker in text for marker in markers)


if __name__ == "__main__":
    raise SystemExit(main())
