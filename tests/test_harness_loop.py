"""The loop against a scripted model: every way a run can end, and the limits."""

import pytest

from harness_fakes import ScriptedModel, Triage, final, toy_tools, turn
from sre_agent.harness import (
    Limits,
    LocalToolClient,
    ModelHTTPError,
    ModelTurn,
    RecordingTrace,
    StopReason,
    Tool,
    ToolArgs,
    ToolCallEvent,
    ToolResult,
    run_agent,
)

_SVC = '{"service": "payments-api"}'


def _run(model: ScriptedModel, limits: Limits | None = None, trace: RecordingTrace | None = None):
    return run_agent(
        model=model,
        tools=toy_tools(),
        system="You triage services.",
        task="Who owns payments-api and is it healthy?",
        answer_model=Triage,
        limits=limits,
        trace=trace,
    )


def test_two_tool_task_completes() -> None:
    model = ScriptedModel(
        [turn(calls=[("get_service_status", _SVC), ("get_owner", _SVC)]), final()]
    )
    result = _run(model)
    assert result.ok
    assert result.answer == Triage(service="payments-api", status="degraded", owner="team-payments")
    assert result.turns == 2
    assert result.usage.total_tokens == 200
    assert result.usage.cost == pytest.approx(0.02)

    # Both results of turn 1 went back together, in order, before turn 2.
    second_request = model.seen[1][0]
    assert [type(m).__name__ for m in second_request] == [
        "SystemMessage",
        "UserMessage",
        "ModelTurn",
        "ToolResult",
        "ToolResult",
    ]
    results = [m for m in second_request if isinstance(m, ToolResult)]
    assert [r.content for r in results] == ["degraded", "team-payments"]


def test_final_answer_tool_is_offered() -> None:
    model = ScriptedModel([final()])
    _run(model)
    names = [spec.name for spec in model.seen[0][1]]
    assert names == ["get_service_status", "get_owner", "final_answer"]


def test_history_is_append_only() -> None:
    model = ScriptedModel([turn(calls=[("get_owner", _SVC)]), turn(calls=[("get_owner", _SVC)])])
    model.replies.append(final())
    _run(model)
    first, second, third = (seen[0] for seen in model.seen)
    assert second[: len(first)] == first
    assert third[: len(second)] == second


def test_max_turns_is_enforced() -> None:
    model = ScriptedModel([turn(calls=[("get_owner", _SVC)]) for _ in range(3)])
    result = _run(model, Limits(max_turns=3))
    assert result.stop_reason is StopReason.MAX_TURNS
    assert result.turns == 3
    assert not model.replies  # exactly three calls, no fourth


def test_token_budget_stops_the_run() -> None:
    model = ScriptedModel([turn(calls=[("get_owner", _SVC)], tokens=600), final()])
    result = _run(model, Limits(max_total_tokens=500))
    assert result.stop_reason is StopReason.TOKEN_BUDGET
    assert result.turns == 1


def test_max_tokens_never_exceeds_the_remaining_budget() -> None:
    model = ScriptedModel([turn(calls=[("get_owner", _SVC)], tokens=4_500), final()])
    _run(model, Limits(max_total_tokens=5_000, max_tokens_per_request=4_000))
    assert [seen[2] for seen in model.seen] == [4_000, 500]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (turn(text="", finish="length"), StopReason.LENGTH),
        (turn(text="partial", finish="length"), StopReason.LENGTH),
        (turn(text=None, finish="stop", native="refusal"), StopReason.REFUSAL),
        (turn(text=None, finish="content_filter"), StopReason.REFUSAL),
        (turn(text=None, refusal="I can't help with that."), StopReason.REFUSAL),
        (turn(text="All good, I think."), StopReason.NO_ANSWER),
        (turn(text="   "), StopReason.NO_ANSWER),
        (turn(text=None, finish="error"), StopReason.MODEL_ERROR),
        (turn(text=None).model_copy(update={"finish_reason": None}), StopReason.MODEL_ERROR),
        (turn(text=None, finish="pause_turn"), StopReason.MODEL_ERROR),
    ],
)
def test_every_finish_reason_ends_explicitly(reply: ModelTurn, reason: StopReason) -> None:
    result = _run(ScriptedModel([reply]))
    assert result.stop_reason is reason
    assert result.answer is None
    assert result.error


def test_model_error_stops_the_run() -> None:
    result = _run(ScriptedModel([ModelHTTPError(402, "key limit reached")]))
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.error == "HTTP 402: key limit reached"


def test_invalid_final_answer_goes_back_to_the_model() -> None:
    model = ScriptedModel([turn(calls=[("final_answer", '{"service": "x"}')]), final()])
    result = _run(model)
    assert result.ok
    rejected = model.seen[1][0][-1]
    assert isinstance(rejected, ToolResult)
    assert rejected.is_error
    assert "invalid final answer" in rejected.content


def test_final_answer_with_other_calls_is_rejected() -> None:
    answer_args = Triage(service="a", status="b", owner="c").model_dump_json()
    model = ScriptedModel(
        [turn(calls=[("get_owner", _SVC), ("final_answer", answer_args)]), final()]
    )
    result = _run(model)
    assert result.ok
    assert result.turns == 2
    owner, rejected = model.seen[1][0][-2:]
    assert isinstance(owner, ToolResult) and owner.content == "team-payments"
    assert isinstance(rejected, ToolResult) and rejected.is_error


def test_tool_errors_become_results_and_the_run_continues() -> None:
    model = ScriptedModel([turn(calls=[("restart_app", _SVC)]), final()])
    result = _run(model)
    assert result.ok
    error = model.seen[1][0][-1]
    assert isinstance(error, ToolResult) and error.is_error


def test_long_tool_output_is_capped() -> None:
    class NoArgs(ToolArgs):
        pass

    flood = Tool(name="flood", description="x", args_model=NoArgs, fn=lambda a: "A" * 1_000)
    model = ScriptedModel([turn(calls=[("flood", "{}")]), final()])
    run_agent(
        model=model,
        tools=LocalToolClient([flood]),
        system="s",
        task="t",
        answer_model=Triage,
        limits=Limits(max_tool_result_chars=100),
    )
    capped = model.seen[1][0][-1]
    assert isinstance(capped, ToolResult)
    assert capped.content.startswith("A" * 100 + "\n[truncated: 900 characters omitted]")


def test_reserved_tool_name_is_refused() -> None:
    class NoArgs(ToolArgs):
        pass

    clash = Tool(name="final_answer", description="x", args_model=NoArgs, fn=lambda a: "")
    with pytest.raises(ValueError, match="reserved"):
        run_agent(
            model=ScriptedModel([]),
            tools=LocalToolClient([clash]),
            system="s",
            task="t",
            answer_model=Triage,
        )


def test_trace_sees_every_call_in_order() -> None:
    trace = RecordingTrace()
    model = ScriptedModel([turn(calls=[("get_owner", _SVC)]), final()])
    _run(model, trace=trace)
    kinds = [
        f"tool:{e.call.name}" if isinstance(e, ToolCallEvent) else "model" for e in trace.events
    ]
    assert kinds == ["model", "tool:get_owner", "model", "tool:final_answer"]


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        Limits(max_turns=0)
