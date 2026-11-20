"""LocalToolClient: every failure becomes an error result, never an exception."""

import pytest

from harness_fakes import ServiceArgs, get_owner, get_service_status, toy_tools
from sre_agent.harness import LocalToolClient, Tool, ToolCall


def _call(name: str, arguments: str) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=arguments)


def test_valid_call_returns_tool_output() -> None:
    result = toy_tools().call(_call("get_owner", '{"service": "payments-api"}'))
    assert result.content == "team-payments"
    assert not result.is_error
    assert result.tool_call_id == "call_1"


def test_unknown_tool_is_an_error_result() -> None:
    result = toy_tools().call(_call("restart_app", "{}"))
    assert result.is_error
    assert "unknown tool 'restart_app'" in result.content


@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        "{}",  # missing field
        '{"service": 3}',  # wrong type
        '{"service": "payments-api", "force": true}',  # unknown field: rejected
    ],
)
def test_invalid_arguments_are_an_error_result(arguments: str) -> None:
    result = toy_tools().call(_call("get_owner", arguments))
    assert result.is_error
    assert result.content.startswith("error: invalid arguments:")


def test_tool_error_message_reaches_the_model() -> None:
    result = toy_tools().call(_call("get_owner", '{"service": "nope"}'))
    assert result.is_error
    assert "no such service 'nope'" in result.content


def test_unexpected_exception_hides_its_message() -> None:
    def broken(args: ServiceArgs) -> str:
        raise RuntimeError("connection string Server=db;Password=hunter2")

    tool = Tool(name="broken", description="x", args_model=ServiceArgs, fn=broken)
    result = LocalToolClient([tool]).call(_call("broken", '{"service": "a"}'))
    assert result.is_error
    assert result.content == "error: tool failed (RuntimeError)"
    assert "hunter2" not in result.content


def test_spec_schema_forbids_extra_fields() -> None:
    spec = get_service_status.spec()
    assert spec.parameters["additionalProperties"] is False
    assert spec.parameters["required"] == ["service"]


def test_bad_and_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        Tool(name="has space", description="x", args_model=ServiceArgs, fn=lambda a: "")
    with pytest.raises(ValueError, match="duplicate tool name"):
        LocalToolClient([get_owner, get_owner])
