# sre-agent

An SRE triage agent: it reads an incident's alert, logs, metrics and recent deploys, finds the likely cause and proposes an action. Every tool and model call goes through the [Agent Governance Gateway](https://github.com/ankitutekar/agent-gateway), which decides what the agent may do.

Work in progress. The agent's scope is described in the gateway repo's [scope doc](docs/scope.md), and the threats it is tested against are in its [threat model](https://github.com/ankitutekar/agent-gateway/blob/main/THREAT_MODEL.md).

The [incident fixtures](fixtures/) are the agent's test data: six clean incidents with known right answers, and four poisoned copies with planted instructions.

## Harness

`src/sre_agent/harness/` is the agent's tool-use loop (plan step 1.3, [ADR 0008](https://github.com/ankitutekar/agent-gateway/blob/main/docs/adr/0008-agent-harness-and-model-path.md)): our own small loop over OpenRouter, with no framework and no vendor SDK. Each run has hard limits (turns, tokens), every way out has a stated reason, the final answer is validated in code, the provider is pinned (no fallbacks), and history is append-only. There are two model backends: Chat Completions (in use) and Anthropic's Messages format (a tested fallback).

Tests run offline: `make check`. They replay real responses recorded in `tests/recordings/openrouter/` and check that every request is byte-identical to the recorded one.

The live smoke test calls OpenRouter and costs a few cents. The maintainer runs it in their own terminal, with the key typed in rather than pasted into a command (so it stays out of shell history):

```powershell
$env:OPENROUTER_API_KEY = Read-Host -MaskInput "OpenRouter key"
uv run python scripts/smoke_openrouter.py --live
Remove-Item Env:OPENROUTER_API_KEY
```

Without `--live` it only describes what it would do. It rewrites the recordings; commit them after a deliberate change to what goes over the wire.
