# sre-agent

An SRE triage agent: it reads an incident's alert, logs, metrics and recent deploys, finds the likely cause and proposes an action. Every tool and model call goes through the [Agent Governance Gateway](https://github.com/ankitutekar/agent-gateway), which decides what the agent may do.

Work in progress. The agent's scope is described in the gateway repo's [scope doc](docs/scope.md), and the threats it is tested against are in its [threat model](https://github.com/ankitutekar/agent-gateway/blob/main/THREAT_MODEL.md).

The [incident fixtures](fixtures/) are the agent's test data: six clean incidents with known right answers, and four poisoned copies with planted instructions.
