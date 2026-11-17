# sre-agent

The SRE triage agent. The workspace `CLAUDE.md` one level up applies here in full: who runs what, teaching style, and keeping the plan in sync.

- Start Claude Code sessions from this folder, so the guard hook in `.claude/hooks/guard.py` loads (a copy of the gateway's, ADR 0004 there). Run `make install` first: the hook runs with this repo's `.venv`.
- `fixtures/`: incident scenarios for evaluating the agent (plan step 1.2).
- The agent's scope and threats live in the gateway repo: `docs/agents/sre.md` and `THREAT_MODEL.md`.
