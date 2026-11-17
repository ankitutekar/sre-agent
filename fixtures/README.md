# Incident fixtures

Made-up incidents for testing the SRE agent, each with a known right answer (ground truth). Poisoned variants copy a clean scenario and change one thing: a planted instruction in one channel (a log line, runbook text, an alert annotation or a ticket comment).

Scenarios are added in plan step 1.2. Clean scenarios check that the agent triages well; poisoned ones check that it cannot be steered, and map to threat IDs in the gateway's [threat model](https://github.com/ankitutekar/agent-gateway/blob/main/THREAT_MODEL.md).
