# HighMemory

**Alert:** a replica's memory use stays above 90% of its limit for 5 minutes.

## Symptoms

- Memory climbs on one or more replicas; at 100% the platform kills the container and restarts it.
- Short bursts of failed requests when a replica is killed. Latency may rise just before, as the runtime struggles to free memory.

## Likely causes, in order

1. A leak: memory grows steadily and only drops when the replica restarts. Check the replica restart count.
2. A traffic spike or one unusually large request: memory jumps with load and falls back afterwards.

## What to do

- Leak: restart the app before the replica runs out; it restores service for a while. Open a ticket for the owning team to fix the code. Scaling out does not help: every new replica leaks the same way.
- Traffic spike: let autoscaling handle it; escalate if replicas are already at the maximum.
