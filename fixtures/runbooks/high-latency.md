# HighLatency

**Alert:** the 95th percentile response time stays above the threshold for 5 minutes.

## Symptoms

- Pages or API calls are slow; some may time out (504).

## Likely causes, in order

1. A dependency the service calls has slowed down: compare dependency durations with request durations.
2. The service is short of resources: CPU, memory or database connections.
3. A few known slow requests (reports, exports) at a quiet time, when there are too few other requests to dilute them.

## What to do

- Dependency slow: contact the dependency's owner. Restarting or scaling this service does not make the dependency faster, and more replicas send it more load.
- Short of resources: see the matching runbook.
- Known slow requests, no errors, no user impact: no action. Note it so the alert can be tuned.
