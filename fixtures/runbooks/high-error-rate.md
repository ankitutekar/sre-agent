# HighErrorRate

**Alert:** more than 5% of requests to a service return 5xx for 5 minutes.

## Symptoms

- Users see failed pages or failed orders.
- Error rate up on the service dashboard; latency may or may not change.

## Likely causes, in order

1. A recent deploy (check deploys in the last 30 minutes first).
2. A dependency the service calls is failing (look for errors naming another host).
3. The service is out of resources (memory, connections).

## What to do

- Recent deploy and errors started after it: roll back to the previous version. Do not restart; the new version will come back broken.
- Dependency failing: contact the dependency's owner; do not scale the service up.
- Out of memory: restart restores service for a while; open a ticket to find the cause.
- Out of database connections: find what is holding them. A restart only hides it until they run out again.
- None of the above: escalate to the service owner.
