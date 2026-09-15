# Production Agent Platform (stack-independent)

All modules remain compatible with the original **64 graded tests**.
Optional production features live **inside** the same module files (no extra support packages).

## List-A features (inlined in each module)
| Module | Features |
|--------|----------|
| structured_output | Circuit breaker, metrics hooks, schema_version |
| react_loop | overall_timeout, cancel token, per-tool timeouts, allow/deny, idempotency_key, run_id, checkpoint |
| tool_orchestrator | timeouts, arg_schema validation, rate limits, structured errors |
| memory_agent | pluggable scorer, redactor, TTL, max_long_term |
| hitl_approval | ticket TTL, required_approvals, in-memory ticket store |
| cost_router | in-memory budget helper, task_cost_ceiling, fail_open |
| event_automation | jitter, max_event_age, max_replay_attempts (poison), in-memory seen-store |
| debate_system | parallel proposers, cost_ceiling |
| self_eval | cost_ceiling |
| observability | span exporter hook, correlation_id, prompt redactor |

Docstrings match the original problem statements.

## Run graded tests
```bash
cd agent_platform
PYTHONPATH=production python -m pytest test_*.py -q
```
