# Production Agent Platform (stack-independent)

All modules remain compatible with the original **64 graded tests**.
Optional production features default off or to in-memory no-ops.

## New support modules
- `config.py` — AgentConfig, CancellationToken, Deadline, RunState
- `interfaces.py` — BudgetChecker, RelevanceScorer, SpanExporter, TicketStore, EventStore, MetricsHook (in-memory defaults)
- `errors.py` — tool/agent error taxonomy

## List-A features implemented
| Module | Features |
|--------|----------|
| structured_output | Circuit breaker, metrics hooks, schema_version |
| react_loop | overall_timeout, cancel_token, per-tool timeouts, allow/deny, idempotency_key, run_id, checkpoint |
| tool_orchestrator | timeouts, arg_schema validation, rate limits, structured errors |
| memory_agent | pluggable scorer, redactor, TTL, max_long_term |
| hitl_approval | ticket TTL, required_approvals (dual-control), TicketStore |
| cost_router | pluggable BudgetChecker, task_cost_ceiling, fail_open |
| event_automation | jitter, max_event_age, max_replay_attempts (poison), EventStore |
| debate_system | parallel proposers, cost_ceiling |
| self_eval | cost_ceiling |
| observability | SpanExporter, correlation_id, prompt redactor |

## Run graded tests
```bash
cd agent_platform
PYTHONPATH=production python -m pytest test_*.py -q
```
