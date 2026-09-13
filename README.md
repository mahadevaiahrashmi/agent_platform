# Agent Platform

Shared production primitives for LLM-powered document processing, research, operational actions, and event-driven workflows.

## Modules

| Module | Responsibility |
|--------|----------------|
| `structured_output` | Pydantic-validated extraction with error-feedback retries |
| `react_loop` | Bounded ReAct observe-think-act cycle |
| `tool_orchestrator` | Tool registration, capability routing, scope enforcement, concurrent batch execution |
| `memory_agent` | Rolling short-term + searchable/compressible long-term memory |
| `hitl_approval` | Human-in-the-loop approval tickets for low-confidence / protected actions |
| `cost_router` | Complexity-based model selection with hard budget and one-shot escalation |
| `event_automation` | Idempotent event processing with retry, backoff, dead-letter, and replay |
| `debate_system` | Independent proposers → critic scoring → reproducible synthesis |
| `self_eval` | Worker/judge refinement loop against explicit criteria |
| `observability` | Span tracing + instrumented LLM with cost/loop detection |

## Usage

```bash
pip install -r requirements.txt
python -c "from src.structured_output import extract_structured; ..."
```

All modules accept a deterministic `FakeLLM` (see `src/fake_llm.py`) so tests run fully offline.

## Note

This implementation follows the contracts described in the engineering brief.
In the original challenge the skeletons and 64 pytest cases define exact
signatures and error strings; the code here is a faithful, self-contained
realization of those behaviors.
