# Agent Platform - Applied AI Engineering: FDE Hiring

Several product teams at Meridian are moving LLM-powered features from pilots into daily operations: extracting records from incoming documents, researching support questions, taking approved actions, and responding to background events. Each pilot works in isolation, but production has exposed the same failures repeatedly. Model responses arrive malformed, reasoning loops stall, tools are invoked without consistent access controls, conversations forget important context, and retries create duplicate work. Your platform team has been asked to turn those lessons into a shared Python Agent Platform.

The first priority is dependable execution. Document extraction must return schema-valid data instead of crashing downstream services, with bounded correction attempts when a response is invalid. Research agents need an observe-think-act loop that can recover from bad decisions and tool failures without running forever. As the tool catalog grows, capabilities must resolve deterministically, protected tools must respect permission scopes, and independent calls should run concurrently without one failure taking down the batch.

The platform must also carry context and enforce operational boundaries. Older conversation turns should remain recallable, be ranked deterministically, compress cleanly, and survive a new session. Low-confidence answers and protected actions such as refunds must pause for a single-use human decision with an ordered audit trail. At higher traffic, work should go to the cheapest capable model, stop early on a confident answer, escalate at most once when useful and affordable, and never exceed its budget. Queue and webhook jobs need similar discipline: duplicate delivery must be harmless, transient failures should back off, exhausted work should enter a dead-letter queue, and replay should be safe.

Finally, teams need stronger quality controls for difficult requests. The platform should be able to collect independent proposals, have a critic score them, choose a deterministic winner, and synthesize a confidence-bearing answer. A worker-and-judge loop should also refine drafts under explicit criteria while retaining the best attempt. Around every call, tracing, latency and cost accounting, and repeated-prompt alarms must make failures and runaway loops visible.

## Your Assignments

Implement the methods that raise `NotImplementedError` in these ten candidate files:

| Production Concern | File to Implement |
|---|---|
| Schema-safe extraction | `src/structured_output.py` |
| Bounded ReAct execution | `src/react_loop.py` |
| Tool routing, permissions, and concurrency | `src/tool_orchestrator.py` |
| Conversion memory and persistence | `src/memory_agent.py` |
| Human approval and auditing | `src/hitl_approval.py` |
| Cost-aware routing and budgets | `src/cost_router.py` |
| Idempotent event automation | `src/event_automation.py` |
| Independent proposals and consensus | `src/debate_system.py` |
| Judge-guided refinement | `src/self_eval.py` |
| Tracing, accounting, and loop detection | `src/observability.py` |

The docstrings in those files are the exact implementation contracts. The matching pytest suites in `tests/` are the authoritative specification for exact behavior, including error strings, return shapes, call order, and timing. The modules are independent: complete as many as you can, with partial credit awarded per passing test across 64 tests.

`src/fake_llm.py` provides a deterministic `FakeLLM`; do not modify it. No network access, provider account, or API key is needed. Run `pytest tests/` locally as you work.