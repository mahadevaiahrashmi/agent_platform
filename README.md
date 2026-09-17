# Agent Platform - Applied AI Engineering

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

The docstrings in those files are the exact implementation contracts. The matching pytest suites in `tests/` are the authoritative specification for exact behavior, including error strings, return shapes, call order, and timing. The modules are independent:

`fake_llm.py` provides a deterministic `FakeLLM`; do not modify it. Run `pytest tests/` locally as you work.

# Agent Platform — Detailed Engineering Brief

Meridian's product teams are moving LLM-powered document processing, support research, operational actions, and event-driven workflows into production. The prototypes proved the product value, but each team has independently encountered malformed model output, stalled planning loops, unsafe tool access, forgotten context, uncontrolled spend, duplicate jobs, uneven answer quality, and poor diagnostics.

You are building the shared **Agent Platform** that addresses those failure modes. The repository contains ten independent Python skeletons. Implement every method that raises `NotImplementedError` in `problems/`, following its docstring contract exactly.

## Platform workstreams

| Production need | File | What the subsystem must do | Key behaviors graded |
|---|---|---|---|
| Downstream services cannot accept free-form or malformed extraction results. | `structured_output.py` | Produce a Pydantic-validated object from an LLM response. | Prompts contain the source text and JSON schema; JSON parsing and schema validation are enforced; each failed attempt logs its number, raw response, and error; the next prompt includes the previous error; `max_retries + 1` total attempts are allowed before `ExtractionError`. |
| Research and task agents must recover from mistakes without looping indefinitely. | `react_loop.py` | Run a bounded ReAct observe-think-act cycle over a supplied tool set. | Every decision sees the goal and prior trace; final decisions stop immediately; tool results feed the next iteration; unknown tools, tool exceptions, and invalid decisions become error observations; the LLM call count never exceeds `max_iterations`; exhaustion returns a non-empty best-effort answer rather than raising. |
| A growing tool catalog needs predictable routing, access control, and fast fan-out. | `tool_orchestrator.py` | Register tools, resolve capabilities, enforce scopes, and execute batches concurrently. | Re-registering a name replaces it; highest priority wins and alphabetical name breaks ties; unknown capabilities raise `KeyError`; missing scopes raise `PermissionDenied` before execution; parallel tasks use real thread concurrency, preserve input order, and isolate failures as `{"ok": False, "error": ...}`. |
| Conversations need useful history without an unbounded prompt or session-local memory. | `memory_agent.py` | Maintain rolling short-term turns and searchable, compressible, persistent long-term memory. | Evicted turns move to long-term storage; relevance is lowercase whitespace-token overlap with an empty-query score of `0.0`; recall filters zero scores, ranks highest first, and keeps insertion order on ties; compression replaces only long-term turn entries with one LLM summary while preserving existing summaries; save/load restores both memory tiers and window semantics. |
| Uncertain output and protected actions cannot proceed without accountable human control. | `hitl_approval.py` | Gate low-confidence or protected actions behind resumable approval tickets. | A request event is always logged first; confident, unprotected work completes automatically; low confidence or a protected action pauses with the specified reason and does not execute the action; approval returns the pending answer, rejection aborts, and each ticket resolves once; audit events remain ordered and can be filtered by ticket, including the human note. |
| Traffic must use capable models without allowing confidence recovery to break the spend ceiling. | `cost_router.py` | Estimate task complexity, choose a model, enforce a hard budget, and report usage. | Complexity uses the exact word-count-plus-keyword heuristic; routing chooses the cheapest capable model or the most capable fallback; budget is checked before every call; a confident routed answer exits early; a low-confidence answer may escalate once to the most capable model only if affordable; the final task result and ledger support total, per-model, call-count, and task escalation analytics. |
| Webhook and queue delivery is at-least-once, so retries must not create duplicate effects. | `event_automation.py` | Process events idempotently with bounded retry, exponential backoff, dead-lettering, and replay. | Successful and dead-lettered IDs reject later duplicates; unknown types dead-letter without retry; handler failures retry `max_retries` additional times with delays `1, 2, 4, ...`; final dead-letter records contain the last error and total attempts; `process()` never raises; replay bypasses dead-letter idempotency, counts recoveries, and does not duplicate failed entries. |
| Difficult product decisions benefit from diverse proposals, but selection must remain reproducible. | `debate_system.py` | Gather independent proposals, score them with a critic, and synthesize the selected answer. | Proposers do not see one another's work; the critic sees every proposal and is called once; highest score wins with lower index breaking ties; confidence is the clamped winner score minus the mean of other scores, or the winner score for one proposer; the aggregator receives the question, winning proposal, and its critique exactly once; the required result fields are returned. |
| Generated deliverables should improve against explicit criteria rather than accept the first draft. | `self_eval.py` | Run a worker/judge refinement loop and retain evaluation history. | The first worker prompt includes task and criteria; every output is judged against the criteria; passing scores stop immediately; retries include both the previous output and critique; execution stops at `max_attempts`; the best-scoring output is returned, while `improvement` is the last score minus the first; every attempt is recorded with 1-based numbering. |
| Operators need evidence of latency, cost, failures, nesting, and repeated-call loops. | `observability.py` | Trace spans and wrap an LLM with per-call accounting and loop detection. | Spans use the injected clock, capture parent names, and are appended in finish order even when exceptions propagate; each wrapped call runs inside a span named `llm.complete`, increments count and cost even on failure, and alarms at every repeated-prompt threshold multiple; reports contain calls, total cost, mean completed LLM-span latency, and alerts. |

## Working contract

- The docstrings in the ten candidate modules are the exact contracts. Do not rename methods, change signatures, or alter required return shapes and exceptions.
- The pytest suites in `tests/` define the authoritative behavior, including exact error strings, return shapes, call order, and timing constraints.
- The modules are independent. You can implement them in any order and complete as many as time allows; partial credit is awarded per passing test across **64 tests**.
- `fake_llm.py` is a provided deterministic `FakeLLM`. It returns scripted text or raises scripted exceptions, records calls, and keeps grading fully offline. Do not modify it; no provider account or API key is required.
- Run `pytest tests/` locally as you work. The visible suite is designed to give immediate feedback for each subsystem.

# Defend Your Build: 3-Minute
Engineering Walkthrough (Video)
Create a video on what you've built
Record a 4 minute video explaining the Agent Platform assignment you just submitted. This is not a demo - we've already run your tests. We want to hear how you think.
Cover these four things, in whatever order works for you:
1. ﻿﻿﻿What you shipped. Which modules you completed, which you skipped, how you decided what to work on first and your framework for prioritizing tasks.
2. ﻿﻿﻿One decision you are proud of: Pick a specific implementation choice (e.g., retry strategies, how you ranked memory, how you enforced permission scopes, how you made the winner deterministic), explain why you chose it, and share what alternative approaches you rejected. — and explain why you did it that way and what you rejected. 
3. ﻿﻿﻿One thing that fought you. A test you couldn't pass, a contract that was ambiguous, or a bug that cost you time. Tell us what was happening and how you tackled it.
4. ﻿﻿﻿What breaks in production. Identify an aspect of your code that would not hold up under real-world enterprise traffic, and explain what you would refactor or change.

Requirements:
* ﻿﻿Show your face and your screen. Talking over your code is encouraged; slides are not needed.
* ﻿﻿Reference specific files, functions, or test names. Generic descriptions of "agentic best practices" score zero.
* ﻿﻿Hard limit: 4 minutes. We stop watching at 4:00. Being concise is part of the evaluation.
* ﻿﻿Unedited single take is fine and preferred. Presentation polish is not what we're scoring.

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
PYTHONPATH=production python -m pytest tests/test_*.py -q
PYTHONPATH=production python -m pytest tests/test_*.py -v //verbose
```

All modules accept a deterministic `FakeLLM` (see `src/fake_llm.py`) so tests run fully offline.

## Note

This implementation follows the contracts described in the engineering brief.
In the original challenge the skeletons and 64 pytest cases define exact
signatures and error strings; the code here is a faithful, self-contained
realization of those behaviors.

