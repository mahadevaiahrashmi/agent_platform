I completed all ten modules and passed all 64 tests.

I started with the clearest contracts—structured output, tool orchestration, and event automation—then the agent loops, and finished with cost routing, memory, and observability. Nothing was skipped.



# One decision I’m proud of

I chose priority-based tool calls, with alphabetical order as the tie-breaker, so behavior is easy to explain, debug, and predict.

When several tools can do the same job, higher priority wins. If priority is the same, we pick by name. Same situation, same tool, every time.

I rejected letting each caller pick their favorite tool. That sounds flexible, but it makes the product inconsistent and harder to support.

---

# One thing that fought me

What was happening: tools finished in random order, so results came back shuffled even when the work was concurrent.

How I tackled it: allocate one slot per task up front and write each result into its own index. Failures stay in their slot and do not wipe the rest of the batch.

---

# Changes I planned but did not finish as hackerrank page got stuck and customer support couldnt fix it

These are improvements I designed and later finished offline. On the platform I stopped short so I would not break the 64 tests:

- Add timeouts on tool calls so a hung API cannot block the agent for hours, and allow cancel between turns.  

- In the tool orchestrator, add simple argument checks and in-process rate limits.  

- For structured output,  Add a circuit breaker after repeated full failures, and keep retries on the extraction call itself and not a second fixer model.

- For human approval, add ticket expiry and multi-approval for high-risk actions, without running protected actions while waiting.  

- For events, add jitter on retry waits, drop very old events, and stop replaying messages that fail too many times.  

- For cost, memory, and observability, add a clear budget check, memory retention and redaction, request ids on logs, and prompt redaction before alerts.

I did not lock in Redis, Postgres, or any other stack. That was out of scope and not approved. The idea was optional in-process controls with defaults that keep the grader green.

---

# What still breaks in production

Even with those changes, some things would not hold up under real enterprise traffic.

Tickets, “already seen” events, and budgets that live only in memory vanish on restart and do not share across machines. I would move them to a shared durable store once the team chooses infrastructure.

Budget checks need one shared atomic counter. If two requests check at the same time with only a local variable, both can spend and go over the limit.

Memory ranked by shared words is fine for tests but weak on real chat where people paraphrase. I would plug in stronger retrieval behind the same API.

Tool timeouts and stronger isolation should be the default in production, not optional. Who can call sensitive tools should come from the real login and roles system, not only from a scope string inside one service.