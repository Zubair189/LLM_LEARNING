# Week 11 – Production LLM Applications

Weeks 1–10 covered how LLMs, RAG, fine-tuning and agents work. This week is about turning them into software that real users can rely on. You will learn to design an LLM-powered service, put it behind an API gateway and a FastAPI backend, and manage conversation history and context within a token budget. You will also defend it against direct and indirect prompt injection, protect API keys, and control spending with rate limiting, caching and cost optimisation. Finally, you will measure quality with evaluation and hallucination detection, watch it with observability, latency and failure monitoring, and keep dangerous actions safe with restricted SQL and human approval. The theme of the week is that **the model is one component; everything around it makes it production-ready.**

## Questions
1. [How do you design an LLM-powered application?](01-How-do-you-design-an-LLM-powered-application.md)
2. [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
3. [How does FastAPI help in AI applications?](03-How-does-FastAPI-help-in-AI-applications.md)
4. [How do you maintain conversation history?](04-How-do-you-maintain-conversation-history.md)
5. [What is context management?](05-What-is-context-management.md)
6. [What is prompt injection?](06-What-is-prompt-injection.md) (general concept, direct attacks)
7. [What is an indirect prompt injection attack?](07-What-is-an-indirect-prompt-injection-attack.md) (attacks hidden in documents and tool output)
8. [How do you protect API keys?](08-How-do-you-protect-API-keys.md)
9. [What is rate limiting?](09-What-is-rate-limiting.md)
10. [What is response caching?](10-What-is-response-caching.md)
11. [What is token consumption?](11-What-is-token-consumption.md) (measuring where tokens go)
12. [How do you reduce LLM API costs?](12-How-do-you-reduce-LLM-API-costs.md) (the optimisation playbook)
13. [What is LLM evaluation?](13-What-is-LLM-evaluation.md) (the discipline; compare Week 12 Q11 for the per-response rubric)
14. [What is hallucination detection?](14-What-is-hallucination-detection.md)
15. [What is observability?](15-What-is-observability.md)
16. [How do you monitor latency and failures?](16-How-do-you-monitor-latency-and-failures.md)
17. [Why should AI-generated SQL be restricted?](17-Why-should-AI-generated-SQL-be-restricted.md)
18. [How do you implement human approval for sensitive operations?](18-How-do-you-implement-human-approval-for-sensitive-operations.md)
19. **BONUS:** [What are guardrails in LLM applications?](19-BONUS-What-are-guardrails-in-LLM-applications.md)

## Practical Exercise
Build a **"production-ready" FAQ assistant** in a single FastAPI app (about 200 lines):

1. `POST /chat` takes `{session_id, message}` (Pydantic, max 2,000 chars) and calls `claude-haiku-5-5` with the Anthropic SDK; the key comes from `ANTHROPIC_API_KEY`.
2. Store history in Redis (or a dict) and keep only the last 6 messages.
3. Add a token-bucket rate limiter: 5 requests per 10 seconds per `x-user-id`, returning 429 with `Retry-After`.
4. Add an exact-match response cache with a 1-hour TTL; return `"cached": true/false` in the response.
5. Log one structured JSON line per request: `request_id`, latency ms, input/output tokens and an illustrative cost estimate.
6. Add an input guardrail that rejects "ignore previous instructions"-style prompts, and an output guardrail that validates a JSON reply schema.
7. Write a 10-question eval script that hits your endpoint and reports pass rate and p95 latency.

**Done when:** a repeated question shows `cached: true` with ~0 tokens, the 6th rapid request gets a 429, and your eval script prints a pass rate and p95.

## Explain Aloud
Practise explaining each of these out loud in under 60 seconds, as in an interview:
1. **Direct vs indirect prompt injection**: who plants the payload, why agents with tools are most at risk, and why least privilege plus human approval matter more than a clever system prompt.
2. **Token consumption → cost reduction**: where hidden input tokens come from (history, RAG chunks, tool schemas), and three levers to cut cost (model routing, caching, context trimming) without hurting quality.
3. **Observability for LLM apps**: logs vs metrics vs traces, what a trace of one RAG request looks like, and why you track p95 latency, tokens and grounding scores rather than just "the server is up".
