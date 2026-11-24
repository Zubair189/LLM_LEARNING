# What is observability?

> Week 11 – Production LLM Applications · Question 15

## Short Answer
Observability is the ability to understand what your system is doing, and why, from the data it emits: **logs, metrics and traces**. For LLM apps it also covers prompts, model outputs, token usage, cost, retrieval results and tool calls for each request. Good observability lets you answer questions like "why was this answer wrong?" or "why did cost double yesterday?" without redeploying code. OpenTelemetry is the standard toolkit for producing this data.

## Detailed Explanation
**The three pillars (+ LLM extras)**
| Signal | What it is | LLM example |
|---|---|---|
| Logs | Timestamped events (structured JSON) | `{"event":"llm_call","model":"claude-sonnet-5-5","in":812,"out":143}` |
| Metrics | Numeric time series | p95 latency, error rate, tokens/min, $/day |
| Traces | One request's path across steps, as nested spans | gateway → retrieve → LLM → tool → LLM |
| LLM-specific | Prompt/version, retrieved doc IDs, tool args, eval scores, user feedback | "answer used chunks 12, 47; judge score 0.4" |

**Trace of one RAG request**
```
trace_id=abc123  POST /ask  (1,940 ms)
 ├─ span: auth                    5 ms
 ├─ span: embed_query            60 ms
 ├─ span: vector_search          45 ms   top_k=5
 ├─ span: llm.generate        1,780 ms   model=claude-sonnet-5-5 in=2100 out=180
 └─ span: grounding_check        50 ms   supported=true
```

**Monitoring vs observability:** monitoring tells you *that* something is wrong (an alert fired). Observability gives you the data to find out *why*.

**Privacy:** prompts and outputs may contain PII. Redact, sample, restrict access and set retention.

## Example
**Analogy:** A car dashboard (metrics), a trip log (logs) and a GPS replay of the exact route (trace). Together they tell the mechanic what happened on a bad journey.

```python
import json, logging, time
import anthropic
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer("qa-app")
logging.basicConfig(level=logging.INFO, format="%(message)s")
client = anthropic.Anthropic()

def ask(question: str) -> str:
    with tracer.start_as_current_span("llm.generate") as span:
        t0 = time.perf_counter()
        r = client.messages.create(model="claude-sonnet-5-5", max_tokens=200,
                                   messages=[{"role": "user", "content": question}])
        ms = round((time.perf_counter() - t0) * 1000)
        span.set_attribute("llm.model", "claude-sonnet-5-5")
        span.set_attribute("llm.input_tokens", r.usage.input_tokens)
        span.set_attribute("llm.output_tokens", r.usage.output_tokens)
        logging.info(json.dumps({"event": "llm_call", "ms": ms,
                                 "in": r.usage.input_tokens, "out": r.usage.output_tokens}))
        return r.content[0].text

ask("What is observability?")
# -> {"event": "llm_call", "ms": 1420, "in": 12, "out": 96}   (illustrative)
# -> span JSON printed with llm.model / token attributes
```
In production, export spans to a backend (Jaeger, Grafana Tempo, Datadog, Honeycomb, or an LLM-focused tool such as Langfuse) instead of the console.

## Common Mistakes / Misconceptions
- Unstructured `print` logs that can't be queried.
- Logging full prompts with PII or secrets and no redaction or retention policy.
- No request/trace ID, so you can't connect gateway, retrieval and LLM logs.
- Tracking only latency and errors, not tokens, cost or answer quality.

## Key Takeaways
- Observability = logs + metrics + traces, plus LLM-specific data (prompts, tokens, sources, scores).
- Traces show where time and tokens are spent within a request.
- Use OpenTelemetry and structured JSON logs.
- Redact sensitive data; observability must not become a data leak.

## Related Questions
- [How do you monitor latency and failures?](16-How-do-you-monitor-latency-and-failures.md)
- [What is token consumption?](11-What-is-token-consumption.md)
- [What is LLM evaluation?](13-What-is-LLM-evaluation.md)
- [How do you deploy an LLM application in production?](../Week-12-Final-Project-and-Revision/12-How-do-you-deploy-an-LLM-application-in-production.md)
