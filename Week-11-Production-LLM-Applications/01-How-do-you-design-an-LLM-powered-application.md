# How do you design an LLM-powered application?

> Week 11 – Production LLM Applications · Question 1

## Short Answer
Design an LLM application like any other production service, with the model as one component and not the whole system. Start from the user task and success criteria. Then put a thin API layer in front of the model, plus prompt and context management, optional retrieval or tools, guardrails on inputs and outputs, and caching, rate limiting and observability around everything. The model is unreliable and costs money on every call, so the rest of the architecture exists to make it safe, cheap and measurable.

## Detailed Explanation
A practical design process:

1. **Define the task and quality bar.** What does a "good" answer look like? Collect 20–50 example inputs with expected outputs. This becomes your evaluation set.
2. **Pick the simplest pattern that works.**
   | Pattern | Use when |
   |---|---|
   | Single prompt | Classification, summarisation, rewriting |
   | RAG | Answers must come from your own documents |
   | Tool/function calling | The model must fetch live data or take actions |
   | Agent loop | The task has several steps and the steps are not known in advance |
3. **Choose models per step.** Use a cheaper model (e.g. `claude-haiku-5-5`) for routing or classification and a stronger one (`claude-sonnet-5-5`) for reasoning.
4. **Wrap the model in a service** with validated inputs, timeouts, retries and structured outputs.
5. **Add production concerns:** auth, rate limits, caching, cost tracking, logging/tracing, guardrails and human approval for risky actions.
6. **Evaluate continuously.** Run the eval set on every prompt or model change.

```
 Client ──► API Gateway (auth, rate limit)
               │
               ▼
        FastAPI service ──► Input guardrails (size, injection checks)
               │
               ├──► Cache (Redis) ──hit──► return
               ├──► Retriever / Vector DB   (RAG, optional)
               ├──► Tools (read-only SQL, APIs; approval for writes)
               ▼
           LLM provider (Anthropic API)
               │
               ▼
        Output guardrails (schema, PII, grounding) ──► Client
               │
        Logs / traces / metrics / cost ──► Observability stack
```

## Example
**Analogy:** A restaurant does not let the chef stand at the door. A host, a menu, an order system and a food inspector surround the chef. The LLM is the chef.

```python
# app.py – minimal skeleton: validation, system prompt, usage logging
import logging
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")
client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI()

class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class Answer(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int

@app.post("/ask", response_model=Answer)
async def ask(body: Ask) -> Answer:
    resp = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        system="You are a concise support assistant. Say 'I don't know' if unsure.",
        messages=[{"role": "user", "content": body.question}],
    )
    log.info("llm_call", extra={"in": resp.usage.input_tokens, "out": resp.usage.output_tokens})
    return Answer(answer=resp.content[0].text,
                  input_tokens=resp.usage.input_tokens,
                  output_tokens=resp.usage.output_tokens)
# POST /ask {"question": "What is RAG?"}
# -> {"answer": "RAG is ...", "input_tokens": 31, "output_tokens": 58}  (illustrative)
```

## Common Mistakes / Misconceptions
- Starting with an agent framework when a single prompt would do.
- Having no evaluation set, so every prompt change becomes a guess.
- Treating LLM output as trusted, e.g. running generated SQL or code directly.
- Ignoring cost and latency until the first large bill or a timeout.

## Key Takeaways
- The LLM is one component; design the system around it.
- Choose the simplest pattern (prompt → RAG → tools → agent) that meets the need.
- Validate inputs, structure outputs and guard both sides.
- Build in caching, rate limiting, observability and evaluation from day one.

## Related Questions
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [How does FastAPI help in AI applications?](03-How-does-FastAPI-help-in-AI-applications.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [How do you deploy an LLM application in production?](../Week-12-Final-Project-and-Revision/12-How-do-you-deploy-an-LLM-application-in-production.md)
