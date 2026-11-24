# Can RAG replace fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 15

## Short Answer
Sometimes, but not in general. They solve different problems. **RAG** (retrieval-augmented generation) gives a model the **knowledge** it needs at query time: fresh, private or citable facts. **Fine-tuning** changes the model's **behaviour**: format, style, task skill, and cost or latency. For "answer questions about our documents," RAG usually replaces fine-tuning entirely. For "always output our exact schema in our voice with a small, cheap model," fine-tuning is the right tool. Many production systems use both.

## Detailed Explanation
| Dimension | RAG | Fine-tuning |
|---|---|---|
| Adds new/changing facts | Yes, just update the index | Poorly; facts go stale and may be hallucinated |
| Citations / traceability | Yes | No |
| Changes style/format/skill | Weakly (via prompt) | Strongly |
| Data needed | Documents | Labelled examples (hundreds+) |
| Update cost | Re-index (minutes) | Re-train + re-evaluate |
| Inference cost | Longer prompts (retrieved context) | Shorter prompts; can use a smaller model |
| Access control | Filter at retrieval per user | Baked in; can't un-learn per user |
| Failure mode | Bad retrieval → wrong/missing context | Overfitting, forgetting |

**RAG wins when**
- Knowledge changes often (prices, policies, tickets).
- You need source citations or per-user permissions.
- You have documents but few labelled Q&A pairs.

**Fine-tuning wins when**
- The problem is *how* to answer, not *what* to know: strict formats, tone, classification, domain jargon.
- You need low latency or cost at high volume (a small fine-tuned model instead of a large prompted one).
- Prompts are getting huge with instructions and few-shot examples.

**Hybrid patterns**
- Fine-tune the model to *use retrieved context well* (cite sources, say "not found" when the context lacks the answer), then use RAG for the facts. This is sometimes called RAFT.
- Fine-tune the **embedding model** or reranker for better retrieval in your domain.

```
User question ──► retriever ──► top-k chunks ──┐
                                               ├──► (fine-tuned) LLM ──► answer + citations
             system prompt / format rules ─────┘
```

## Example
**Analogy:** RAG is giving a lawyer the case files before the meeting. Fine-tuning is sending them to law school for a specialisation. A good lawyer needs both, but if the case files change daily you can't send them back to school each morning.

```python
# Decide per requirement, not per project
needs = {"fresh_facts": True, "citations": True, "strict_format": True, "low_cost": False}

plan = []
if needs["fresh_facts"] or needs["citations"]:
    plan.append("RAG")
if needs["strict_format"] or needs["low_cost"]:
    plan.append("fine-tune (LoRA) or structured outputs")
print(" + ".join(plan))   # RAG + fine-tune (LoRA) or structured outputs
```

## Common Mistakes / Misconceptions
- Fine-tuning on documents to "make the model know them." Recall is unreliable and goes stale; use RAG.
- Thinking RAG fixes style or format problems. Retrieval doesn't teach output conventions.
- Assuming long context windows remove the need for both. Cost, latency and "lost in the middle" effects still matter.
- Framing it as either/or. The strongest systems often combine RAG with a lightly fine-tuned model.

## Key Takeaways
- RAG = knowledge at query time; fine-tuning = behaviour baked into the weights.
- For document Q&A, RAG usually replaces fine-tuning.
- For format, style, narrow skills and cost, fine-tuning remains valuable.
- Hybrids (RAG + fine-tuned generator or embedder) are common in production.

## Related Questions
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is long-term memory? (Week 10)](../Week-10-AI-Agents-and-Tool-Calling/08-What-is-long-term-memory.md)
