# How is RAG different from fine-tuning?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 3

## Short Answer
RAG gives the model knowledge *at query time* by retrieving documents and putting them in the prompt; the model's weights never change. Fine-tuning *changes the model's weights* by training on examples, which is best for teaching behaviour, style, format, or a specialized skill. Rule of thumb: use RAG for **knowledge** (facts that change or are private), fine-tuning for **behaviour**, and combine them when you need both.

## Detailed Explanation
| Dimension | RAG | Fine-tuning |
|---|---|---|
| What changes | Prompt (context) | Model weights |
| Best for | Facts, documents, fresh/private data | Tone, format, domain style, task skills |
| Updating knowledge | Re-index docs (minutes) | Retrain (hours–days) |
| Citations | Easy (cite chunks) | Not possible natively |
| Hallucination control | Grounded in retrieved evidence | Can still invent facts |
| Access control | Filter per user | Everything baked in for all users |
| Upfront cost | Low (embeddings + vector DB) | Data prep + training compute |
| Per-query cost | Higher (longer prompts, retrieval) | Lower prompts |
| Data needed | Raw documents | Hundreds–thousands of labelled examples |
| Failure mode | Bad retrieval | Overfitting, forgetting, stale knowledge |

**Decision guide:**
```
Does the model lack FACTS?            → RAG
Does it answer in the wrong WAY?      → prompt engineering first, then fine-tuning
Both (e.g. medical Q&A in strict JSON)→ RAG + fine-tuned (or well-prompted) model
```

**Why fine-tuning is a poor fact store:** models memorize facts unreliably from a few examples, can't tell you their source, and need retraining whenever facts change. Studies and practitioner experience consistently show RAG outperforms fine-tuning for knowledge injection.

**When fine-tuning wins:** consistent output format, a specific brand voice, classification/extraction tasks, reducing prompt length at high volume, or distilling a large model's behaviour into a smaller, cheaper one. Parameter-efficient methods (LoRA/QLoRA) make it affordable.

**Combined approaches:** fine-tune the *embedding model* or *reranker* on your domain to improve retrieval, or fine-tune the generator to better use retrieved context (RAFT-style training).

## Example
**Analogy:** RAG is giving a lawyer the case files before a meeting: they read and cite them. Fine-tuning is sending the lawyer to a year of specialist training: they think and write like a tax lawyer, but they still need the specific case files.

```python
import anthropic

# RAG: knowledge arrives in the prompt at query time, weights untouched
retrieved = ["Plan Pro costs $49/month (price list, 2026-11-01)."]
prompt = "Context:\n" + "\n".join(retrieved) + "\n\nQuestion: How much is Plan Pro?"

resp = anthropic.Anthropic().messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the provided context...",
    messages=[{"role": "user", "content": prompt}],
)
print(resp.content[0].text)   # e.g. "Plan Pro costs $49 per month."

# Price changes tomorrow? Update the document in the index. No retraining.
# Fine-tuning instead: build a JSONL of examples like
# {"messages": [{"role":"user","content":"How much is Plan Pro?"},
#               {"role":"assistant","content":"$49/month"}]}
# ...then retrain whenever the price changes. That's slow, and the old price may linger.
```

## Common Mistakes / Misconceptions
- **Fine-tuning to "upload documents" into a model.** Unreliable and stale; use RAG.
- **Thinking it's either/or.** Many production systems use both.
- **Jumping to fine-tuning before trying prompt engineering** and few-shot examples.
- **Assuming RAG is free.** Retrieval adds latency, longer prompts, and infrastructure to maintain and evaluate.

## Key Takeaways
- RAG = knowledge via context; fine-tuning = behaviour via weights.
- RAG wins for changing, private, citable facts; fine-tuning wins for style, format, and specialized skills.
- Updating RAG knowledge is fast; updating fine-tuned knowledge needs retraining.
- Combine them when you need both grounded facts and specialized behaviour.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
