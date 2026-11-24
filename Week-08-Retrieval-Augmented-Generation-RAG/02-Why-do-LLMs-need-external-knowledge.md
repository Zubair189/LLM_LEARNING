# Why do LLMs need external knowledge?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 2

## Short Answer
An LLM only "knows" what was compressed into its weights during training, so it has a knowledge cutoff, has never seen your private data, and stores facts imprecisely, which leads to confident hallucinations. External knowledge (documents, databases, APIs, web search) supplied at query time gives the model current, specific, verifiable facts. That is why RAG and tool calling are standard in production LLM apps.

## Detailed Explanation
**Limitations of parametric (in-weights) knowledge:**

| Limitation | Example | External knowledge fix |
|---|---|---|
| Knowledge cutoff | Doesn't know last week's price change | Retrieve current docs / call an API |
| No private data | Never saw your internal wiki, contracts, tickets | Index company documents |
| Imprecise recall | Mixes up version numbers, statistics | Quote the exact source |
| Long-tail facts | Rare products, local regulations | Retrieve niche documents |
| No provenance | Can't say *where* a fact came from | Cite retrieved chunks |
| Per-user data | Doesn't know *your* order history | Query user-scoped databases |
| Permissions | Can't "forget" or restrict facts per user | Filter retrieval by access rights |

**Parametric vs non-parametric memory:**
```
Parametric (weights):    broad language skill, reasoning, general facts  → frozen at training
Non-parametric (index):  specific, current, private facts                → update anytime
```
LLMs are excellent *reasoners and writers*; external knowledge makes them reliable *fact users*.

**Why not just retrain?** Retraining or fine-tuning for every new document is slow, expensive, doesn't reliably store facts, and can't enforce per-user access. Updating an index takes seconds.

**Why not just paste everything into a long context window?** Context windows are large now, but stuffing hundreds of documents is costly, slower, hits limits for big corpora, and models can miss facts buried in the middle ("lost in the middle"). Retrieval selects the few passages that matter.

**Forms of external knowledge:** vector-store RAG, keyword/web search, SQL queries, APIs via tool calling, knowledge graphs, and MCP servers connecting agents to live systems.

## Example
**Analogy:** A brilliant new employee who graduated last year. They reason well, but they don't know your company's products, this quarter's prices, or yesterday's policy update. You give them access to the intranet, and suddenly their answers are accurate.

```python
import anthropic

client = anthropic.Anthropic()
question = "What is the late-return fee at Northwind Library?"

# Without external knowledge: model can only guess
# With external knowledge: inject the actual policy
policy = "Northwind Library policy (updated 2026-10-01): late returns cost $0.25 per day, max $10."
resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="Answer only from the provided context. If it's not there, say you don't know.",
    messages=[{"role": "user", "content": f"Context:\n{policy}\n\nQuestion: {question}"}],
)
print(resp.content[0].text)
# e.g. "The late-return fee is $0.25 per day, up to a maximum of $10."
```

Without the context, a well-behaved model should say it doesn't know; a poorly prompted one may invent a plausible fee.

## Common Mistakes / Misconceptions
- **"Bigger models know everything."** Size doesn't fix cutoffs or private data.
- **"Fine-tuning is how you add facts."** Fine-tuning mostly shapes behaviour/style; it's unreliable for injecting facts.
- **"Long context makes retrieval unnecessary."** It helps, but cost, latency, scale, and accuracy still favour retrieval.
- **Trusting any external source blindly.** Retrieved content can be outdated or malicious (prompt injection).

## Key Takeaways
- LLM knowledge is frozen, general, and imprecise; real apps need current, specific, private facts.
- External knowledge is provided at inference time via retrieval, tools, or APIs.
- It enables citations, freshness, and access control.
- RAG is the most common way to supply external knowledge.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
