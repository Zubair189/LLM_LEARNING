# Why do we divide documents into chunks?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 6

## Short Answer
We chunk documents because (1) embedding models have input limits and produce one vector per input, so a whole document's meaning gets blurred into a single vague vector; (2) small focused chunks match specific questions much more precisely; and (3) we can only fit a limited, relevant amount of text into the LLM's prompt. Chunking lets retrieval return just the paragraph that answers the question, which improves accuracy, cost, and citations.

## Detailed Explanation
**Scope note:** Q5 explains *how* to chunk; this question explains *why*.

**1. Embedding limits and dilution.** A 40-page manual embedded as one vector is an "average" of every topic in it, close to nothing in particular. Many models also truncate input (e.g. `all-MiniLM-L6-v2` reads ~256 word-pieces), silently ignoring the rest.

```
Whole doc vector  = avg(refunds, shipping, warranty, careers, privacy...)  → vague
Chunk vectors     = [refunds] [shipping] [warranty] ...                     → sharp
Query "refund window?" matches the [refunds] chunk strongly
```

**2. Retrieval precision.** Search returns *units*. If the unit is a whole document, the LLM gets 39 irrelevant pages plus the one useful paragraph.

**3. Context window, cost, and latency.** Even with very long context windows, every token costs money and time, and models can overlook facts buried in long inputs ("lost in the middle"). Sending 5 relevant chunks (~2k tokens) beats 200k tokens of whole documents.

**4. Citations and trust.** Chunk-level metadata (doc, page, section) lets answers cite exactly where a fact came from.

**5. Efficient updates.** Changing one section only requires re-embedding its chunks.

**The size trade-off:**

| Chunk size | Retrieval precision | Context per hit | Risk |
|---|---|---|---|
| Very small (1 sentence) | High | Too little | Missing context ("it costs $40": what does?) |
| Medium (200–500 tokens) | Good | Good | Sweet spot for most text |
| Very large (pages) | Low | Lots | Diluted embeddings, wasted tokens |

Mitigations: overlap, parent-child retrieval (match small, return large), and adding section titles to chunks.

## Example
**Analogy:** Searching a cookbook for "how long to boil an egg." An index that points to page 47, paragraph 2 (chunks) is far more useful than one that says "it's somewhere in this book" (whole-document retrieval).

```python
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("all-MiniLM-L6-v2")
sections = [
    "Refunds: you can return items within 30 days for a full refund.",
    "Shipping: orders ship in 2 business days; free over $50.",
    "Careers: we are hiring engineers in Berlin and Austin.",
    "Privacy: we never sell your personal data.",
]
whole_doc = " ".join(sections)
query = "How many days do I have to return an item?"

q = m.encode(query, normalize_embeddings=True)
print(round(float(m.encode(whole_doc, normalize_embeddings=True) @ q), 2))
# approx 0.4x  (whole document: diluted)
print([round(float(m.encode(s, normalize_embeddings=True) @ q), 2) for s in sections])
# approx [0.6x, 0.2x, 0.0x, 0.0x]  -> the Refunds chunk is a much sharper match
```

## Common Mistakes / Misconceptions
- **"Long-context models make chunking obsolete."** Retrieval over chunks is still cheaper, faster, and more precise at scale.
- **Chunking purely by size** and splitting tables, lists, or code functions in half.
- **Ignoring the embedding model's max length**: chunks longer than it are truncated.
- **Not evaluating chunk size.** Measure context recall/precision with a few settings.

## Key Takeaways
- Chunks give sharp, specific embeddings instead of a blurred document average.
- They make retrieval precise and keep prompts small, cheap, and focused.
- Chunk-level metadata enables precise citations and cheap updates.
- Chunk size is a precision-vs-context trade-off; tune it with evaluation.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [What is chunk overlap?](../Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
