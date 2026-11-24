# Why do we divide documents into chunks?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 6

## In One Line
We chunk documents so that search can find **the exact paragraph that answers a question**, instead of a whole blurry document.

## Simple Explanation
Q5 explains **how** to chunk; this question explains **why**.

**Reason 1: one vector per document is too blurry.** An embedding squeezes a text into one list of numbers. If a 40-page manual becomes one vector, that vector is an "average" of refunds, shipping, careers, privacy… It isn't close to any single topic.

**Reason 2: length limits.** Embedding models can only read a limited amount of text. Anything longer is cut off and ignored.

**Reason 3: precise answers.** Search returns whole units. If the unit is a whole document, the LLM gets 39 useless pages plus the one useful paragraph.

**Reason 4: cost and focus.** Every word sent to the LLM costs money and time. Five relevant chunks are cheaper and clearer than a huge document.

**Reason 5: sources.** Each chunk keeps labels like page and section, so the answer can say exactly where a fact came from.

**The balance:** very small chunks lack context; very big chunks are blurry. A paragraph or a few hundred tokens is a good middle.

**Link to LLMs:** smaller, focused chunks mean the LLM sees only what matters, which gives better answers.

## Real-Life Example
Looking up "how long to boil an egg" in a cookbook. An index that says "page 47, paragraph 2" is far more useful than one that says "it's somewhere in this book".

## Code Example
```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

# Made-up vectors: [refunds, shipping, careers]
sections = {
    "Refunds: return within 30 days": [1.0, 0.0, 0.0],
    "Shipping: free over $50":        [0.0, 1.0, 0.0],
    "Careers: we are hiring":         [0.0, 0.0, 1.0],
}
# One vector for the WHOLE document = the average of all sections
whole_doc = [sum(v[i] for v in sections.values()) / 3 for i in range(3)]

query = [1.0, 0.0, 0.0]   # "How many days do I have to return an item?"

print(round(cosine(query, whole_doc), 2))     # 0.58 -> blurry match
for text, vec in sections.items():
    print(round(cosine(query, vec), 2), text)
# 1.0 Refunds: return within 30 days   <- sharp match
# 0.0 Shipping: free over $50
# 0.0 Careers: we are hiring
```

**What happened:** the whole-document vector only scored 0.58 because its meaning is mixed. The refunds chunk scored a perfect 1.0.

## Watch Out
- "Long-context models make chunking unnecessary" is a myth. Sending only the right chunks is still cheaper, faster and more accurate.
- Don't cut tables, lists or code in half just to hit a size limit.

## Remember
- Chunks give sharp, focused embeddings.
- They keep prompts small, cheap and relevant.
- They allow exact source citations.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [What is chunk overlap?](../Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
