# What is semantic similarity?

> Week 07 – Embeddings and Vector Databases · Question 6

## Short Answer
Semantic similarity is how close two pieces of text are in *meaning*, regardless of the exact words used. "The flight was delayed" and "My plane left late" share almost no words but are semantically very similar. In practice we estimate it by embedding both texts and computing a score such as cosine similarity, where higher means more alike in meaning.

## Detailed Explanation
- **Lexical vs semantic:**

| Pair | Lexical overlap | Semantic similarity |
|---|---|---|
| "flight delayed" / "plane left late" | Low | High |
| "Apple stock rose" / "apple pie recipe" | High ("apple") | Low |
| "I love this" / "I don't love this" | Very high | Mixed (topic same, sentiment opposite) |

- **How it's computed:** `score = cosine(embed(A), embed(B))`. Typical ranges for modern sentence models: ~0.7+ for paraphrases, ~0.3–0.6 for related topics, ~0–0.2 for unrelated. These ranges vary by model, so calibrate thresholds on your own data.
- **What "similar" means depends on training.** A model trained on question→answer pairs (asymmetric retrieval) treats a question and its answer as similar; a paraphrase model treats rewordings as similar. Some models (E5, BGE) expect prefixes such as `"query: "` / `"passage: "` for best results.
- **Related but distinct notions:**
  - *Semantic similarity* (same meaning) vs *relatedness* ("coffee" and "cup" are related but not synonyms).
  - *Relevance* (does this passage answer the question?) is what RAG actually needs. Embedding similarity is a proxy for relevance; **rerankers** (cross-encoders) estimate relevance more directly.
- **Known weak spots:** negation, numbers, dates, and rare names. Embeddings often score "Python 3.12 release" and "Python 3.8 release" as nearly identical.

## Example
**Analogy:** Two people describing the same movie in completely different words. A good listener recognizes they mean the same film; that recognition is semantic similarity.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
pairs = [
    ("The flight was delayed", "My plane left late"),
    ("Apple stock rose today", "How to bake an apple pie"),
    ("I love this phone", "I don't love this phone"),
]
for a, b in pairs:
    ea, eb = model.encode([a, b], normalize_embeddings=True)
    print(f"{float(ea @ eb):.2f}  {a!r} vs {b!r}")
# approx output (exact values vary by version):
# 0.6x  'The flight was delayed' vs 'My plane left late'
# 0.2x  'Apple stock rose today' vs 'How to bake an apple pie'
# 0.8x  'I love this phone' vs "I don't love this phone"   <- negation weakness!
```

The third pair shows why you should not treat similarity as "agreement": embeddings focus on topic.

## Common Mistakes / Misconceptions
- **Using a universal threshold** like "0.8 = match" across models. Score distributions differ per model.
- **Equating similarity with relevance or truth.** A similar passage may not answer the question, or may be outdated.
- **Expecting embeddings to handle negation or exact numbers.** Use keyword filters, metadata, or rerankers.
- **Forgetting model-specific prefixes/instructions** (E5, BGE, Nomic), which silently lowers quality.

## Key Takeaways
- Semantic similarity measures closeness of meaning, not shared words.
- It is computed via embeddings plus a similarity metric (usually cosine).
- It's a proxy for relevance; reranking and evaluation close the gap.
- Calibrate thresholds per model and per dataset.

## Related Questions
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [Why do we convert text into vectors?](../Week-07-Embeddings-and-Vector-Databases/02-Why-do-we-convert-text-into-vectors.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
