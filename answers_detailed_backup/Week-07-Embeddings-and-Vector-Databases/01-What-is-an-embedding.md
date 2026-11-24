# What is an embedding?

> Week 07 – Embeddings and Vector Databases · Question 1

## Short Answer
An embedding is a dense, fixed-length list of numbers (a vector) produced by a trained model so that items with similar *meaning* end up close together in vector space. For example, `all-MiniLM-L6-v2` maps any sentence to 384 floats, and "How do I reset my password?" lands near "I forgot my login credentials." Embeddings are the bridge that lets software compare meaning using math, and they power semantic search, RAG, clustering, and recommendations.

## Detailed Explanation
- **Learned, not hand-written.** An embedding model (usually a Transformer encoder) is trained, often with *contrastive learning*, to pull related pairs together and push unrelated pairs apart. The numbers have no individual human meaning; the *geometry* (distances and directions) carries the meaning.
- **Dense and fixed size.** Every input, whether 3 words or 300, becomes a vector of the same length (e.g. 384, 768, 1024, 1536, 3072). Most values are non-zero, unlike sparse one-hot or keyword vectors.
- **Many modalities.** Text, images (CLIP), audio, code, and even users/products can be embedded. Multimodal models put text and images in one shared space.
- **Typical pipeline:**

```
"Refund policy for damaged items"
        │  tokenizer
        ▼
  [101, 25416, 3343, ...]       token IDs
        │  Transformer encoder + pooling
        ▼
  [0.021, -0.113, ..., 0.067]   384-dim embedding
```

| Term | Meaning |
|---|---|
| Embedding model | The network that produces vectors (e.g. MiniLM, BGE, E5, Voyage, OpenAI text-embedding-3) |
| Embedding / vector | The output list of floats |
| Embedding space | The geometric space where all vectors live |

**Distinction from Q3 (vector representation):** "vector representation" is the broad idea of describing anything as a list of numbers (including sparse one-hot or TF-IDF counts). An *embedding* is a specific kind: a **learned, dense** vector whose geometry encodes semantic similarity.

## Example
**Analogy:** Think of a map where every city has GPS coordinates. You can't tell much from "48.85, 2.35" alone, but cities that are close in coordinates are close in reality. Embeddings are GPS coordinates for meaning.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "What's the weather in Paris?",
]
emb = model.encode(sentences, normalize_embeddings=True)
print(emb.shape)            # (3, 384)
print(emb[0][:4])           # e.g. [-0.03  0.05  0.01 -0.08]  (values vary)

sims = model.similarity(emb, emb)   # cosine similarity matrix
print(sims[0])  # approx tensor([1.00, 0.6x, 0.0x]) -> password ~ login, not weather
```

## Common Mistakes / Misconceptions
- **"Each dimension means something specific (e.g. dim 17 = sentiment)."** Individual dimensions are rarely interpretable; meaning is distributed.
- **Mixing models.** Vectors from different embedding models (or versions) live in different spaces and cannot be compared. Re-embed everything if you switch models.
- **Assuming embeddings capture exact facts.** They capture topical/semantic closeness; they are weak at exact numbers, negation, and rare identifiers (SKUs, error codes). That is why hybrid search exists.
- **Confusing embeddings with LLM generation.** Embedding models encode; they don't generate text.

## Key Takeaways
- An embedding is a learned, dense, fixed-length vector that encodes meaning as geometry.
- Similar meaning → nearby vectors; similarity is measured with cosine similarity or dot product.
- Always use the same model for indexing and querying.
- Embeddings are the foundation of semantic search, vector databases, and RAG.

## Related Questions
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
- [What is cosine similarity?](../Week-07-Embeddings-and-Vector-Databases/07-What-is-cosine-similarity.md)
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
