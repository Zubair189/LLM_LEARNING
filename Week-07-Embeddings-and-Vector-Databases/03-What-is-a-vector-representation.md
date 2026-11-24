# What is a vector representation?

> Week 07 – Embeddings and Vector Databases · Question 3

## Short Answer
A vector representation is any way of describing an object (a word, document, image, user) as an ordered list of numbers so a computer can process it mathematically. Representations range from simple, sparse, hand-designed ones (one-hot, bag-of-words, TF-IDF) to learned, dense ones (embeddings). The choice of representation decides what "similar" means: shared words for sparse vectors, shared meaning for dense embeddings.

## Detailed Explanation
**How this differs from Q1:** Q1 ("What is an embedding?") is about one *specific* kind of vector representation, the learned dense kind. This question is the umbrella concept: *every* embedding is a vector representation, but not every vector representation is an embedding.

| Representation | Dense/Sparse | Length | Captures meaning? | Example use |
|---|---|---|---|---|
| One-hot | Sparse | Vocabulary size (50k+) | No: all words equally distant | Old NLP, categorical features |
| Bag-of-words / counts | Sparse | Vocabulary size | Only word overlap | Spam filters |
| TF-IDF / BM25 | Sparse | Vocabulary size | Weighted word overlap | Keyword search engines |
| Learned sparse (SPLADE) | Sparse | Vocabulary size | Partly (term expansion) | Hybrid retrieval |
| Dense embedding | Dense | 256–3072 | Yes, semantic | Semantic search, RAG |

```
one-hot "cat"  = [0, 0, 1, 0, 0, ..., 0]      (50,000 dims, one 1)
TF-IDF doc     = [0, 0.41, 0, 0, 0.88, ..., 0] (mostly zeros)
embedding "cat"= [0.12, -0.53, 0.07, ..., 0.31] (384 dims, all non-zero)
```

**Key properties of a good representation:**
- **Fixed length**, so items can be compared and stored in an index.
- **Similarity-preserving**: similar items get similar vectors under the chosen metric.
- **Efficient**: sparse vectors are cheap to store when mostly zeros; dense vectors are compact and GPU-friendly.

Modern retrieval often uses **both**: a sparse vector (BM25/SPLADE) for exact terms and a dense vector for meaning, i.e. hybrid search.

## Example
**Analogy:** Describing a house. A sparse representation is a checklist of 10,000 features with ticks ("has pool: 1, has moat: 0..."). A dense representation is a compact profile of 8 scores ("size 0.7, luxury 0.4, urban 0.9...") learned from what buyers actually consider similar.

Pure-Python bag-of-words (no model needed):

```python
docs = ["the cat sat", "the dog sat", "a cat purred"]
vocab = sorted({w for d in docs for w in d.split()})
print(vocab)
# ['a', 'cat', 'dog', 'purred', 'sat', 'the']

def bow(text):
    words = text.split()
    return [words.count(v) for v in vocab]

for d in docs:
    print(d, "->", bow(d))
# the cat sat  -> [0, 1, 0, 0, 1, 1]
# the dog sat  -> [0, 0, 1, 0, 1, 1]
# a cat purred -> [1, 1, 0, 1, 0, 0]
```

Here "the cat sat" and "the dog sat" look most similar (they share 2 words) even though "a cat purred" is about the same animal. A dense embedding from `all-MiniLM-L6-v2` (`model.encode("the cat sat")`, shape `(384,)`) would capture the cat-to-cat relationship instead.

## Common Mistakes / Misconceptions
- **Using "vector representation" and "embedding" as exact synonyms.** Embeddings are a subset: learned and dense.
- **Thinking sparse = obsolete.** BM25 still wins on exact identifiers and is a core part of hybrid search.
- **Comparing vectors from different representations** (e.g. TF-IDF vs embedding). They live in different spaces.

## Key Takeaways
- A vector representation turns any object into a fixed-length list of numbers.
- Sparse representations capture word overlap; dense embeddings capture meaning.
- The representation defines what "similar" means for your system.
- Production retrieval often combines sparse and dense vectors.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [Why do we convert text into vectors?](../Week-07-Embeddings-and-Vector-Databases/02-Why-do-we-convert-text-into-vectors.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
