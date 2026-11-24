# What is an embedding dimension?

> Week 07 – Embeddings and Vector Databases · Question 4

## In One Line
The embedding dimension is **how many numbers are in each embedding**, for example 384 or 1536.

## Simple Explanation
An embedding is a list of numbers. The **dimension** is simply the **length of that list**. Each number is one "dimension".

1. **The model decides it.** `all-MiniLM-L6-v2` always gives 384 numbers. Bigger models give 768, 1024, 1536 or 3072. You don't choose it per sentence: a short sentence and a long paragraph both get the same length.
2. **More numbers = more detail.** A longer list can capture finer differences in meaning.
3. **More numbers = more cost.** Longer lists need more storage, more memory and slower search.

**A quick storage sum:** each number usually takes 4 bytes. So one million chunks × 384 numbers × 4 bytes ≈ **1.5 GB**. With 3072 numbers it is about **12 GB**.

**Bigger is not always better.** A good 768-number model can beat a weaker 1536-number one. Test on your own data.

**Link to LLMs:** when you build a RAG system, the vector database must be set up for your model's dimension. A 384-number index cannot store 768-number vectors.

## Real-Life Example
Describing a wine. With 3 details (red or white, sweet or dry, price) you can roughly group wines. With 50 details (oak, acidity, region…) an expert can tell very similar wines apart, but the notes take longer to write and compare.

## Code Example
```python
# A tiny "embedding" with 4 dimensions (real models: 384, 768, 1536...)
vec = [0.12, -0.53, 0.07, 0.31]
print(len(vec))                 # 4  <- this is the dimension

# Storage: each number (float32) takes 4 bytes
dims = 384                      # e.g. all-MiniLM-L6-v2
num_chunks = 1_000_000          # one million text chunks
total_bytes = num_chunks * dims * 4
print(total_bytes / 1e9, "GB")  # 1.536 GB

dims = 3072                     # a large API model
print(num_chunks * dims * 4 / 1e9, "GB")   # 12.288 GB
```

**What happened:** the dimension is just `len(vector)`. Multiplying it by the number of chunks and 4 bytes tells you how much storage you need.

## Watch Out
- **Dimension mismatch:** if you switch to a model with a different dimension, you must re-embed everything and rebuild the index.
- Dimension is **not** the same as context length. Dimension = size of the output list; context length = how much text the model can read.

## Remember
- Dimension = number of values in each embedding.
- It is fixed by the model.
- Higher dimension = more detail but more storage and slower search.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is a vector representation?](../Week-07-Embeddings-and-Vector-Databases/03-What-is-a-vector-representation.md)
- [What is approximate nearest neighbor search?](../Week-07-Embeddings-and-Vector-Databases/14-What-is-approximate-nearest-neighbor-search.md)
- [What is a vector database?](../Week-07-Embeddings-and-Vector-Databases/10-What-is-a-vector-database.md)
