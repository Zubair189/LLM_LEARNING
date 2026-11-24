# What is a vector representation?

> Week 07 – Embeddings and Vector Databases · Question 3

## In One Line
A vector representation is **any way of describing something as a list of numbers** so a computer can work with it.

## Simple Explanation
A **vector** is just an ordered list of numbers, like `[0, 1, 0, 1]`. A **vector representation** means we describe a word, sentence, picture or user with such a list.

There are two main kinds:

1. **Sparse vectors (counting words):** one slot for every word in the vocabulary (the list of all known words). Most slots are 0. Examples: **one-hot** (a single 1 marks the word) and **bag-of-words** (counts how often each word appears). They only know which words are shared, not what they mean.
2. **Dense vectors (embeddings):** a shorter list (e.g. 384 numbers) where almost every number is non-zero. A trained model chooses the numbers so that similar **meaning** gives similar vectors.

**How this differs from Q1:** "vector representation" is the big umbrella idea. An **embedding** (Q1) is one special kind: learned and dense. Every embedding is a vector representation, but not every vector representation is an embedding.

**Why it matters:** the kind of vector you choose decides what "similar" means in your system: **shared words** (sparse) or **shared meaning** (dense).

**Link to LLMs:** RAG systems mostly use dense embeddings, often together with a sparse keyword method (this mix is called hybrid search).

## Real-Life Example
Describing a house. A sparse description is a long checklist: "has pool: 1, has moat: 0, has tower: 0…", mostly zeros. A dense description is a short profile of scores: "size 0.7, luxury 0.4, city 0.9". The short profile is easier to compare.

## Code Example
A bag-of-words vector, built by hand:

```python
docs = ["the cat sat", "the dog sat", "a cat purred"]

# Build the word list (vocabulary) from all documents
vocab = sorted({w for d in docs for w in d.split()})
print(vocab)  # ['a', 'cat', 'dog', 'purred', 'sat', 'the']

def bag_of_words(text):
    words = text.split()
    return [words.count(v) for v in vocab]   # count each vocab word

for d in docs:
    print(d, "->", bag_of_words(d))
# the cat sat -> [0, 1, 0, 0, 1, 1]
# the dog sat -> [0, 0, 1, 0, 1, 1]
# a cat purred -> [1, 1, 0, 1, 0, 0]
```

**What happened:** "the cat sat" and "the dog sat" look most alike because they share two words, even though "a cat purred" is about the same animal. A dense embedding would notice the cat-to-cat link instead.

## Watch Out
- "Vector representation" and "embedding" are not exact synonyms. Embeddings are the learned, dense kind.
- Sparse vectors are not useless. They are still the best at matching exact words like names and codes.

## Remember
- Vector representation = anything described as a list of numbers.
- Sparse = word counts (mostly zeros); dense = learned meaning.
- Your choice decides what "similar" means.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [Why do we convert text into vectors?](../Week-07-Embeddings-and-Vector-Databases/02-Why-do-we-convert-text-into-vectors.md)
- [What is an embedding dimension?](../Week-07-Embeddings-and-Vector-Databases/04-What-is-an-embedding-dimension.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
