# Why do we convert text into vectors?

> Week 07 – Embeddings and Vector Databases · Question 2

## In One Line
We turn text into vectors (lists of numbers) so a computer can **measure how similar two texts are in meaning**, not just whether they share the same words.

## Simple Explanation
A computer can check if two words are **exactly the same**, but it cannot tell that "car" and "automobile" mean the same thing. Words are just letters to it.

A **vector** is simply a list of numbers, like `[0.8, 0.2]`. When an embedding model turns text into a vector, texts with similar meaning get similar numbers. Now the computer can use maths to compare them.

This unlocks three big things:

1. **Measure similarity:** give a score for how close two texts are in meaning.
2. **Search by meaning:** "cancel my subscription" can find "how to stop your plan", even with zero shared words.
3. **Feed text into AI models:** machine-learning models only accept numbers.

**Why it matters:** keyword search misses synonyms ("refund" vs "money back") and gets fooled by shared words with a different meaning ("apple" the fruit vs Apple the company).

**Link to LLMs:** In RAG, we turn all documents into vectors once. When a user asks a question, we turn it into a vector too and quickly find the documents closest in meaning, then give them to the LLM.

## Real-Life Example
Imagine a library sorted only by book title in A–Z order. To find books about cooking, you would have to guess the titles. Now imagine a library where books are shelved **by topic**. Walk to one cookbook and all the others are right next to it. Vectors give us the second kind of library.

## Code Example
```python
query = "cancel my subscription"
docs = ["how to stop your plan", "subscription to our newsletter is free"]

# 1) Keyword way: count words shared with the query
for d in docs:
    shared = set(query.split()) & set(d.split())   # common words
    print(len(shared), d)
# 0 how to stop your plan                    <- the right answer gets 0!
# 1 subscription to our newsletter is free   <- wrong doc wins

# 2) Vector way: tiny made-up vectors [about_cancelling, about_newsletters]
q_vec = [0.9, 0.1]
doc_vecs = [[0.8, 0.2], [0.1, 0.9]]
for d, v in zip(docs, doc_vecs):
    score = q_vec[0] * v[0] + q_vec[1] * v[1]      # dot product = similarity
    print(round(score, 2), d)
# 0.74 how to stop your plan                 <- right doc wins
# 0.18 subscription to our newsletter is free
```

**What happened:** keyword matching picked the wrong document because it shared the word "subscription". Comparing vectors picked the one with the right **meaning**.

## Watch Out
- Vectors do **not** replace keyword search. For exact things like error codes or product names, keywords are better. Many systems use both (hybrid search).
- Simple word counts are also vectors, but they don't capture meaning. You need a trained embedding model for that.

## Remember
- Computers can do maths on numbers, not on words.
- Vectors let us **measure meaning** and search by it.
- This is the first step of every RAG system.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is hybrid search (keyword + vector)?](../Week-07-Embeddings-and-Vector-Databases/16-BONUS-What-is-hybrid-search-keyword-plus-vector.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
