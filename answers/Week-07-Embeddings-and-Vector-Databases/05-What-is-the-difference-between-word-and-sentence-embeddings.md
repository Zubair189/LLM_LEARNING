# What is the difference between word and sentence embeddings?

> Week 07 – Embeddings and Vector Databases · Question 5

## In One Line
A **word embedding** gives one vector for each word; a **sentence embedding** gives one vector for a whole sentence or paragraph.

## Simple Explanation
**Word embeddings** came first (Word2Vec, GloVe, around 2013). Every word gets its own fixed list of numbers. They became famous for tricks like `king − man + woman ≈ queen`.

But they have two problems:

1. **No context.** "bank" has the same vector in "river bank" and "bank account".
2. **Whole sentences are hard.** If you just average the word vectors, word order is lost, so "dog bites man" looks the same as "man bites dog".

**Sentence embeddings** fix this. A model such as Sentence-BERT or `all-MiniLM-L6-v2` reads the **whole sentence at once** using a **Transformer** (the same kind of network inside LLMs). It understands each word from its neighbours, then combines everything into **one vector** for the full text.

So "I deposited cash at the bank" ends up close to "I withdrew money from my account" and far from "a picnic on the river bank".

**Link to LLMs:** for semantic search and RAG, we compare whole chunks of text with whole questions, so we use **sentence embeddings**.

## Real-Life Example
Word embeddings are like a dictionary: every word has one fixed definition. Sentence embeddings are like a friend who reads your whole sentence and tells you what you **meant**, so "bank" is understood from context.

## Code Example
Why simply averaging word vectors is not enough:

```python
# Word embeddings: one small vector per word (made-up numbers)
word_vec = {
    "dog":   [0.9, 0.1],
    "bites": [0.2, 0.8],
    "man":   [0.7, 0.3],
}

def average_sentence(sentence):
    vecs = [word_vec[w] for w in sentence.split()]          # look up each word
    return [round(sum(v[i] for v in vecs) / len(vecs), 2)   # average each position
            for i in range(2)]

print(average_sentence("dog bites man"))   # [0.6, 0.4]
print(average_sentence("man bites dog"))   # [0.6, 0.4]  <- same! order is lost
```

**What happened:** two sentences with opposite meanings got the same vector. A real sentence embedding model reads word order and context, so it gives them different vectors.

## Watch Out
- Sentence models have a **length limit** (e.g. about 256 word-pieces for MiniLM). Longer text gets cut off, so split long documents into chunks first.
- Word embeddings are still a nice teaching tool, but use sentence embeddings for search.

## Remember
- Word embedding = one vector per word, same in every context.
- Sentence embedding = one vector per whole text, aware of context.
- For search and RAG, use sentence embeddings.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
