# What is the difference between word and sentence embeddings?

> Week 07 – Embeddings and Vector Databases · Question 5

## Short Answer
A word embedding gives one vector per word (or token), while a sentence embedding gives one vector for a whole sentence, paragraph, or chunk. Classic word embeddings (Word2Vec, GloVe) are *static*: "bank" always has the same vector regardless of context. Sentence embedding models (Sentence-BERT, MiniLM, BGE, E5) read the full text with a Transformer and pool the result into a single context-aware vector, which is what you need for semantic search and RAG.

## Detailed Explanation
| Aspect | Word embeddings | Sentence embeddings |
|---|---|---|
| Unit | Single word/token | Sentence, paragraph, chunk |
| Classic models | Word2Vec (2013), GloVe, fastText | Sentence-BERT (2019), all-MiniLM-L6-v2, BGE, E5, GTE |
| Context | Static (one vector per word type) | Contextual (whole input considered) |
| Typical dims | 50–300 | 384–3072 |
| Word order | Ignored if you average | Captured by the Transformer |
| Main use today | Teaching, lightweight features | Semantic search, RAG, clustering |

**Three stages of evolution:**
1. **Static word vectors** (Word2Vec/GloVe): famous for analogies like `king − man + woman ≈ queen`. But "bank" in "river bank" and "bank account" is the same vector.
2. **Contextual token vectors** (BERT): each token gets a vector that depends on its neighbours, so the two "bank"s differ. Still one vector *per token*.
3. **Sentence embeddings**: a model fine-tuned (often contrastively, on question/answer or paraphrase pairs) to output one vector for the whole text, usually via **mean pooling** over token vectors:

```
"the river bank flooded"
  tokens → [v_the, v_river, v_bank, v_flooded]   (contextual token vectors)
  mean pooling → one 384-dim sentence vector
```

**Why not just average Word2Vec vectors?** It's a cheap baseline, but it ignores word order and negation ("dog bites man" = "man bites dog"), and raw BERT token averages without contrastive fine-tuning perform poorly for similarity. Purpose-trained sentence models are far better.

## Example
**Analogy:** Word embeddings are a dictionary: each word has one definition. Sentence embeddings are a reader who reads the whole sentence and summarizes its meaning, so "bank" is understood from context.

```python
# Word embeddings (static): pip install gensim
import gensim.downloader as api
wv = api.load("glove-wiki-gigaword-50")          # 50-dim word vectors
print(wv["bank"].shape)                           # (50,)
print(wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1))
# [('queen', 0.85...)]

# Sentence embeddings (contextual)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
s = model.encode(["I deposited cash at the bank",
                  "We had a picnic on the river bank",
                  "I withdrew money from my account"],
                 normalize_embeddings=True)
print(s.shape)                                    # (3, 384)
print(round(float(s[0] @ s[2]), 2), round(float(s[0] @ s[1]), 2))
# approx 0.5x vs 0.2x -> the money sentences are closer, despite "bank" appearing in both 0 and 1
```

## Common Mistakes / Misconceptions
- **Averaging word vectors and expecting sentence-level quality.** It loses order and negation.
- **Using raw BERT `[CLS]` output as a sentence embedding** without a model fine-tuned for similarity; results are often worse than GloVe averages.
- **Embedding whole documents as one "sentence."** Models truncate at their max length (MiniLM: 256 tokens); chunk first.
- **Thinking word embeddings are useless now.** They are still a good teaching tool and a cheap feature for small models.

## Key Takeaways
- Word embeddings: one vector per word, usually static.
- Sentence embeddings: one context-aware vector per text span, produced by pooling a Transformer's outputs.
- For search and RAG, use a sentence embedding model.
- Mind max sequence length; long text must be chunked.

## Related Questions
- [What is an embedding?](../Week-07-Embeddings-and-Vector-Databases/01-What-is-an-embedding.md)
- [What is semantic similarity?](../Week-07-Embeddings-and-Vector-Databases/06-What-is-semantic-similarity.md)
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
