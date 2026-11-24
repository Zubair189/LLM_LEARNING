# What is document chunking?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 5

## In One Line
Chunking means **cutting long documents into small pieces** (chunks) that can each be searched on their own.

## Simple Explanation
This question is about **how** to chunk; Q6 explains **why** we chunk.

A **chunk** is a small piece of a document, often a paragraph or a few hundred **tokens** (a token is roughly ¾ of a word). Each chunk gets its own embedding and can be found on its own.

**Common ways to cut:**

1. **Fixed size:** every N words or characters. Simple, but can cut a sentence in half.
2. **By sentences:** add whole sentences until the chunk is full. Clean edges.
3. **By paragraphs or separators:** try to split at blank lines first, then at sentence ends, then at spaces. This "recursive" method is a popular default.
4. **By structure:** split at headings, sections or code functions. Keeps logical parts together.

**Two settings to choose:**
- **Chunk size:** a good start is 200–500 tokens. Smaller = more precise matches; bigger = more context in each chunk.
- **Overlap:** repeat a little text between neighbouring chunks so facts at the edges aren't lost (see Q7).

**Tip:** add the document title or section heading to each chunk, e.g. "Handbook > Leave: …". It helps the search understand what the chunk is about.

**Link to LLMs:** the chunks are what the LLM actually reads, so good chunks lead to good answers.

## Real-Life Example
Making flash cards from a textbook. One card per idea is perfect. If you cut every 50 words no matter what, cards start mid-sentence. If you put a whole chapter on one card, you can't find anything.

## Code Example
A simple sentence-based chunker:

```python
import re

def chunk_sentences(text, max_chars=80):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())      # split after . ! ?
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > max_chars:  # chunk would be too long
            chunks.append(current)                             # save it
            current = s                                        # start a new chunk
        else:
            current = (current + " " + s).strip()              # add sentence to chunk
    if current:
        chunks.append(current)                                 # save the last chunk
    return chunks

doc = ("Our refund window is 30 days. Refunds go to the original payment method. "
       "Shipping fees are non-refundable. Contact support to start a return.")
for c in chunk_sentences(doc):
    print(len(c), c)
# 72 Our refund window is 30 days. Refunds go to the original payment method.
# 68 Shipping fees are non-refundable. Contact support to start a return.
```

**What happened:** the text was split into two chunks of whole sentences, each under 80 characters.

## Watch Out
- Chunks that are **too small** lose meaning ("It costs $40." — what does?). Chunks that are **too big** blur the topic.
- Embedding models have a length limit. Text beyond it is silently cut off, so keep chunks below it.

## Remember
- Chunking = splitting documents into small searchable pieces.
- Prefer splitting at natural edges (sentences, paragraphs, headings).
- Start with 200–500 tokens and a small overlap, then test.

## Related Questions
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
- [What is chunk overlap?](../Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [What is the difference between word and sentence embeddings?](../Week-07-Embeddings-and-Vector-Databases/05-What-is-the-difference-between-word-and-sentence-embeddings.md)
