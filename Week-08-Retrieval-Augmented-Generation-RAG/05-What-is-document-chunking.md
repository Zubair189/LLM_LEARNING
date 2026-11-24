# What is document chunking?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 5

## Short Answer
Document chunking is splitting long documents into smaller, self-contained pieces (chunks), typically a few hundred tokens each, that are embedded and retrieved individually. Strategies range from simple fixed-size splitting to recursive, sentence-, structure- (headings/Markdown), and semantic-aware splitting. Good chunks are small enough to be specific and large enough to make sense on their own.

## Detailed Explanation
**Scope note:** this question is about *what chunking is and how to do it* (strategies, parameters). Q6 covers *why* we chunk at all.

**Common strategies:**

| Strategy | How it splits | Pros | Cons |
|---|---|---|---|
| Fixed-size (chars/tokens) | Every N tokens | Simple, predictable | Cuts mid-sentence |
| Recursive character | Try `\n\n`, then `\n`, then `. `, then space, until under size | Respects paragraphs; popular default | Still size-driven |
| Sentence-based | Group whole sentences up to a limit | Clean boundaries | Uneven sizes |
| Structure-aware | By headings, Markdown/HTML sections, code functions, table rows | Preserves logical units | Needs parsing |
| Semantic | Split where embedding similarity between sentences drops | Topic-coherent | Slower, more tuning |
| Hierarchical / parent-child | Retrieve small chunks, return their larger parent section | Precise match + full context | More complex |

**Key parameters:**
- **Chunk size:** starting point 200–500 tokens (≈150–400 words). Smaller = more precise matches; larger = more context per hit.
- **Overlap:** 10–20% of size, so facts at boundaries survive (see Q7).
- **Model limits:** don't exceed the embedding model's max sequence length (e.g. 256 word-pieces for `all-MiniLM-L6-v2`; text beyond that is truncated).
- **Metadata per chunk:** source, section heading, page, position, so you can cite and filter.

**Enhancements:** prepend the document title/section path to each chunk ("Employee Handbook > Leave > Vacation: ..."), or use LLM-generated chunk context ("contextual retrieval"); this significantly improves retrieval for ambiguous chunks.

## Example
**Analogy:** Cutting a textbook into index cards for studying. One card per concept is ideal. Cut every 50 words regardless of meaning and cards start mid-sentence; put a whole chapter on one card and you can't find anything specific.

Sentence-aware chunker (pure Python, exact output):

```python
import re

def chunk_sentences(text, max_chars=80):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks

doc = ("Our refund window is 30 days. Refunds go to the original payment method. "
       "Shipping fees are non-refundable. Contact support to start a return.")
for c in chunk_sentences(doc):
    print(len(c), repr(c))
# 72 'Our refund window is 30 days. Refunds go to the original payment method.'
# 68 'Shipping fees are non-refundable. Contact support to start a return.'
```

Library equivalent (LangChain):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(long_text)   # list[str]
```

## Common Mistakes / Misconceptions
- **One-size-fits-all chunking.** Code, tables, FAQs, and legal text need different strategies.
- **Chunks too small** (single sentences without context like "It costs $40.") or **too large** (whole pages that dilute the embedding).
- **Exceeding the embedding model's token limit**, so the tail of each chunk is silently ignored.
- **Never looking at the chunks.** Print a sample and read them; it's the fastest RAG debugging step.

## Key Takeaways
- Chunking splits documents into retrievable units of a few hundred tokens.
- Prefer boundary-aware strategies (recursive, sentence, structure) over blind fixed-size cuts.
- Tune size and overlap with retrieval evaluation, not guesswork.
- Enrich chunks with headings/metadata for better retrieval and citations.

## Related Questions
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
- [What is chunk overlap?](../Week-08-Retrieval-Augmented-Generation-RAG/07-What-is-chunk-overlap.md)
- [What is document ingestion?](../Week-08-Retrieval-Augmented-Generation-RAG/04-What-is-document-ingestion.md)
- [What is the difference between word and sentence embeddings?](../Week-07-Embeddings-and-Vector-Databases/05-What-is-the-difference-between-word-and-sentence-embeddings.md)
