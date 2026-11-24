# What is chunk overlap?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 7

## Short Answer
Chunk overlap is the number of tokens (or characters/words) repeated between the end of one chunk and the start of the next. It prevents information that straddles a chunk boundary from being split apart and lost, so a fact like "the plan costs 40 dollars per month" stays intact in at least one chunk. A typical setting is 10–20% of chunk size (e.g. 500-token chunks with 50–100 tokens of overlap).

## Detailed Explanation
```
No overlap (size 6):   [The premium plan costs 40 dollars] [per month and includes priority support]
                                                     ↑ "40 dollars" separated from "per month"

Overlap 2 (size 6):    [The premium plan costs 40 dollars]
                                         [40 dollars per month and includes]
                                                            [and includes priority support]
```

**Mechanics:** with size `S` and overlap `O`, each new chunk starts `S − O` units after the previous one (the *stride*).

**Trade-offs:**

| Overlap | Benefit | Cost |
|---|---|---|
| 0% | Fewest chunks, no duplication | Boundary facts split |
| 10–20% | Good boundary safety | ~10–25% more chunks/storage |
| 50%+ | Very robust | ~2x chunks, near-duplicate results crowd top-k |

Number of chunks ≈ `ceil((N − O) / (S − O))` for N tokens. For N = 10,000, S = 500: O = 0 → 20 chunks; O = 100 → 25 chunks.

**When overlap matters less:** structure-aware or sentence-based chunking already avoids mid-sentence cuts, so a small overlap (or one sentence) is enough. Parent-child retrieval and adding section headers also reduce dependence on overlap.

**Side effect:** neighbouring overlapping chunks can both appear in the top-k, wasting context slots. Deduplicate or merge adjacent hits before building the prompt.

## Example
**Analogy:** Overlapping roof shingles: each shingle covers the edge of the one below so no rain gets through the seams. Overlap covers the "seams" between chunks so no fact falls through.

```python
def chunk_words(text, size, overlap):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size])
            for i in range(0, max(len(words) - overlap, 1), step)]

t = "The premium plan costs 40 dollars per month and includes priority support"
print(chunk_words(t, 6, 0))
# ['The premium plan costs 40 dollars', 'per month and includes priority support']
print(chunk_words(t, 6, 2))
# ['The premium plan costs 40 dollars', '40 dollars per month and includes',
#  'and includes priority support']
```

A query like "monthly price?" now has a chunk containing "40 dollars per month."

With LangChain: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)`.

## Common Mistakes / Misconceptions
- **Overlap ≥ chunk size**, which creates an infinite loop or a stride of zero; libraries raise an error.
- **Huge overlap "just to be safe"**: bloats the index and fills top-k with near-duplicates.
- **Thinking overlap fixes bad chunking.** It patches boundaries; it doesn't fix splitting tables or sections in the wrong place.
- **Mixing units**: setting size in tokens but overlap in characters.

## Key Takeaways
- Overlap repeats content across chunk boundaries so facts aren't split.
- Start with 10–20% of chunk size; stride = size − overlap.
- More overlap = more storage and duplicate hits; deduplicate before prompting.
- Boundary-aware chunking reduces how much overlap you need.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
