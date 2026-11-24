# What is chunk overlap?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 7

## In One Line
Chunk overlap means **repeating a few words from the end of one chunk at the start of the next**, so facts on the border don't get cut in half.

## Simple Explanation
When you cut text into chunks, a cut can land in the middle of an important fact:

```
No overlap:   [The premium plan costs 40 dollars] [per month and includes priority support]
                                                   ↑ "40 dollars" is split from "per month"
```

With overlap, the next chunk starts a little earlier, so it repeats the end of the previous chunk:

```
Overlap of 2: [The premium plan costs 40 dollars]
                                     [40 dollars per month and includes]
                                                          [and includes priority support]
```

Now one chunk contains the full fact "40 dollars per month".

**How much overlap?** A common start is **10–20% of the chunk size**. For 500-token chunks, that's 50–100 tokens of overlap.

**The cost:** more overlap means more chunks to store, and very similar chunks may both show up in the results. That wastes space in the prompt.

**Link to LLMs:** overlap helps the retriever find complete facts, so the LLM gets the whole sentence it needs.

## Real-Life Example
Roof tiles overlap a little so rain can't slip through the gaps. Chunk overlap covers the gaps between chunks so facts can't slip through.

## Code Example
```python
def chunk_words(text, size, overlap):
    words = text.split()
    step = size - overlap                     # how far each new chunk moves forward
    return [" ".join(words[i:i + size])
            for i in range(0, max(len(words) - overlap, 1), step)]

t = "The premium plan costs 40 dollars per month and includes priority support"

print(chunk_words(t, 6, 0))   # no overlap
# ['The premium plan costs 40 dollars', 'per month and includes priority support']

print(chunk_words(t, 6, 2))   # 2 words of overlap
# ['The premium plan costs 40 dollars', '40 dollars per month and includes', 'and includes priority support']
```

**What happened:** with overlap, the middle chunk "40 dollars per month and includes" keeps the price and "per month" together, so a question like "monthly price?" can find it.

## Watch Out
- Overlap must be **smaller** than the chunk size, or the chunks never move forward.
- More overlap is not always better. Too much creates near-duplicate chunks and a bigger index.

## Remember
- Overlap = repeated words between neighbouring chunks.
- It keeps border facts in one piece.
- Start with 10–20% of the chunk size.

## Related Questions
- [What is document chunking?](../Week-08-Retrieval-Augmented-Generation-RAG/05-What-is-document-chunking.md)
- [Why do we divide documents into chunks?](../Week-08-Retrieval-Augmented-Generation-RAG/06-Why-do-we-divide-documents-into-chunks.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
