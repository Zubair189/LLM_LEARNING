# How can we evaluate RAG answer quality?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 15

## In One Line
Test the RAG system with a **set of sample questions** and score two things separately: **did it find the right chunks**, and **did it write a good, faithful answer**?

## Simple Explanation
**Step 1: build a test set.** Write 50–200 realistic questions. For each, note the correct answer and/or which chunks contain it. Include a few questions the documents **can't** answer, to check the system says "I don't know".

**Step 2: score retrieval** (did we find the right chunks?):
- **Precision:** of the chunks we returned, how many were useful?
- **Recall:** of all the useful chunks, how many did we find?
- **MRR (Mean Reciprocal Rank):** how high up was the first useful chunk? 1st place = 1.0, 2nd = 0.5, 3rd = 0.33. Averaged over all test questions.

**Step 3: score generation** (was the answer good?):
- **Faithfulness:** is every claim backed by the retrieved chunks? (No made-up parts.)
- **Answer relevance:** does it actually answer the question?
- **Correctness:** does it match the known right answer?

These are usually scored by an **LLM-as-judge**: another LLM call with clear grading rules. Tools like **Ragas** and **DeepEval** do this for you. Check a sample by hand too, because judges can be wrong.

**Step 4: repeat after every change** (new chunk size, new model, new prompt) and compare the scores.

**Link to LLMs:** scoring retrieval and generation separately tells you **which part** to fix.

## Real-Life Example
Grading an open-book exam on two things: did the student open the **right pages** (retrieval), and did their answer **correctly use** those pages and answer the question (generation)?

## Code Example
Retrieval scores for one test question:

```python
retrieved = ["c3", "c7", "c1", "c9"]   # what the retriever returned, best first
relevant  = {"c1", "c3", "c5"}         # chunks a human marked as truly useful

hits = [c in relevant for c in retrieved]       # [True, False, True, False]
precision = sum(hits) / len(retrieved)          # useful share of what we got
recall = sum(hits) / len(relevant)              # share of useful chunks we found
mrr = 1 / (hits.index(True) + 1)                # 1 / position of first useful hit

print(f"precision={precision:.2f} recall={recall:.2f} MRR={mrr:.2f}")
# precision=0.50 recall=0.67 MRR=1.00
```

**What happened:** 2 of the 4 returned chunks were useful (precision 0.50); we found 2 of the 3 useful chunks (recall 0.67); the very first result was useful (MRR 1.00).

## Watch Out
- Reading a few answers and saying "looks good" is not evaluation. Use a fixed test set and track scores.
- Only scoring the final answer hides **where** things went wrong. Score retrieval and generation separately.

## Remember
- Build a test set, including questions with no answer.
- Retrieval: precision, recall, MRR. Generation: faithfulness, relevance, correctness.
- Re-run the tests after every change.

## Related Questions
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is the difference between retrieval and generation?](../Week-08-Retrieval-Augmented-Generation-RAG/14-What-is-the-difference-between-retrieval-and-generation.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
