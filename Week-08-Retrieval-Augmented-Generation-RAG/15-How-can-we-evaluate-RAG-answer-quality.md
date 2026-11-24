# How can we evaluate RAG answer quality?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 15

## Short Answer
Evaluate RAG at two levels using a test set of realistic questions (ideally with reference answers or known relevant chunks). For **retrieval**, measure context precision, context recall, recall@k, and MRR. For **generation**, measure faithfulness/groundedness (is every claim supported by the context?), answer relevance (does it address the question?), and correctness against a reference. Automate scoring with tools like Ragas, DeepEval, or TruLens, or with an LLM-as-judge, and spot-check with human review.

## Detailed Explanation
**The RAG triad + retrieval metrics:**

| Metric | Measures | Stage | Needs reference? |
|---|---|---|---|
| **Context precision** | Are retrieved chunks relevant, and ranked high? | Retrieval | Relevance labels or reference answer |
| **Context recall** | Did we retrieve everything needed to answer? | Retrieval | Reference answer / labels |
| Recall@k, MRR, nDCG | Classic IR ranking quality | Retrieval | Relevant-chunk labels |
| **Faithfulness / groundedness** | Are answer claims supported by retrieved context? | Generation | No |
| **Answer relevance** | Does the answer address the question? | Generation | No |
| Answer correctness | Matches ground truth? | End-to-end | Yes |
| Latency, cost, refusal rate | Operational quality | System | No |

```
            question
           /        \
  context relevance  answer relevance
         /              \
   context ──faithfulness── answer          (the "RAG triad")
```

**How to build an evaluation process:**
1. **Golden dataset:** 50–200 real or synthetic questions (Ragas and other tools can generate synthetic Q/A from your docs), with reference answers and/or relevant chunk IDs. Include unanswerable questions to test "I don't know."
2. **Automatic scoring:** run retrieval metrics in code; use LLM-as-judge for faithfulness and relevance, with a clear rubric, a strong judge model, and structured output.
3. **Calibrate the judge** against a sample of human labels; watch for position/verbosity bias.
4. **Regression testing:** rerun on every change (chunk size, embedding model, prompt, reranker) and compare.
5. **Production monitoring:** log queries, retrieved chunks, answers, and user feedback (thumbs up/down); sample for review.

## Example
**Analogy:** Grading an open-book exam on two things: did the student open the right pages (retrieval), and did their answer faithfully use those pages and actually answer the question (generation)?

Retrieval metrics (exact output):

```python
retrieved = ["c3", "c7", "c1", "c9"]        # ranked results for one question
relevant  = {"c1", "c3", "c5"}              # labelled relevant chunks

hits = [d in relevant for d in retrieved]
precision = sum(hits) / len(retrieved)
recall = sum(hits) / len(relevant)
mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
# rank-aware context precision (mean of precision@i at each relevant position)
ap = [sum(hits[:i + 1]) / (i + 1) for i, h in enumerate(hits) if h]
ctx_precision = sum(ap) / len(ap)
print(f"precision@4={precision:.2f} recall@4={recall:.2f} MRR={mrr:.2f} "
      f"context_precision={ctx_precision:.2f}")
# precision@4=0.50 recall@4=0.67 MRR=1.00 context_precision=0.83
```

Generation metrics with LLM-as-judge:

```python
import json, anthropic

def judge(question, context, answer):
    prompt = (f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\n"
              "Score 1-5 for FAITHFULNESS (all claims supported by context) and "
              "RELEVANCE (answers the question). Reply with only JSON: "
              '{"faithfulness": n, "relevance": n, "reason": "..."}')
    resp = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5", max_tokens=1024,
        system="You are a strict RAG evaluator. Judge only against the provided context.",
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.content[0].text)

print(judge("What is the refund window?",
            "Refunds are accepted within 30 days of delivery.",
            "You can get a refund within 30 days, and shipping is always free."))
# e.g. {'faithfulness': 3, 'relevance': 5, 'reason': 'Free shipping claim is not in context.'}
```

## Common Mistakes / Misconceptions
- **Eyeballing a few answers** instead of a fixed test set and tracked scores.
- **Measuring only end-to-end correctness**, which hides whether retrieval or generation failed.
- **Trusting LLM judges blindly.** Calibrate against human labels and use clear rubrics.
- **Ignoring unanswerable questions**: a good system should refuse rather than invent.

## Key Takeaways
- Evaluate retrieval (context precision/recall, recall@k, MRR) and generation (faithfulness, answer relevance, correctness) separately.
- Build a golden dataset, including unanswerable questions.
- Automate with Ragas/DeepEval/TruLens or LLM-as-judge, calibrated by humans.
- Run evals on every change and monitor production with feedback.

## Related Questions
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is the difference between retrieval and generation?](../Week-08-Retrieval-Augmented-Generation-RAG/14-What-is-the-difference-between-retrieval-and-generation.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
