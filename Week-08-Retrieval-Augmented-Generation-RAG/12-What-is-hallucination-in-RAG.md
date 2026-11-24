# What is hallucination in RAG?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 12

## Short Answer
Hallucination in RAG is when the generated answer contains claims that are not supported by (or contradict) the retrieved context, even though the model was given evidence. It happens when retrieval misses the right chunk and the model fills the gap from memory, when the model misreads or over-generalizes the context, or when it blends multiple sources incorrectly. It's measured as low **faithfulness/groundedness** and reduced through better retrieval, strict prompting, citations, and verification.

## Detailed Explanation
**Two root causes:**

| Type | What went wrong | Example |
|---|---|---|
| Retrieval failure | Right chunk not retrieved, so the model guesses | Asked about 2026 policy, only 2024 chunk retrieved |
| Generation failure | Right chunk present, but model ignores/distorts it | Context says "up to 5 days," answer says "5 days guaranteed" |
| Conflation | Merges facts from different chunks wrongly | Combines Plan A's price with Plan B's features |
| Unsupported extrapolation | Adds plausible details not in context | Invents a phone number or deadline |
| Stale/conflicting sources | Context itself is outdated or contradictory | Two versions of the policy indexed |

**Faithfulness vs correctness:** an answer can be *faithful* (supported by context) but *wrong* (the document is outdated), or *correct* but *unfaithful* (true from the model's memory, but not in the context). RAG aims for both; faithfulness is what you can control and audit.

**Mitigations:**
- **Better retrieval:** hybrid search, reranking, good chunking, metadata filters for latest versions.
- **Prompting:** "answer only from the context; if not present, say you don't know"; require citations per claim.
- **Score thresholds:** don't call the LLM (or force "I don't know") when nothing relevant is retrieved.
- **Verification:** an LLM-as-judge or NLI check that each sentence is supported by cited chunks; Anthropic's Citations feature or quote-first prompting ("extract relevant quotes, then answer").
- **Lower temperature** for factual Q&A.
- **Evaluation:** track faithfulness/groundedness with Ragas, DeepEval, or a custom judge on a test set.

## Example
**Analogy:** A student in an open-book exam who can't find the right page, so writes a confident guess anyway. Or finds the page but misquotes it. The book was available; the answer still wasn't grounded in it.

A simple groundedness check with an LLM-as-judge:

```python
import anthropic

client = anthropic.Anthropic()
context = "Our warranty covers manufacturing defects for 12 months. Accidental damage is not covered."
answer = "The warranty lasts 12 months and covers accidental damage like drops."

judge_prompt = f"""Context:
{context}

Answer to check:
{answer}

List each claim in the answer and mark it SUPPORTED or UNSUPPORTED by the context.
Finish with a line 'FAITHFULNESS: <supported claims>/<total claims>'."""

resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    system="You are a strict fact-checker. Judge only against the provided context.",
    messages=[{"role": "user", "content": judge_prompt}],
)
print(resp.content[0].text)
# e.g.
# 1. Warranty lasts 12 months: SUPPORTED
# 2. Covers accidental damage like drops: UNSUPPORTED (context says not covered)
# FAITHFULNESS: 1/2
```

## Common Mistakes / Misconceptions
- **"RAG solves hallucination."** It reduces it; it doesn't eliminate it.
- **Blaming only the LLM.** Many hallucinations start with retrieval misses; check the retrieved context first.
- **Equating faithfulness with truth.** Faithful answers based on stale documents are still wrong.
- **No "I don't know" path.** Forcing an answer for every question guarantees fabrication.

## Key Takeaways
- RAG hallucination = answer claims not supported by retrieved context.
- Causes: retrieval failure, generation failure, conflation, stale sources.
- Fix with better retrieval, strict grounded prompts, citations, thresholds, and verification.
- Measure faithfulness/groundedness continuously.

## Related Questions
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
- [What is reranking and why does it improve RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/16-BONUS-What-is-reranking-and-why-does-it-improve-RAG.md)
