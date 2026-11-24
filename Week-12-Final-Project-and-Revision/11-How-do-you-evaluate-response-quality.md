# How do you evaluate response quality?

> Week 12 – Final Project and Revision · Question 11

> **How this differs from Week 11 Q13 ([What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)):** Q13 covers the *discipline* of evaluating models and systems overall — benchmarks, eval datasets, offline vs online evaluation, regression testing. This question is the *practical rubric* for judging whether an **individual response** is good, pulling together everything from the course: faithfulness, relevance, completeness, format, and safety.

## Short Answer
Judge each response against a **rubric**: is it **faithful** (supported by the provided context, no hallucinations), **relevant** (answers the actual question), **complete**, **correctly formatted** (valid JSON/schema, citations present), and **safe** (no PII leaks, policy violations, or injected instructions followed)? Check the cheap, objective parts with **code** (schema validation, citation checks, regex), the subjective parts with an **LLM-as-judge** calibrated against **human review**, and track scores over a fixed test set plus live user feedback.

## Detailed Explanation

### End-to-end evaluation flow

```
          Question + retrieved context + response
                         │
      ┌──────────────────┼────────────────────────────┐
      ▼                  ▼                            ▼
 Code checks        LLM-as-judge (rubric)        Human review
 - JSON schema      - faithfulness 1-5          - sample 5-10%
 - citations exist  - relevance 1-5             - calibrate judge
 - length, PII      - completeness 1-5          - label hard cases
 - banned content   - reason for each score
      │                  │                            │
      └──────────┬───────┴────────────┬───────────────┘
                 ▼                    ▼
        Per-response score      Aggregate dashboards
        pass / fail + reasons   (by prompt version, model, topic)
                 │                    │
                 ▼                    ▼
   Block/flag in production    Regression gate in CI before deploy
```

### The rubric

| Dimension | Question to ask | How to check |
|---|---|---|
| Faithfulness / groundedness | Is every claim supported by the context? | LLM judge with context; citation-to-chunk match |
| Relevance | Does it answer *this* question? | LLM judge; embedding similarity question↔answer (weak signal) |
| Completeness | Are all parts of the question covered? | Judge vs reference answer / key points list |
| Correctness | Is it factually right? | Compare to gold answer (exact match, F1, or judge) |
| Format | Valid JSON, required fields, citations, length | Pydantic / JSON Schema, regex — deterministic |
| Safety | No PII leaks, toxic content, followed injected instructions? | Classifiers, regex for secrets/PII, judge |
| Tone & helpfulness | Matches brand voice, actionable | Judge + human |
| Efficiency | Latency, tokens, cost | Logs |

For RAG, also score **retrieval** separately (was the right chunk retrieved?) so you know whether to fix retrieval or generation.

### Methods compared
- **Reference-based** (gold answers): reliable but needs labeled data.
- **Reference-free** (judge only uses question + context): scales, but must be calibrated.
- **Human review**: ground truth for subjective quality; expensive, so sample.
- **Online signals**: thumbs up/down, re-asks, escalations to humans — noisy but real.

### Making LLM-as-judge trustworthy
- Use a specific rubric with anchored scale descriptions and ask for a reason first.
- Use low temperature; ideally a strong model as judge.
- Check agreement with human labels on 50–100 examples before trusting it.
- Watch for biases: preference for longer answers, for its own style, for the first option in pairwise comparisons.

## Example
**Analogy:** Grading an essay with a marking scheme: a spell-checker catches format issues automatically (code checks), a teaching assistant grades against the rubric (LLM judge), and the professor spot-checks a sample to keep the TA honest (human review).

**Python – deterministic checks + LLM-as-judge:**
```python
import json, re
import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

class Scores(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    reason: str

def code_checks(answer: str) -> dict:
    return {"has_citation": bool(re.search(r"\[[\w.\-]+\]", answer)),
            "no_email_leak": not re.search(r"\b\S+@\S+\.\w+\b", answer)}

def judge(question: str, context: str, answer: str) -> Scores:
    resp = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=300, temperature=0,
        system=("You grade answers. Faithfulness: 5 = every claim supported by context, "
                "1 = mostly unsupported. Relevance: does it answer the question. "
                "Completeness: covers all parts. Reply with JSON only: "
                '{"faithfulness":n,"relevance":n,"completeness":n,"reason":"..."}'),
        messages=[{"role": "user", "content":
                   f"<context>{context}</context>\n<question>{question}</question>\n"
                   f"<answer>{answer}</answer>"}],
    )
    return Scores(**json.loads(resp.content[0].text))

ctx = "[refunds.md] Refunds are accepted within 30 days with a receipt."
ans = "You can get a refund within 60 days [refunds.md]."
print(code_checks(ans))     # {'has_citation': True, 'no_email_leak': True}
print(judge("What is the refund window?", ctx, ans))
# faithfulness=1 relevance=5 completeness=4 reason='Says 60 days; context says 30.'  (illustrative)
```
Note the format check passed but faithfulness failed — you need both kinds of checks.

## Common Mistakes / Misconceptions
- **Eyeballing a few examples** instead of using a fixed test set with tracked scores.
- **Using only similarity metrics** (BLEU/ROUGE, embedding similarity) — they miss factual errors.
- **Trusting an uncalibrated LLM judge** — verify against human labels.
- **Scoring only the final answer in RAG** — you can't tell if retrieval or generation failed.

## Key Takeaways
- Rubric: faithfulness, relevance, completeness/correctness, format, safety (+ tone, cost).
- Deterministic code checks first; LLM-as-judge for subjective dimensions; humans to calibrate.
- Evaluate retrieval and generation separately for RAG.
- Run the same eval set on every prompt/model change as a regression gate.
- Combine offline scores with online user signals.

## Related Questions
- [What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)
- [What is hallucination detection?](../Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
- [Why do LLMs sometimes hallucinate?](../Week-02-Introduction-to-LLMs-and-Generative-AI/12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [What is the difference between training and testing data?](../Week-01-Introduction-to-Artificial-Intelligence/11-What-is-the-difference-between-training-and-testing-data.md)
- [How do you deploy an LLM application in production?](12-How-do-you-deploy-an-LLM-application-in-production.md)
