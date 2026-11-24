# What is LLM evaluation?

> Week 11 – Production LLM Applications · Question 13

## Short Answer
LLM evaluation is the systematic process of measuring how well a model, or a whole LLM application, performs on the tasks you care about. You run it against a fixed test set and score the outputs with automatic metrics, LLM-as-a-judge and human review. It turns "it seems better" into numbers you can compare across prompts, models and releases. In production it continues as monitoring of real traffic and user feedback.

> **Distinction:** this question covers evaluation as a *discipline*: datasets, methods, offline vs online, regression testing. Week 12 Q11 ([How do you evaluate response quality?](../Week-12-Final-Project-and-Revision/11-How-do-you-evaluate-response-quality.md)) zooms in on the *rubric for judging an individual answer* (faithfulness, relevance, completeness, format, safety).

## Detailed Explanation
**What you evaluate**
| Level | Example questions |
|---|---|
| Model | Which model is best/cheapest for my task? |
| Prompt | Did prompt v7 beat v6? |
| Component | Does the retriever find the right chunks (recall@k)? |
| End-to-end system | Is the final answer correct, grounded, safe, fast and affordable? |

**Methods**
- **Reference-based:** exact match, F1, BLEU/ROUGE, or simple assertions (JSON valid? contains required field?).
- **LLM-as-a-judge:** a model scores outputs against a rubric. Cheap and scalable, but validate it against human labels.
- **Human evaluation:** gold standard for nuance; expensive, so used on samples.
- **Operational metrics:** latency, cost per request, error rate, refusal rate.

**Offline vs online**
```
OFFLINE (before release)                    ONLINE (after release)
golden test set ─► run pipeline ─► scores    real traffic ─► sampled judging,
       │ compare to baseline                  thumbs up/down, A/B tests, drift alerts
       └─ block release if regression ◄───────── new failure cases added to test set
```

## Example
**Analogy:** A driving test. The examiner has a fixed route (test set), a checklist (rubric) and a pass mark. Every new driver, or here every new prompt version, takes the same test.

```python
import anthropic

client = anthropic.Anthropic()
tests = [
    {"q": "Capital of France?", "must_contain": "Paris"},
    {"q": "2 + 2 = ?", "must_contain": "4"},
    {"q": "Who wrote Hamlet?", "must_contain": "Shakespeare"},
]

def run(system_prompt: str) -> float:
    passed = 0
    for t in tests:
        r = client.messages.create(model="claude-haiku-5-5", max_tokens=50,
                                   temperature=0, system=system_prompt,
                                   messages=[{"role": "user", "content": t["q"]}])
        passed += t["must_contain"].lower() in r.content[0].text.lower()
    return passed / len(tests)

print("v1:", run("Answer in one word."))
print("v2:", run("Answer in a short sentence."))
# -> v1: 1.0
# -> v2: 1.0   (illustrative; real sets need 50+ diverse cases)
```

## Common Mistakes / Misconceptions
- "Vibe checking" a few prompts by hand and calling it evaluated.
- Test sets that are too small, too easy, or leaked into examples in the prompt.
- Trusting LLM-as-a-judge without checking agreement with humans; judges can favour long answers.
- Evaluating only quality and ignoring latency, cost and safety.

## Key Takeaways
- Evaluation = fixed dataset + scoring method + baseline comparison.
- Mix automatic checks, LLM judges and human review.
- Run offline before release (regression gate) and online after (monitoring).
- Feed production failures back into the test set.

## Related Questions
- [What is hallucination detection?](14-What-is-hallucination-detection.md)
- [How do you evaluate response quality?](../Week-12-Final-Project-and-Revision/11-How-do-you-evaluate-response-quality.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
