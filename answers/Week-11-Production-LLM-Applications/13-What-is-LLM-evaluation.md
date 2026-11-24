# What is LLM evaluation?

> Week 11 – Production LLM Applications · Question 13

## In One Line
LLM evaluation is **testing your model or app on a fixed set of questions** and scoring the answers, so you know with numbers whether a change made it better or worse.

## Simple Explanation
Without evaluation, people try a few prompts by hand and say "looks good". That's a guess. Evaluation turns the guess into a score.

It has three parts:
1. **A test set:** a fixed list of questions, ideally with expected answers. Start with 20–50 and grow it.
2. **A way to score:**
   - **Automatic checks:** does the answer contain "Paris"? Is the JSON valid?
   - **LLM-as-a-judge:** another model grades the answer against a checklist. Fast, but check it against human grades.
   - **Human review:** the most reliable, but slow, so use it on a sample.
3. **A comparison:** run the test set before and after every prompt or model change. If the score drops, don't release.

**Two times to evaluate:**
- **Offline (before release):** run the test set like an exam.
- **Online (after release):** watch real traffic, thumbs up/down and errors. Add new failures to your test set.

```
Change prompt → Run test set → Score → Better? release : fix it
                                         ↑ add real failures from users
```

Also track **speed, cost and safety**, not just correctness.

This question is about evaluation as a whole process. Week 12 Q11 is the checklist for judging **one single answer**.

## Real-Life Example
A driving test. Every driver drives the same route (test set), the examiner uses the same checklist (scoring), and there's a pass mark. Every new prompt version takes the same test.

## Code Example
A mini test set with a simple "must contain" check. The fake model gets one answer wrong on purpose.

```python
tests = [                                              # our fixed test set
    {"q": "Capital of France?", "must_contain": "Paris"},
    {"q": "2 + 2 = ?", "must_contain": "4"},
    {"q": "Who wrote Hamlet?", "must_contain": "Shakespeare"},
]

def fake_model(question):                              # stand-in for the real app
    answers = {"Capital of France?": "It is Paris.",
               "2 + 2 = ?": "The answer is 4.",
               "Who wrote Hamlet?": "Charles Dickens."}  # wrong on purpose
    return answers[question]

passed = 0
for t in tests:
    answer = fake_model(t["q"])
    ok = t["must_contain"].lower() in answer.lower()     # simple automatic check
    print("PASS" if ok else "FAIL", "|", t["q"], "->", answer)
    passed += ok
print(f"Score: {passed}/{len(tests)}")
# PASS | Capital of France? -> It is Paris.
# PASS | 2 + 2 = ? -> The answer is 4.
# FAIL | Who wrote Hamlet? -> Charles Dickens.
# Score: 2/3
```

**What happened:** the test caught the wrong answer automatically. In a real app, `fake_model` would call your actual pipeline.

## Watch Out
- A test set that is too small or too easy gives false confidence.
- LLM judges can be biased, for example liking longer answers. Compare them with human grades first.

## Remember
- Evaluation = **test set + scoring + comparison**.
- Mix automatic checks, LLM judges and humans.
- Test **before** release and keep watching **after**.

## Related Questions
- [What is hallucination detection?](14-What-is-hallucination-detection.md)
- [How do you evaluate response quality?](../Week-12-Final-Project-and-Revision/11-How-do-you-evaluate-response-quality.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
