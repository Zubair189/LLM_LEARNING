# How do you evaluate a fine-tuned model?

> Week 09 – Fine-tuning and Model Optimization · Question 14

## In One Line
You test the fine-tuned model and the **original model** on the **same hidden test questions**, and check that it got better at your task without getting worse at everything else.

## Simple Explanation
A falling training **loss** (the error score during training) only shows the model learned *something*. It doesn't prove it's better for real users. You need a proper test.

Steps:

1. **Build a test set before training.** Use real examples the model will never train on.
2. **Pick a baseline**, meaning what you're comparing against: usually the original model with your best prompt.
3. **Run both models** on the test set, with the same settings.
4. **Score them:**
   - **Task metrics:** automatic checks such as "is the answer exactly right?" or "is the output valid JSON?".
   - **Open-ended quality:** for free-text answers, ask humans or a strong AI ("LLM-as-judge") to rate them with a clear checklist.
5. **Check for damage (regressions):** ask some general and safety questions too, to catch **catastrophic forgetting**.
6. **Check cost and speed:** response time and cost per request.

Only ship the fine-tuned model if it clearly beats the baseline.

**Tips:**
- Look at results by category (easy vs hard, by topic) to see exactly where it wins and loses.
- If you use an AI judge, randomly swap the order of answers A and B, because judges tend to favour whichever comes first.

## Real-Life Example
A football coach tries a new training plan. They don't just ask "did the players sweat?" (training loss). They compare match results before and after, against the same kind of opponents, and also check nobody got injured (regressions).

## Code Example
Compare a baseline and a fine-tuned model on 3 test emails.

```python
import json

expected   = ['{"intent":"refund"}', '{"intent":"late"}', '{"intent":"cancel"}']
baseline   = ['Sure! Refund.',       '{"intent":"late"}', '{"intent":"refund"}']
fine_tuned = ['{"intent":"refund"}', '{"intent":"late"}', '{"intent":"cancel"}']

def is_json(text):
    try:
        json.loads(text); return True        # parses -> valid JSON
    except ValueError:
        return False

def score(outputs):
    valid = sum(is_json(o) for o in outputs)                     # how many are valid JSON
    exact = sum(o == e for o, e in zip(outputs, expected))       # how many are exactly right
    return f"valid JSON {valid}/3, exact match {exact}/3"

print("Baseline:  ", score(baseline))    # Baseline:   valid JSON 2/3, exact match 1/3
print("Fine-tuned:", score(fine_tuned))  # Fine-tuned: valid JSON 3/3, exact match 3/3
```

**What happened:** both models answered the same test questions, and simple automatic checks showed the fine-tuned one is better. Real test sets should have hundreds of examples.

## Watch Out
- Don't test on examples the model trained on; that's like giving students the exam answers in advance.
- A low training loss can hide **overfitting** (memorising instead of learning).

## Remember
- Compare against a baseline on the same hidden test set.
- Measure task quality **and** general/safety skills.
- Ship only if it clearly wins.

## Related Questions
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is catastrophic forgetting?](13-What-is-catastrophic-forgetting.md)
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
