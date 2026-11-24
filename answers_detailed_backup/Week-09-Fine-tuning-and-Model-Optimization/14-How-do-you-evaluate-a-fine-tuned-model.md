# How do you evaluate a fine-tuned model?

> Week 09 – Fine-tuning and Model Optimization · Question 14

## Short Answer
Compare the fine-tuned model against a **baseline** (the base model with your best prompt, or your current production system) on a **held-out test set** that reflects real usage. Use task metrics (exact match, F1, JSON validity), LLM-as-judge or human ratings for open-ended quality, and **regression checks** for general skills and safety. Training and validation loss only tell you that learning happened; they don't tell you the model is better for users.

## Detailed Explanation
**Layers of evaluation**

| Layer | What to measure | Examples |
|---|---|---|
| Training signals | Train vs validation loss | Is the validation loss rising? That signals overfitting |
| Task metrics | Automatic, objective | Accuracy, F1, exact match, BLEU/ROUGE (weak), schema-valid %, pass@k for code |
| Open-ended quality | Subjective | LLM-as-judge with a rubric, pairwise win-rate vs baseline, human review |
| Regression | Did we break anything? | Small general benchmark subset (MMLU, GSM8K, IFEval), safety/refusal tests |
| Operational | Cost and speed | Latency, tokens per answer, memory, cost per 1k requests |
| Online | Real users | A/B test, thumbs-up rate, escalation rate |

**Good practice**
- Build the eval set **before** training, from real data, deduplicated against the training set.
- Evaluate **base + prompt**, **fine-tuned**, and ideally **base + RAG** on the same set.
- For LLM-as-judge: use a strong judge model, a clear rubric, randomised A/B order (to avoid position bias), and spot-check against human labels.
- Use the same decoding settings (temperature 0 or a fixed seed) and the same chat template as production.
- Slice the results by category (easy/hard, language, intent) to see where the model wins and loses.
- Use tooling such as `lm-evaluation-harness` for benchmarks, and your own scripts or eval frameworks (promptfoo, Inspect, OpenAI Evals-style harnesses) for task tests.

```
          ┌─ task metrics ─┐
test set ─┼─ judge/human ──┼─► compare vs baseline ─► ship / iterate
          └─ regressions ──┘
```

## Example
**Analogy:** a new hire's training course ends with a test they haven't seen (held-out), a comparison with an experienced colleague (baseline), and a check that they still remember the basics (regression).

```python
import json

def evaluate(generate, test_path="test.jsonl"):
    rows = [json.loads(l) for l in open(test_path)]
    exact, valid = 0, 0
    for r in rows:
        prompt, gold = r["messages"][:-1], r["messages"][-1]["content"]
        pred = generate(prompt)                  # your model call (temperature 0)
        try:
            valid += json.loads(pred) is not None
        except json.JSONDecodeError:
            pass
        exact += pred.strip() == gold.strip()
    n = len(rows)
    return {"exact_match": exact / n, "json_valid": valid / n}

# print(evaluate(base_generate))       # e.g. {'exact_match': 0.41, 'json_valid': 0.78}
# print(evaluate(finetuned_generate))  # e.g. {'exact_match': 0.83, 'json_valid': 0.99}
```

## Common Mistakes / Misconceptions
- Judging only by training loss going down.
- Test-set leakage: near-duplicates of training examples in the test set inflate the scores.
- No baseline, so "83% accuracy" means nothing without knowing that prompting gets 80%.
- Skipping regression and safety checks, and missing catastrophic forgetting.
- Trusting LLM-as-judge blindly. Judges have position and verbosity biases.

## Key Takeaways
- Always compare against a baseline on a held-out, realistic test set.
- Combine task metrics, judge or human ratings, and regression checks.
- Keep decoding settings and the chat template identical to production.
- Watch cost and latency too; often that's the real reason you fine-tuned.

## Related Questions
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is catastrophic forgetting?](13-What-is-catastrophic-forgetting.md)
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
