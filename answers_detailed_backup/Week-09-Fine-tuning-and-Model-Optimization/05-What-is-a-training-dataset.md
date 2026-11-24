# What is a training dataset?

> Week 09 – Fine-tuning and Model Optimization · Question 5

> **Note:** The general definition of a training dataset is covered in [Week 01 – What is a training dataset?](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md). This answer focuses on **fine-tuning datasets for LLMs**: format, quality, size and splits.

## Short Answer
In LLM fine-tuning, the training dataset is a curated collection of example conversations (usually chat-format JSONL) showing exactly the inputs the model will see and the outputs you want. Its **quality, consistency and coverage** matter far more than its size: a few hundred to a few thousand clean, diverse examples are often enough for LoRA SFT. It is always paired with a held-out validation set and a separate test/eval set so you can detect overfitting and measure real improvement.

## Detailed Explanation
**Format.** Most tools (TRL, Axolotl, hosted fine-tuning APIs) accept one JSON object per line:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```
Other common layouts are `{"prompt":..., "completion":...}` for SFT and `{"prompt", "chosen", "rejected"}` for DPO.

**Quality checklist**
- Targets are **correct** and in exactly the format you want at inference.
- **Consistent** style: one way of answering, not five.
- **Diverse** inputs that cover real user phrasing, edge cases and refusals.
- **Deduplicated**, PII removed, licences checked.
- Matches production: same system prompt and same chat template.

**How much data?**

| Goal | Rough size |
|---|---|
| Format/style adaptation (LoRA) | 200–1,000 |
| Narrow task (classification/extraction) | 1k–10k |
| Broad new skill or domain | 10k–100k+ |

**Splits**
```
all data ──► dedupe ──► shuffle ──► train 80–90% │ validation 5–10% │ test 5–10%
```
- **Train:** gradients are computed on it.
- **Validation:** watched during training (eval loss, early stopping).
- **Test/eval:** touched only at the end. Ideally it is written independently, e.g. from real production logs.
- Split by **source/user/document**, not by row, so near-duplicates don't leak across splits.

**Synthetic data.** It's common to generate examples with a strong model and then filter them by hand or with an LLM judge. Always review a sample manually.

## Example
**Analogy:** a fine-tuning dataset is a stack of model exam answers for a trainee. A small stack of perfect answers teaches more than a huge stack of sloppy ones, and you keep some exam papers locked away for the final test.

```python
import json, random

rows = [json.loads(l) for l in open("all.jsonl")]
# dedupe on the user message
seen, clean = set(), []
for r in rows:
    key = r["messages"][-2]["content"].strip().lower()
    if key not in seen and r["messages"][-1]["role"] == "assistant":
        seen.add(key); clean.append(r)

random.seed(42); random.shuffle(clean)
n = len(clean); a, b = int(.9 * n), int(.95 * n)
splits = {"train": clean[:a], "val": clean[a:b], "test": clean[b:]}
for name, data in splits.items():
    with open(f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in data)
print({k: len(v) for k, v in splits.items()})  # e.g. {'train': 900, 'val': 50, 'test': 50}
```

## Common Mistakes / Misconceptions
- Evaluating on data that leaked from training (duplicates or paraphrases), which inflates the scores.
- Using a different system prompt or template in training than in production.
- Scraping thousands of low-quality examples instead of curating hundreds of good ones.
- Teaching facts the model didn't know in answers, which can encourage hallucination.

## Key Takeaways
- Fine-tuning datasets are usually chat-format JSONL that mirrors production inputs.
- Quality and consistency beat volume; hundreds to thousands of examples is typical.
- Keep train / validation / test splits, and dedupe before splitting.
- Review synthetic data and strip PII before training.

## Related Questions
- [What is a training dataset? (Week 01 – general definition)](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
