# What is model pretraining?

> Week 09 – Fine-tuning and Model Optimization · Question 1

## Short Answer
Pretraining is the first, most expensive stage of building an LLM: a randomly initialized transformer is trained on trillions of tokens of unlabeled text with a self-supervised objective, usually **next-token prediction**. The result is a *base model* that has absorbed grammar, facts, reasoning patterns and code, but is not yet good at following instructions. Everything later (SFT, preference tuning, LoRA) is cheap adaptation on top of this foundation.

## Detailed Explanation
- **Objective:** given tokens `t1..tn`, predict `t(n+1)`. The loss is cross-entropy over the vocabulary. No human labels are needed; the text supplies its own labels ("self-supervised").
- **Data:** web crawls (filtered Common Crawl), books, code, papers and multilingual text, deduplicated and quality-filtered. Modern open models are trained on roughly 10–40T tokens.
- **Compute:** a rough rule is `FLOPs ≈ 6 × params × tokens`. For example, 8B params × 15T tokens × 6 ≈ 7×10²³ FLOPs, which takes thousands of GPUs for weeks.
- **Output:** a base model that *completes* text. Ask it "What is the capital of France?" and it may continue with more quiz questions instead of answering.

| Stage | Data | Cost | Purpose |
|---|---|---|---|
| Pretraining | Trillions of raw tokens | $1M–$100M+ | General knowledge and language |
| SFT | 1k–100k curated examples | $10–$10k | Follow instructions, set format |
| Preference tuning | Ranked pairs | Similar to SFT | Make it helpful, harmless and on-style |

```
raw text ──► [pretraining] ──► base model ──► [SFT] ──► instruct model ──► [DPO/RLHF] ──► chat model
```

You will almost never pretrain from scratch. Some teams do **continued pretraining**, which means running more next-token training on a large domain corpus (legal, biomedical) before SFT.

## Example
**Analogy:** pretraining is like a student reading an entire library for years. They know a huge amount, but nobody has taught them how to answer an exam question. Fine-tuning is the short exam-prep course.

```python
# Next-token prediction objective, shown with a small pretrained model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

text = "The capital of France is Paris."
ids = tok(text, return_tensors="pt").input_ids
out = model(ids, labels=ids)          # labels = inputs; HF shifts them internally
print(round(out.loss.item(), 2))      # e.g. ~3.0 (average cross-entropy per token)

# Base-model behaviour: it completes text rather than answering
gen = model.generate(tok("Q: What is 2+2?\nA:", return_tensors="pt").input_ids,
                     max_new_tokens=8, do_sample=False)
print(tok.decode(gen[0]))  # continues the pattern; may or may not answer cleanly
```

## Common Mistakes / Misconceptions
- **"Pretraining uses labeled data."** It does not. The next token *is* the label.
- **"Fine-tuning is how the model learns facts."** Most knowledge comes from pretraining. Fine-tuning mainly shapes behaviour and format; for fresh facts, use RAG.
- **"A base model is a chatbot."** Base models need SFT and preference tuning before they follow instructions reliably.
- Confusing **continued pretraining** (raw domain text, next-token loss on everything) with **SFT** (prompt/response pairs, loss usually only on the response).

## Key Takeaways
- Pretraining = self-supervised next-token prediction on massive unlabeled text.
- It is by far the most expensive stage, and it produces general capabilities.
- Its output is a base model; instruction-following comes later.
- Practitioners usually start from an open pretrained checkpoint and adapt it.

## Related Questions
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is a training dataset? (Week 01)](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)
