# When should you fine-tune a model?

> Week 09 – Fine-tuning and Model Optimization · Question 3

## Short Answer
Fine-tune when prompting and RAG have plateaued and you need **consistent behaviour** that is hard to describe in a prompt: a strict output format, a specific tone, a narrow classification or extraction task, or lower cost and latency by using a smaller model. Don't fine-tune just to add facts or knowledge that changes often; use retrieval for that. Fine-tune only when you have a few hundred or more good examples and an evaluation set to prove the fine-tuned model beats the baseline.

## Detailed Explanation
**Recommended escalation ladder:**
```
1. Prompt engineering (clear instructions, examples)
2. Few-shot + structured output / tool schemas
3. RAG for knowledge
4. Fine-tuning (LoRA/QLoRA) for behaviour
5. Full fine-tuning / preference tuning for the last few %
```

**Good reasons to fine-tune**
- **Format and style:** every answer must follow a template or brand voice, and prompts keep drifting.
- **Narrow, high-volume tasks:** classification, entity extraction, routing. A fine-tuned 1–8B model can match a large model at a fraction of the cost.
- **Latency and cost:** shorter prompts (no long few-shot block) and smaller models.
- **Distillation:** generate outputs with a strong model and train a small one to imitate them.
- **Data residency:** you need an open model running on your own hardware.

**Poor reasons**
- "The model doesn't know our product catalogue." Use RAG.
- "We have 30 examples." That is too few; use few-shot prompting.
- "It hallucinates." Fine-tuning can *increase* confident hallucination if the training answers contain facts the model didn't know.

| Need | Best tool |
|---|---|
| Up-to-date facts and citations | RAG |
| Consistent JSON or tone | Fine-tune (or structured outputs) |
| New task the model can already do with examples | Few-shot prompting |
| Cheaper inference at scale | Fine-tune a smaller model |

## Example
**Analogy:** you don't send an employee on a 3-month course (fine-tuning) to learn today's price list. You hand them the price sheet (RAG). You *do* train them for months on how to write reports in the company format.

```python
# A simple decision helper (rule of thumb, not a law)
def should_finetune(n_examples, needs_fresh_facts, prompt_baseline_ok):
    if needs_fresh_facts:
        return "Use RAG first"
    if prompt_baseline_ok:
        return "Keep prompting"
    if n_examples < 200:
        return "Collect more data / few-shot"
    return "Fine-tune (start with LoRA)"

print(should_finetune(2000, False, False))  # Fine-tune (start with LoRA)
print(should_finetune(5000, True, False))   # Use RAG first
print(should_finetune(50, False, False))    # Collect more data / few-shot
```

## Common Mistakes / Misconceptions
- Fine-tuning before building an eval set, so you can't tell whether it helped.
- Treating fine-tuning and RAG as either/or. They combine well: fine-tune for format and use RAG for facts.
- Underestimating maintenance. Each base-model upgrade means re-training and re-evaluating.
- Ignoring structured-output features, which often solve "format" problems with no training.

## Key Takeaways
- Prompt → RAG → fine-tune is the usual order of escalation.
- Fine-tune for **behaviour, format, cost and latency**, not for knowledge.
- You need enough high-quality examples plus a held-out eval set.
- Start with LoRA/QLoRA; reserve full fine-tuning for when LoRA falls short.

## Related Questions
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is a training dataset?](05-What-is-a-training-dataset.md)
