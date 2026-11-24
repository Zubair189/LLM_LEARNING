# What is catastrophic forgetting?

> Week 09 – Fine-tuning and Model Optimization · Question 13

## Short Answer
Catastrophic forgetting is when fine-tuning a model on a new task makes it **lose abilities it previously had**. A model fine-tuned on legal summaries might, for example, get worse at maths, coding, other languages or general chat, or lose its safety behaviour. It happens because gradient updates for the new data overwrite weights that encoded old skills. You reduce it with parameter-efficient methods (LoRA), low learning rates, few epochs, and by mixing general "replay" data into the fine-tuning set.

## Detailed Explanation
**Why it happens**
- Neural networks store knowledge in shared, distributed weights.
- Optimising only for task B pushes the weights toward B's optimum and away from the region that was good for A.
- The narrower and more repetitive the new data, and the larger the updates (high LR, many epochs, full fine-tuning), the worse the forgetting.

```
loss on old skills ▲
                   │            ╱ full FT, high LR
                   │         ╱
                   │      ╱    ── LoRA, low LR + replay
                   │  ╱  ─────────────────
                   └──────────────────────► training steps on new task
```

**Symptoms to watch**
- General benchmarks (MMLU-style, GSM8K, coding) drop after fine-tuning.
- The model answers *everything* in the new format, even unrelated questions.
- Safety or refusal behaviour weakens. Fine-tuning on even benign data can erode safety alignment.
- Instruction following degrades for prompts outside the training distribution.

**Mitigations**

| Technique | How it helps |
|---|---|
| LoRA/QLoRA | Base weights are frozen; the adapter can be removed. LoRA "learns less and forgets less." |
| Lower LR, 1–3 epochs, early stopping | Smaller and fewer weight changes |
| Replay / data mixing | Add 5–30% general instruction data to the training set |
| Regularisation (e.g. EWC, KL penalty) | Penalises moving important weights or drifting from the base model |
| Per-task adapters | Keep separate LoRA adapters and route by task |
| Evaluate broadly | Track general and safety evals, not just the target task |

## Example
**Analogy:** a pianist who practises only one jazz piece for three months straight may find their classical repertoire has gone rusty. Mixing in old pieces each week (replay) keeps both alive.

```python
import json, random

task = [json.loads(l) for l in open("legal_sft.jsonl")]        # e.g. 4,000 rows
general = [json.loads(l) for l in open("general_chat.jsonl")]  # broad instruction data

replay_ratio = 0.2
n_general = int(len(task) * replay_ratio / (1 - replay_ratio))
mix = task + random.sample(general, n_general)
random.shuffle(mix)
print(len(task), n_general, len(mix))   # 4000 1000 5000

# Then: LoRA (r=16), lr=1e-4, 2 epochs, and evaluate BOTH the legal eval
# and a general eval (e.g. a small MMLU/GSM8K subset) before vs after.
```

## Common Mistakes / Misconceptions
- Only evaluating on the target task, then discovering in production that the model can't do basic things anymore.
- Assuming LoRA makes forgetting impossible. It reduces it, but a merged LoRA trained hard can still cause forgetting.
- Training for many epochs on a small dataset. This both overfits and causes forgetting.
- Forgetting that safety behaviour can be "forgotten" too. Re-run safety and red-team checks after every fine-tune.

## Key Takeaways
- Fine-tuning on narrow data can overwrite general skills; this is catastrophic forgetting.
- Bigger updates (full FT, high LR, many epochs) mean more forgetting.
- Use LoRA, gentle hyperparameters and replay data to limit it.
- Always evaluate general and safety skills alongside the target task.

## Related Questions
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is LoRA?](07-What-is-LoRA.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
