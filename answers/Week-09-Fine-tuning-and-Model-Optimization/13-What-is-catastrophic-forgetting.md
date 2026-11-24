# What is catastrophic forgetting?

> Week 09 – Fine-tuning and Model Optimization · Question 13

## In One Line
Catastrophic forgetting is when training a model on something new makes it **lose skills it already had**.

## Simple Explanation
A model stores everything it knows in the same shared set of weights. There is no separate "maths drawer" and "French drawer".

So when you fine-tune on a new task, here's what can happen:

1. The training only cares about the new task.
2. It moves the weights toward whatever is best for that task.
3. Some of those same weights were holding old skills, so those skills get worse.

For example, a model fine-tuned only on legal summaries might get worse at maths, coding or casual chat. It might even answer *every* question in the legal-summary style. Fine-tuning can also weaken the model's safety behaviour.

**It gets worse when:**
- The new data is narrow and repetitive.
- The learning rate (the size of each update step) is high.
- You train for many passes (epochs).
- You change all weights (full fine-tuning).

**How to reduce it:**
- Use **LoRA**: the base model stays frozen, and you can remove the adapter.
- Use a small learning rate and only 1–3 epochs.
- **Replay:** mix some general examples (say 5–30%) into your training data, so old skills keep getting practised.
- **Test broadly:** check general and safety questions too, not only your new task.

## Real-Life Example
Someone moves abroad and speaks only the new language for years. When they return home, they struggle to find words in their first language. The new practice slowly pushed out the old one, because nothing kept it fresh.

## Code Example
A one-number "model" learns rule A, then rule B. Watch its error on A.

```python
def train(w, data, lr=0.01, steps=200):
    for _ in range(steps):
        for x, y in data:
            w -= lr * 2 * (w * x - y) * x   # nudge w to reduce the error
    return w

def error(w, data):
    return round(sum((w * x - y) ** 2 for x, y in data) / len(data), 2)

task_a = [(1, 2), (2, 4)]                  # old skill: y = 2x
task_b = [(1, 3), (2, 6)]                  # new skill: y = 3x

w = train(0.0, task_a)
print("Error on A after learning A:", error(w, task_a))                  # 0.0
print("Error on A after only B:   ", error(train(w, task_b), task_a))   # 2.5
print("Error on A after B + replay:", error(train(w, task_b + task_a), task_a))  # 0.56
```

**What happened:** training only on B wiped out A. Mixing some A examples back in ("replay") kept the error on A much lower. A real LLM has billions of weights, so with care it can keep both skills.

## Watch Out
- Your new task's score can go **up** while other skills quietly go **down**. Only broad testing catches this.
- It also affects safety, not just knowledge.

## Remember
- New training can overwrite old skills.
- LoRA, a low learning rate, few epochs and replay data all help.
- Always test general and safety skills after fine-tuning.

## Related Questions
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is LoRA?](07-What-is-LoRA.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
