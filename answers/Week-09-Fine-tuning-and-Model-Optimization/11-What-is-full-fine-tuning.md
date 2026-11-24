# What is full fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 11

## In One Line
Full fine-tuning means updating **every single weight** in the model during training, instead of only a small add-on like LoRA.

## Simple Explanation
In full fine-tuning, the whole model is allowed to change. This gives it the most freedom to learn, which helps for big changes such as a new language or a large new subject area.

The problem is **memory**. For every weight being trained, the GPU must keep several extra numbers:

1. The weight itself (2 bytes).
2. Its **gradient**, meaning the direction it should move (2 bytes).
3. A high-precision copy of the weight (4 bytes).
4. Two running averages used by the **Adam optimizer**, the method that decides how big each update step is (4 + 4 bytes).

That adds up to about **16 bytes per weight**. For a 7B model: 7 billion × 16 bytes ≈ **112 GB**, before counting anything else. That needs several large GPUs working together. LoRA needs roughly 16–20 GB and QLoRA roughly 6–12 GB for the same model.

**When it is worth it:**
- You have a very large dataset (hundreds of thousands of examples or more).
- LoRA has been tried and isn't good enough.
- The model is small (around 3B or less), so it's affordable anyway.

**Downsides:**
- Expensive and slower.
- Each saved version is a full copy (about 14 GB for 7B), versus about 30 MB for a LoRA adapter.
- Higher risk of **catastrophic forgetting** (losing old skills). Use a small learning rate and mix in some general data.

## Real-Life Example
LoRA is like adding a new extension to a house. Full fine-tuning is renovating every room. You can change anything, but it costs much more and you might accidentally break things that worked fine.

## Code Example
Back-of-envelope memory for full fine-tuning.

```python
params = 7_000_000_000          # 7B model

bytes_per_param = (2            # weight (16-bit)
                   + 2          # gradient
                   + 4          # high-precision copy of the weight
                   + 4 + 4)     # Adam optimizer's two running averages

total_gb = params * bytes_per_param / 1e9
print(bytes_per_param, "bytes per weight")   # 16 bytes per weight
print(f"about {total_gb:.0f} GB")            # about 112 GB
```

**What happened:** 16 bytes for each of 7 billion weights gives about 112 GB, which is far more than one normal GPU holds.

## Watch Out
- "Full" does not mean "always better". For most tasks LoRA gets very close.
- The 112 GB doesn't include **activations** (temporary values during training), which add more.

## Remember
- Full fine-tuning changes all weights.
- About 16 bytes per weight → about 112 GB for 7B.
- Use it only when LoRA/QLoRA aren't enough.

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is catastrophic forgetting?](13-What-is-catastrophic-forgetting.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
