# What is LoRA?

> Week 09 – Fine-tuning and Model Optimization · Question 7

## In One Line
LoRA (Low-Rank Adaptation) is a cheap way to fine-tune: you **freeze the big model** and train only a **tiny add-on** next to it.

## Simple Explanation
An LLM is mostly made of big tables of numbers called **weight matrices**. One such table in a 7B model can be 4096 × 4096, which is about 16.8 million numbers.

Full fine-tuning changes all of those numbers. LoRA does something smarter:

1. **Freeze** the original table. It is never changed.
2. Add two **thin** tables next to it, called **A** and **B**. How thin is set by the **rank `r`**, for example 16.
3. Train only A and B. Multiplied together, they make a small "correction" that is added to the original table's output.

Because A and B are thin, they have far fewer numbers. With r = 16 that is only about 131 thousand numbers instead of 16.8 million, under 1%.

**Why this is great:**
- Much less GPU memory needed for training.
- The saved add-on (called an **adapter**) is only megabytes, not gigabytes.
- You can keep many adapters (one per customer or task) for a single base model and swap them.
- After training, the adapter can be **merged** into the model, so it runs just as fast.

Quality is usually close to full fine-tuning for most tasks.

## Real-Life Example
Instead of rewriting a whole textbook, you stick a few sticky notes on the pages that need changes. The book stays the same; the notes adjust it. And you can peel the notes off or swap them for a different set.

## Code Example
Count the numbers we train with and without LoRA.

```python
d = 4096          # size of one weight table: d x d
r = 16            # LoRA rank: how "thin" the add-on tables are

full = d * d                  # numbers in the original table
lora = d * r + r * d          # B is d x r, A is r x d

print(f"Full table : {full:,}")             # Full table : 16,777,216
print(f"LoRA (A+B) : {lora:,}")             # LoRA (A+B) : 131,072
print(f"Share      : {lora / full:.2%}")    # Share      : 0.78%
```

**What happened:** for one table, LoRA trains about 0.8% as many numbers. Across a whole 7B model with typical settings, it is often only around 0.1% of all weights.

In real code you set this up with the `peft` library: `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])`.

## Watch Out
- LoRA still needs the **whole base model in memory**; it only saves the training extras. QLoRA also shrinks the base.
- A bigger `r` means more capacity, but also more memory; 8–64 is typical.

## Remember
- LoRA = freeze the model, train a small add-on.
- It trains well under 1% of the numbers.
- Adapters are small files you can merge or swap.

## Related Questions
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
