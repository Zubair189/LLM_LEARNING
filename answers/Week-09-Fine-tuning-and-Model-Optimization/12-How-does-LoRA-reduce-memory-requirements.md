# How does LoRA reduce memory requirements?

> Week 09 – Fine-tuning and Model Optimization · Question 12

## In One Line
LoRA saves memory because the expensive training extras (gradients and optimizer data) are kept only for a **tiny add-on**, not for all the model's weights.

## Simple Explanation
When you train a weight, the GPU stores much more than the weight itself. For each trainable weight it keeps its **gradient** (which way to move) and the **optimizer's** bookkeeping numbers. Together with the weight, that's about **16 bytes per trainable weight**.

Here's the key idea:

1. **Full fine-tuning:** all 7 billion weights are trainable, so 7B × 16 bytes ≈ **112 GB**.
2. **LoRA:** the 7 billion base weights are **frozen**. They need only their own 2 bytes each (≈ 14 GB) and no extras.
3. The small LoRA add-on, for example about 8.4 million weights, is the only trainable part: 8.4M × 16 bytes ≈ **0.13 GB**.

So LoRA's total is about 14 GB + 0.13 GB, plus some **activations** (temporary values during training). That's roughly **16–20 GB** instead of 112 GB.

**QLoRA goes further:** it also shrinks the frozen base to 4 bits (≈ 3.5 GB), bringing the total to roughly **6–12 GB**.

**What LoRA does not shrink:**
- The base model itself (QLoRA fixes this).
- Activations (use smaller batches, shorter texts, or "gradient checkpointing").
- The model's size when you later run it.

**Bonus:** the saved adapter file is only tens of megabytes.

## Real-Life Example
Imagine a 1,000-page book where you must keep a notebook of corrections for every page you're allowed to edit. If you may edit every page, you need 1,000 notebooks. If you only edit a short sticky-note supplement, you need just a few notebooks. The book is still on your desk, but the paperwork shrinks massively.

## Code Example
Compare memory for full fine-tuning and LoRA on a 7B model.

```python
base = 7_000_000_000        # weights in the base model
lora = 8_400_000            # trainable LoRA weights (typical small setup)
GB = 1e9

full_ft = base * 16 / GB                     # every weight trained: 16 bytes each
lora_ft = base * 2 / GB + lora * 16 / GB     # base frozen (2 bytes) + small trained add-on
qlora_ft = base * 0.5 / GB + lora * 16 / GB  # base frozen in 4-bit (0.5 bytes)

print(f"Full fine-tuning: {full_ft:.0f} GB")   # Full fine-tuning: 112 GB
print(f"LoRA:             {lora_ft:.1f} GB")   # LoRA:             14.1 GB
print(f"QLoRA:            {qlora_ft:.1f} GB")  # QLoRA:            3.6 GB
```

**What happened:** these numbers are before activations. Add a few GB for those and you get the usual estimates: ~112 GB+, ~16–20 GB and ~6–12 GB.

## Watch Out
- LoRA mainly saves **training extras**, not the base model's own memory.
- Long texts and big batches can still run you out of memory, because of activations.

## Remember
- Training costs about 16 bytes per **trainable** weight.
- LoRA makes only ~0.1% of weights trainable.
- 7B: ~112 GB (full) → ~16–20 GB (LoRA) → ~6–12 GB (QLoRA).

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
