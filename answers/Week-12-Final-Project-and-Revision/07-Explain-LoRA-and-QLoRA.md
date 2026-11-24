# Explain LoRA and QLoRA.

> Week 12 – Final Project and Revision · Question 7

## In One Line
**LoRA** fine-tunes a model by **freezing it and training a few small add-on pieces**; **QLoRA** does the same but first **shrinks the frozen model to 4-bit** so it fits on a single GPU.

## Simple Explanation
**Full fine-tuning** updates *every* weight (internal number) in the model. A 7-billion-weight model needs huge amounts of GPU memory for that, because training stores extra information for every weight it changes.

**LoRA (Low-Rank Adaptation)** has a smarter idea: the *change* a model needs for a new task is usually small and simple. So instead of changing the big weight tables, we:
- **freeze** the original weights (don't touch them), and
- add two **small** tables, called `A` and `B`, next to some layers. Only these get trained.

Their product `B × A` acts as a small "correction" added to the original layer.

**How small?** For one 4096 × 4096 weight table (about 16.8 million numbers), LoRA with a "rank" of 8 trains only 2 × 4096 × 8 = 65,536 numbers, about 0.4%. Across the whole model, typically well under 1% of the weights are trained.

**QLoRA** adds one more trick: **quantization**, which stores numbers with fewer bits (less precision) to save memory. The frozen base model is stored in **4-bit** instead of 16-bit, so it is about 4 times smaller. The small LoRA tables are still trained in higher precision.

```
Base model → (QLoRA: shrink to 4-bit) → Freeze it → Add small LoRA tables → Train only those → Save small adapter file
```

**Step by step:**
1. **Prepare data:** example pairs of input and ideal output.
2. **Load the base model:** in 16-bit for LoRA, or in 4-bit for QLoRA.
3. **Freeze** all its weights.
4. **Attach LoRA adapters** (the small `A` and `B` tables) to chosen layers, usually the attention layers.
5. **Train** only the adapters.
6. **Evaluate** against the original model on a test set.
7. **Save the adapter.** It is only a few megabytes. At use time, load it on top of the base model, or merge it in.

**Rough memory picture (approximate):** a 7B model's weights are about 14 GB in 16-bit, but about 3.5–4 GB in 4-bit. That's why QLoRA can fine-tune it on a single 16–24 GB GPU.

**Main settings:**
- **r (rank):** the size of the small tables (8–64 is common). Bigger means more capacity.
- **alpha:** how strongly the correction is applied (often 2 × r).
- **target modules:** which layers get adapters.

**Pros:** cheap, fast, small files; you can keep several adapters for different tasks on one base model. The base is frozen, so it forgets less of what it already knew.
**Cons:** sometimes slightly below full fine-tuning; QLoRA trains a bit slower; you still need good data and a model with open weights.

## Real-Life Example
Instead of rewriting a 1,000-page textbook (full fine-tuning), you add sticky notes with corrections (LoRA). QLoRA is the same, but the textbook is a compact pocket edition (4-bit) so it fits in your bag, while your sticky notes stay full quality.

## Watch Out
- QLoRA does **not** train a 4-bit model. The base is stored in 4-bit and frozen; the adapters train in higher precision.
- LoRA is best for teaching **behaviour and style**, not for adding facts that change. Use RAG for those.

## Remember
- LoRA: **freeze the model, train small add-on tables**, usually under 1% of the weights.
- QLoRA: LoRA on a **4-bit** base, so big models fit on one GPU.
- Adapters are **small, swappable** files; good data and evaluation still matter most.

## Related Questions
- [What is LoRA?](../Week-09-Fine-tuning-and-Model-Optimization/07-What-is-LoRA.md)
- [What is QLoRA?](../Week-09-Fine-tuning-and-Model-Optimization/08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](../Week-09-Fine-tuning-and-Model-Optimization/12-How-does-LoRA-reduce-memory-requirements.md)
- [What is 4-bit quantization?](../Week-09-Fine-tuning-and-Model-Optimization/10-What-is-4-bit-quantization.md)
- [What is full fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/11-What-is-full-fine-tuning.md)
- [Compare RAG and fine-tuning.](06-Compare-RAG-and-fine-tuning.md)
