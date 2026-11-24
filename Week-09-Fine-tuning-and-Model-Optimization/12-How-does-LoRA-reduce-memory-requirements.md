# How does LoRA reduce memory requirements?

> Week 09 – Fine-tuning and Model Optimization · Question 12

## Short Answer
In training, most GPU memory goes not to the weights themselves but to the **gradients and optimizer state** kept for every trainable parameter (about 14 extra bytes per parameter with AdamW). LoRA freezes the base model, so those costs apply only to the tiny adapter, typically 0.1–1% of the parameters. A 7B model therefore drops from ~112 GB (full fine-tuning) to ~16–20 GB, and combining LoRA with a 4-bit base (QLoRA) brings it to ~6–12 GB. As a bonus, checkpoints shrink from GBs to MBs.

## Detailed Explanation
**Where memory goes (7B, mixed-precision AdamW)**

| Component | Full FT | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|---|
| Base weights | 14 GB | 14 GB | ~3.5–4 GB |
| Gradients | 14 GB | ~0.02 GB | ~0.02 GB |
| fp32 master + Adam m, v | 84 GB | ~0.1 GB | ~0.1 GB |
| Activations* | 4–20 GB | 4–20 GB | 4–20 GB |
| **Total (approx.)** | **~112 GB + act.** | **~16–20 GB** | **~6–12 GB** |

\*Activations depend on batch size and sequence length; gradient checkpointing cuts them a lot.

**The low-rank trick, in numbers**
- One 4096 × 4096 projection = **16,777,216** params.
- LoRA r=16: `A` (16×4096) + `B` (4096×16) = 2·4096·16 = **131,072** params, which is **128× fewer**.
- `q_proj` + `v_proj` in all 32 layers of a 7B model = 32 · 2 · 131,072 ≈ **8.4M** trainable params.
- Optimizer and gradient memory for 8.4M params ≈ 8.4M × 16 B ≈ **134 MB**, versus ~98 GB for 7B params (gradients + master weights + Adam).

```
Full FT : [weights][grads][master][adam m][adam v]   for 7,000M params
LoRA    : [weights]                                  for 7,000M params (frozen)
          [w][g][m][v]                               for     8.4M params only
```

**What LoRA does *not* reduce**
- The frozen base weights. They still sit in memory at full precision (QLoRA fixes this).
- Activations. Use gradient checkpointing, smaller micro-batches or shorter sequences.
- Inference memory. Once merged, the model is the same size as the base.

**Extra savings:** many adapters can share one base model in memory (multi-LoRA serving in vLLM or LoRAX), and adapter files are ~10–100 MB.

## Example
**Analogy:** in full fine-tuning you keep a detailed revision history for every page of a 1,000-page book. With LoRA you keep revision history only for a 2-page insert; the book itself is read-only.

```python
def train_mem_gb(total_b, trainable_m, base_bytes=2):
    weights = total_b * base_bytes             # GB (frozen or not)
    per_trainable = 2 + 4 + 4 + 4              # grad + fp32 master + Adam m + v
    return weights + trainable_m / 1000 * per_trainable

print(round(train_mem_gb(7, 7000), 1))              # full FT: 112.0 GB
print(round(train_mem_gb(7, 8.4), 2))               # LoRA:    14.12 GB (+ activations)
print(round(train_mem_gb(7, 8.4, base_bytes=0.5), 2))  # QLoRA: 3.62 GB (+ activations)

r, d = 16, 4096
print(d * d, 2 * d * r)   # 16777216 131072
```

## Common Mistakes / Misconceptions
- Thinking LoRA shrinks the model. The base is still full size; only the *trainable* part is small.
- Ignoring activations. A long `max_seq_length` can make them the largest memory item.
- Believing that a higher rank makes memory explode. Even r=64 on all linear layers is usually <2% of the parameters.
- Assuming a LoRA run on a 24 GB card will also fit when serving at long contexts. The KV cache is separate.

## Key Takeaways
- Memory in training is dominated by gradients and optimizer state for trainable parameters.
- LoRA makes the trainable set tiny (r=16: 131k vs 16.7M per 4096² matrix).
- 7B: ~112 GB full FT → ~16–20 GB LoRA → ~6–12 GB QLoRA.
- Add gradient checkpointing for activations, and a 4-bit base (QLoRA) for the weights.

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
