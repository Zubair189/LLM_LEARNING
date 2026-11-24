# What is full fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 11

## Short Answer
Full fine-tuning updates **every weight** in the pretrained model on your new data, rather than training a small adapter as LoRA does. It has the most capacity to change the model and can give the best quality for big shifts (new language, large domain corpus). However, it needs far more GPU memory, roughly **16 bytes per parameter** with Adam in mixed precision, produces a full-size checkpoint for every run, and carries a higher risk of catastrophic forgetting.

## Detailed Explanation
**Memory per parameter (mixed-precision AdamW)**

| Item | Bytes/param |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam first moment (m) | 4 |
| Adam second moment (v) | 4 |
| **Total** | **~16** |

**7B model:** 7×10⁹ × 16 B ≈ **112 GB** before activations. That means several 80 GB GPUs with sharding (DeepSpeed ZeRO-3 / FSDP). Compare LoRA at ~16–20 GB and QLoRA at ~6–12 GB.

**When full fine-tuning is worth it**
- Large datasets (hundreds of thousands to millions of examples) or continued pretraining on a new domain or language.
- When you need the absolute best quality and LoRA evaluations have plateaued.
- Training smaller models (≤3B), where it is affordable anyway.
- When the vendor or base-model creator builds a new instruct model from a base model.

**Downsides**
- Cost: multi-GPU, longer runs, more engineering.
- Storage: each version is a full copy (14 GB for 7B in bf16), versus ~30 MB for a LoRA adapter.
- Forgetting: changing all the weights can erase general skills. Use a low learning rate (~1e-5 to 2e-5) and mix in general data.

| | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Trainable params (7B) | 7B | ~4–40M | ~4–40M |
| GPU memory (7B) | ~112 GB+ | ~18 GB | ~8 GB |
| Checkpoint | 14 GB | ~10–100 MB | ~10–100 MB |
| Quality ceiling | Highest | Close | Close |

## Example
**Analogy:** renovating the entire house versus adding furniture. Full fine-tuning can change anything, including the walls, but it's expensive, slow and you might knock down something you needed.

```python
# Memory estimator
def full_ft_gb(params_b, bytes_per_param=16):
    return params_b * bytes_per_param

for p in (1, 7, 70):
    print(p, "B ->", full_ft_gb(p), "GB")
# 1 B -> 16 GB
# 7 B -> 112 GB
# 70 B -> 1120 GB

# Full fine-tuning with TRL: simply omit peft_config
# from trl import SFTTrainer, SFTConfig
# SFTTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", train_dataset=ds,
#            args=SFTConfig(output_dir="full", learning_rate=1e-5, bf16=True,
#                           gradient_checkpointing=True)).train()
```

## Common Mistakes / Misconceptions
- Using LoRA-style learning rates (2e-4) for full fine-tuning, which destroys the model. Use ~10× smaller.
- Assuming full fine-tuning is always better. On small datasets LoRA often matches it and overfits less.
- Budgeting only for the weights. Gradients, optimizer state and activations dominate.
- Forgetting tricks like gradient checkpointing, 8-bit Adam and ZeRO sharding, which make it feasible.

## Key Takeaways
- Full fine-tuning updates all weights, giving maximum flexibility at maximum cost.
- Budget roughly 16 bytes/param plus activations (7B ≈ 112 GB+).
- Higher forgetting risk; use low learning rates and data mixing.
- Default to LoRA/QLoRA and move to full fine-tuning only when evals justify it.

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is catastrophic forgetting?](13-What-is-catastrophic-forgetting.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
