# What is LoRA?

> Week 09 – Fine-tuning and Model Optimization · Question 7

## Short Answer
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. It **freezes all the original weights** and adds a pair of small trainable matrices `B·A` next to selected layers (typically the attention projections). Because the weight update is constrained to low rank `r` (e.g. 16), you train well under 1% of the parameters. That gives tiny checkpoints (MBs instead of GBs), much less GPU memory, and quality close to full fine-tuning for most SFT tasks.

## Detailed Explanation
For a frozen weight `W` (d × k), LoRA learns an update `ΔW = B·A`, where `B` is d × r and `A` is r × k, with r ≪ d, k:

```
        x ──► [ W (frozen, d×k) ] ───────────────┐
        │                                        (+) ──► h = Wx + (α/r)·B(Ax)
        └──► [ A (r×k) ] ──► [ B (d×r) ] ────────┘
                trainable, B initialised to 0 so training starts at ΔW = 0
```

**Parameter math for one 4096 × 4096 projection, r = 16**
- Full matrix: 4096 × 4096 = **16,777,216** params.
- LoRA: A (16 × 4096) + B (4096 × 16) = 2 · 4096 · 16 = **131,072** params, which is **0.78%**.
- Llama-2-7B with `q_proj` and `v_proj` across 32 layers: 32 × 2 × 131,072 = **8.4M** trainable params out of 6.7B (≈0.12%).

**Key hyperparameters**

| Param | Meaning | Typical |
|---|---|---|
| `r` | Rank, i.e. capacity of the update | 8–64 |
| `lora_alpha` | Scaling; the effective scale is α/r | often 2·r |
| `target_modules` | Which layers get adapters | `q_proj`,`v_proj` up to all linear layers |
| `lora_dropout` | Regularisation | 0.0–0.1 |

**Benefits**
- Low memory: no gradients or optimizer state for the frozen weights.
- Small adapter files (~10–100 MB), so you can keep **many adapters for one base model** and hot-swap them per customer or task.
- Can be **merged** into `W` after training, so inference has zero extra latency.

**Variants:** QLoRA (4-bit base), DoRA (separate magnitude/direction), and rsLoRA (rank-stabilised scaling).

## Example
**Analogy:** instead of rewriting a 1,000-page textbook (full fine-tuning), you add a thin booklet of sticky-note corrections. The book is unchanged; you read it together with the notes.

```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                             torch_dtype=torch.bfloat16, device_map="auto")
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243

# after training:
# merged = model.merge_and_unload()   # bake ΔW into W for deployment
```

## Common Mistakes / Misconceptions
- **"LoRA makes inference faster."** It doesn't. It makes *training* cheaper; merged models run at the same speed as the base model.
- **"LoRA reduces the memory of the base weights."** The frozen weights are still loaded in full precision. That is what QLoRA fixes.
- Using too small an `r` or too few target modules for a hard task. Adding all linear layers (`"all-linear"`) often helps.
- Forgetting that changing `r` without adjusting `alpha` changes the effective learning rate.

## Key Takeaways
- LoRA = frozen base + trainable low-rank `B·A` update.
- r=16 on a 4096×4096 matrix trains 131k params instead of 16.7M.
- Adapters are small, swappable and mergeable.
- It is the default starting point for fine-tuning open models.

## Related Questions
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
