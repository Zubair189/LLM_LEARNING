# Explain LoRA and QLoRA.

> Week 12 – Final Project and Revision · Question 7

## Short Answer
**LoRA (Low-Rank Adaptation)** fine-tunes a model by **freezing the original weights** and training two small low-rank matrices `A` and `B` per targeted layer, so the weight update is `ΔW = B·A`; this trains well under 1% of the parameters and needs far less GPU memory. **QLoRA** goes further by loading the frozen base model in **4-bit quantized** form (NF4) and training LoRA adapters on top, making it possible to fine-tune large models on a single consumer or cloud GPU with quality close to full fine-tuning.

## Detailed Explanation

### End-to-end picture

```
 Full fine-tuning                LoRA                          QLoRA
 ────────────────                ────                          ─────
 W (d×d) trainable  ┌──────────────────────────┐   ┌──────────────────────────┐
 16-bit, all grads  │ W frozen (16-bit)         │   │ W frozen, 4-bit NF4       │
 + optimizer state  │      +                    │   │   (dequantized on the fly)│
 for every weight   │ ΔW = B(d×r) · A(r×d)      │   │      +                    │
                    │ trainable, r = 8..64      │   │ ΔW = B·A trainable (16-bit)│
                    └──────────────────────────┘   │ + paged optimizers        │
                                                   └──────────────────────────┘
 Forward pass (both): h = W·x + (α/r)·B·A·x

 Workflow: dataset (prompt→answer) ─▶ load base ─▶ attach adapters (q,k,v,o proj)
           ─▶ train adapters ─▶ evaluate ─▶ save adapter (MBs) ─▶ merge or load at inference
```

### Why low rank works
The *change* a model needs for a new task tends to be low-dimensional. Instead of learning a full `d×d` update, LoRA learns `B (d×r)` and `A (r×d)` with small `r`.

Illustrative math for one 4096×4096 attention matrix:
| Method | Trainable params |
|---|---|
| Full update | 4096 × 4096 ≈ 16.8M |
| LoRA, r=8 | 2 × 4096 × 8 = 65,536 (≈0.4%) |

### Where the memory goes
| Memory item | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Base weights | 16-bit | 16-bit (frozen) | **4-bit** (frozen) |
| Gradients | All params | Adapters only | Adapters only |
| Optimizer state (Adam ≈ 2× params) | All params | Adapters only | Adapters only, paged |
Rough illustration: a 7B model's weights are ~14 GB in 16-bit vs ~3.5–4 GB in 4-bit, so QLoRA can fit on a single ~16–24 GB GPU. (Numbers are approximate; actual usage depends on sequence length and batch size.)

### Key hyperparameters
- **r (rank):** capacity of the adapter (8–64 typical).
- **lora_alpha:** scaling factor (often 2×r).
- **target_modules:** which layers get adapters (attention projections, sometimes MLP).
- **lora_dropout:** regularization.

### Benefits and trade-offs
- ✅ Cheap, fast, small adapter files (MBs); swap multiple adapters on one base model; less catastrophic forgetting since base weights are frozen.
- ⚠️ Slightly below full fine-tuning on some tasks; QLoRA is slower per step due to dequantization; still needs good data and evaluation; requires open weights.

## Example
**Analogy:** Instead of rewriting a 1,000-page textbook (full fine-tuning), you add sticky notes with corrections (LoRA). QLoRA is the same, but the textbook is a compact pocket edition (4-bit) so it fits in your bag, while your sticky notes stay full quality.

**Python – QLoRA setup with Hugging Face `transformers` + `peft`:**
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(            # the "Q" in QLoRA
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "your-org/open-weight-7b-model", quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(                    # the "LoRA" part
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
# trainable params: ~17M || all params: ~7B || trainable%: ~0.25   (illustrative)
# ...then train with a Trainer/SFTTrainer and save: model.save_pretrained("adapter/")
```
(Replace the model name with a real open-weight checkpoint; requires a CUDA GPU and `bitsandbytes`.)

## Common Mistakes / Misconceptions
- **"QLoRA trains a 4-bit model."** The base is stored in 4-bit and frozen; adapters train in higher precision.
- **"LoRA adds new knowledge reliably."** It is best at behavior/format/domain style; use RAG for changing facts.
- **Skipping evaluation** against the base model — adapters can overfit small datasets.
- **Thinking you can LoRA a closed API model** — you need open weights (or a provider's tuning service).

## Key Takeaways
- LoRA: freeze W, learn low-rank `B·A`; trains ~0.1–1% of parameters.
- QLoRA: LoRA on a 4-bit (NF4) quantized frozen base → large models on one GPU.
- Adapters are small, swappable, and mergeable.
- Data quality and evaluation matter more than the choice of rank.

## Related Questions
- [What is LoRA?](../Week-09-Fine-tuning-and-Model-Optimization/07-What-is-LoRA.md)
- [What is QLoRA?](../Week-09-Fine-tuning-and-Model-Optimization/08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](../Week-09-Fine-tuning-and-Model-Optimization/12-How-does-LoRA-reduce-memory-requirements.md)
- [What is 4-bit quantization?](../Week-09-Fine-tuning-and-Model-Optimization/10-What-is-4-bit-quantization.md)
- [What is full fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/11-What-is-full-fine-tuning.md)
- [What is a matrix?](../Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
- [What is Hugging Face?](../Week-06-Working-with-LLMs-and-Model-APIs/03-What-is-Hugging-Face.md)
- [Compare RAG and fine-tuning.](06-Compare-RAG-and-fine-tuning.md)
