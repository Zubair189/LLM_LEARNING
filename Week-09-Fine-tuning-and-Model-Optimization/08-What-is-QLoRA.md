# What is QLoRA?

> Week 09 – Fine-tuning and Model Optimization · Question 8

## Short Answer
QLoRA (Quantized LoRA) combines **4-bit quantization of the frozen base model** with **LoRA adapters trained in 16-bit**. The base weights are stored in the 4-bit NF4 format and de-quantized on the fly for each matrix multiply, while gradients flow only into the small LoRA matrices. This cuts the base model's memory by about 4× (7B: ~14 GB → ~3.5–4 GB), so you can fine-tune a 7–8B model on one consumer GPU, with quality usually close to 16-bit LoRA.

## Detailed Explanation
**Three ideas from the QLoRA paper (Dettmers et al., 2023)**
1. **NF4 (4-bit NormalFloat):** a 4-bit data type whose 16 levels are placed for normally distributed weights, which suits neural-network weights well.
2. **Double quantization:** the per-block scaling constants are themselves quantized, saving ~0.37 bits/param (roughly 3 GB on a 65B model).
3. **Paged optimizers:** optimizer state can spill to CPU RAM on memory spikes instead of crashing with an out-of-memory error.

**Compute flow**
```
W_nf4 (4-bit, frozen) ──dequantize──► W_bf16 ──┐
                                               ├──► output = W x + (α/r) B A x
LoRA A, B (bf16, trainable) ───────────────────┘
```

**Memory estimate for a 7B model**

| Component | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|
| Base weights | 7B × 2 B ≈ 14 GB | 7B × 0.5 B ≈ 3.5 GB (+~0.2 GB constants) |
| LoRA params + grads + Adam (8.4M) | ~0.13 GB | ~0.13 GB |
| Activations (batch/seq dependent) | 2–8 GB | 2–8 GB |
| **Total (rough)** | **~18–22 GB** | **~6–12 GB** |

**Trade-offs**
- Training is ~20–40% slower because of the de-quantization overhead.
- Quality is a little below bf16 LoRA but usually close.
- For deployment, you either keep the 4-bit base plus the adapter, or merge the adapter into a bf16 copy of the base and re-quantize if needed.

## Example
**Analogy:** you compress a huge reference encyclopedia into a pocket edition (4-bit), and you write your own notes in full-size handwriting (16-bit LoRA). You can carry everything in one backpack (one GPU).

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                             quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)   # enables gradient checkpointing, casts norms
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                       target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                       task_type="CAUSAL_LM"))
print(f"{model.get_memory_footprint() / 1e9:.1f} GB")  # ~3.9 GB (vs ~13.5 GB in bf16)
model.print_trainable_parameters()  # trainable params: 8,388,608 || ... || trainable%: ~0.2
# Pass `model` to trl.SFTTrainer as usual.
```
(The trainable % looks higher than in plain LoRA because 4-bit weights are counted as packed elements.)

## Common Mistakes / Misconceptions
- **"QLoRA trains a 4-bit model."** No. Training happens in bf16 on the LoRA adapters; only the frozen base is 4-bit.
- Forgetting `bnb_4bit_compute_dtype=torch.bfloat16`. The default compute in fp32 is slow.
- Skipping `prepare_model_for_kbit_training` or gradient checkpointing, then running out of memory on activations.
- Expecting QLoRA to speed up inference. Its point is to save **memory**.

## Key Takeaways
- QLoRA = 4-bit NF4 frozen base + bf16 LoRA adapters.
- It brings 7B fine-tuning down from ~20 GB to ~6–12 GB of GPU memory.
- NF4, double quantization and paged optimizers are the key tricks.
- You trade some speed (and a little quality) for a large memory saving.

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is quantization?](09-What-is-quantization.md)
