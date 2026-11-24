# What is 4-bit quantization?

> Week 09 – Fine-tuning and Model Optimization · Question 10

> **Scope note:** For the general idea of quantization (scales, PTQ vs QAT, int8) see [What is quantization?](09-What-is-quantization.md). This answer focuses on **4-bit specifically**: the formats, how they differ, and when to use each.

## Short Answer
4-bit quantization stores each weight in just 4 bits, so only 16 possible values per block, with a shared scale per small block of weights. That is a ~4× memory reduction versus fp16, so a 7B model needs ~3.5–4 GB instead of 14 GB. Popular formats are **NF4** (bitsandbytes, used for QLoRA training), **GPTQ** and **AWQ** (calibrated, for fast GPU inference), and **GGUF Q4_K_M** (llama.cpp/Ollama, for CPU and laptops). It is the usual sweet spot between size and quality.

## Detailed Explanation
**How 4 bits can work at all**
- Weights are split into blocks (e.g. 64 values). Each block stores 64 × 4-bit codes plus one scale (and maybe a zero-point).
- Effective size ≈ 4 bits + overhead ≈ **4.1–4.8 bits/weight**.
- The 16 levels are either evenly spaced (INT4) or shaped to the weight distribution (**NF4**, which puts more levels near zero, where most weights are).

```
INT4 levels: |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|   (uniform)
NF4  levels: |----|---|--|-|-||||-|-|--|---|----|              (dense near 0)
```

**Format comparison**

| Format | Calibration data? | Main use | Notes |
|---|---|---|---|
| NF4 (bitsandbytes) | No | QLoRA training, quick loading | Easy; slower inference |
| GPTQ | Yes (small sample) | GPU inference | Uses second-order error correction |
| AWQ | Yes | GPU inference (vLLM, TGI) | Protects "salient" weights based on activations |
| GGUF Q4_K_M | Optional (imatrix) | CPU/Mac (llama.cpp, Ollama) | Mixed-precision blocks |
| FP4 / MXFP4 / NVFP4 | Varies | Newest GPUs (Blackwell) | Hardware-native 4-bit floats |

**Memory math, 7B model**
- fp16: 7×10⁹ × 2 B = 14 GB
- 4-bit: 7×10⁹ × 0.5 B = 3.5 GB, plus ~0.3 GB of scales ≈ 3.8 GB
- A 70B model at 4-bit ≈ 35–40 GB, so it fits on one 48 GB GPU, or on 80 GB with room for the KV cache.

**Quality:** larger models tolerate 4-bit better. 70B at 4-bit usually beats 13B at 16-bit. Small models (≤3B) can lose more, so measure.

## Example
**Analogy:** describing every colour in a painting with a 16-crayon box. If you pick the crayons smartly for that painting (NF4/AWQ), the result looks surprisingly close to the original.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                         quantization_config=bnb, device_map="auto")
print(f"{m.get_memory_footprint()/1e9:.1f} GB")   # ~4.5 GB (vs ~14.5 GB in bf16; embeddings stay 16-bit)

# Back-of-envelope helper
gb = lambda params_b, bits: params_b * bits / 8
print(gb(7, 16), gb(7, 4.5))   # 14.0 3.9375
```

## Common Mistakes / Misconceptions
- Treating "4-bit" as one thing. NF4 is for training via QLoRA; AWQ/GPTQ/GGUF are mostly for serving.
- Expecting exactly `params/2` bytes. Scales and some layers kept in 16-bit (embeddings, lm_head) add overhead.
- Ignoring the KV cache. With long contexts it can exceed the 4-bit weights.
- Skipping evaluation. 4-bit can hurt maths, code or non-English text more than average benchmarks suggest.

## Key Takeaways
- 4-bit = 16 levels per block plus a scale, about a 4× memory saving.
- 7B ≈ 3.5–4 GB, and 70B ≈ 35–40 GB.
- NF4 → QLoRA training; GPTQ/AWQ → GPU serving; GGUF → local/CPU.
- Bigger models handle 4-bit better; always check your own evals.

## Related Questions
- [What is quantization?](09-What-is-quantization.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is LoRA?](07-What-is-LoRA.md)
