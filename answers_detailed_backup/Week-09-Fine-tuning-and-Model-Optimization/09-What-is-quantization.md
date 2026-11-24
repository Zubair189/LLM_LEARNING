# What is quantization?

> Week 09 – Fine-tuning and Model Optimization · Question 9

> **Scope note:** This question covers the *general concept* of quantization (why, how, and the types). For the specifics of 4-bit formats (NF4, GPTQ, AWQ, GGUF Q4) see [What is 4-bit quantization?](10-What-is-4-bit-quantization.md).

## Short Answer
Quantization stores a model's numbers (weights, and sometimes activations and the KV cache) with **fewer bits**, for example going from 16-bit floats to 8-bit or 4-bit integers. It maps each value to the nearest point on a coarser grid using a scale factor. This shrinks memory roughly in proportion to the bit-width and often speeds up inference, because LLM decoding is limited by memory bandwidth. The cost is a small, usually acceptable, loss of accuracy.

## Detailed Explanation
**Basic recipe (symmetric int8)**
```
scale = max(|w|) / 127
q     = round(w / scale)      # stored as int8
w_hat = q * scale             # de-quantized at compute time
error = w - w_hat             # quantization error
```
Real schemes compute a scale **per block** (e.g. every 32–128 weights) or per channel, so that one outlier doesn't ruin the precision of everything else.

**Memory for a 7B model**

| Precision | Bytes/param | 7B weights |
|---|---|---|
| fp32 | 4 | 28 GB |
| fp16 / bf16 | 2 | **14 GB** |
| int8 / fp8 | 1 | 7 GB |
| 4-bit | 0.5 | **3.5 GB** (+ scales ≈ 4 GB) |

**Types**
- **Post-training quantization (PTQ):** quantize a trained model with no retraining. Examples: bitsandbytes, GPTQ, AWQ, GGUF (llama.cpp). This is the most common approach.
- **Quantization-aware training (QAT):** simulate quantization during training so the model learns to tolerate it. Better quality, but more work.
- **What gets quantized:** weights only (most common), weights and activations (e.g. W8A8, FP8 on H100/B200-class GPUs), or the KV cache (which helps with long contexts).

**Why it speeds up inference:** generating each token requires reading all the weights from GPU memory. Half the bytes means roughly half the read time, as long as the kernels support the format.

**Quality rule of thumb:** 8-bit is nearly lossless; 4-bit loses a little (often <1–2 points on benchmarks); 3-bit and 2-bit degrade noticeably, especially for small models.

## Example
**Analogy:** storing a photo as JPEG instead of RAW. The file is much smaller and looks almost the same, but zoom in far enough and you'll see artifacts.

```python
import torch

w = torch.randn(4096) * 0.02          # pretend weight row
scale = w.abs().max() / 127
q = torch.round(w / scale).to(torch.int8)
w_hat = q.float() * scale

print(w.element_size(), q.element_size())     # 4 1  -> 4x smaller than fp32
print(f"{(w - w_hat).abs().mean():.2e}")       # ~1.4e-04 mean abs error (about 1% of a typical weight)

# Loading an LLM in 8-bit with Hugging Face:
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# m = AutoModelForCausalLM.from_pretrained(model_id,
#        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
```

## Common Mistakes / Misconceptions
- **"Quantization always makes things faster."** Only if there are efficient kernels; some formats de-quantize on the fly and can be slower than bf16.
- Forgetting the KV cache and activations. Weight quantization alone doesn't shrink them.
- Assuming all 4-bit formats are interchangeable. NF4, GPTQ, AWQ and GGUF differ in quality, speed and tooling.
- Quantizing then fine-tuning the quantized weights directly. Use QLoRA (train adapters) instead.

## Key Takeaways
- Quantization = fewer bits per number, so smaller and usually faster models.
- Memory scales with bits: 7B is 14 GB in fp16, 7 GB in int8 and ~3.5 GB in 4-bit.
- PTQ is the common path; QAT gives better quality for more effort.
- Always re-run your evals after quantizing.

## Related Questions
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
