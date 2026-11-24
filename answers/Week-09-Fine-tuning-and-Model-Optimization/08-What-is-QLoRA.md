# What is QLoRA?

> Week 09 – Fine-tuning and Model Optimization · Question 8

## In One Line
QLoRA is LoRA plus a **shrunk (4-bit) base model**, so you can fine-tune a big model on a single normal GPU.

## Simple Explanation
With plain LoRA, the base model is frozen but still stored at full size. A 7B model in normal 16-bit format needs about 14 GB just to hold its weights (7 billion × 2 bytes).

QLoRA (Quantized LoRA) adds one idea: **store the frozen base model in 4 bits** instead of 16. That is called **quantization**, meaning storing numbers with less detail to save space.

How it works:

1. Load the base model in a 4-bit format called **NF4**. Now 7B weights take about 3.5 GB (7 billion × 0.5 bytes), plus a little extra.
2. Add normal LoRA adapters, kept at full 16-bit quality.
3. During training, each 4-bit table is briefly expanded back to 16-bit to do the maths, then thrown away.
4. Only the small LoRA adapters learn.

The QLoRA paper also added two helpers: **double quantization** (shrinking the extra scale numbers too) and **paged optimizers** (moving memory to the computer's normal RAM during spikes, instead of crashing).

**Result:** fine-tuning a 7B model fits in roughly 6–12 GB instead of about 18–22 GB for plain LoRA. Training is a bit slower (around 20–40%), and quality is usually very close to plain LoRA.

## Real-Life Example
Imagine carrying a huge reference book. QLoRA is like carrying a compressed pocket edition instead, while your own handwritten notes (the adapters) stay full-size and clear.

## Code Example
Rough memory for just the base weights of a 7B model.

```python
params = 7_000_000_000                # 7 billion weights

bytes_16bit = 2                       # 16-bit = 2 bytes per weight
bytes_4bit = 0.5                      # 4-bit = half a byte per weight

gb = 1_000_000_000                    # bytes in a GB (rounded)
print(f"LoRA base (16-bit) : {params * bytes_16bit / gb} GB")   # 14.0 GB
print(f"QLoRA base (4-bit) : {params * bytes_4bit / gb} GB")    # 3.5 GB
```

**What happened:** going from 16 to 4 bits makes the base model 4× smaller. In real code you ask for this with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` from the `transformers` library.

## Watch Out
- The LoRA adapters are **not** 4-bit; only the frozen base is.
- You still need memory for **activations** (temporary values during training), which grow with batch size and text length.

## Remember
- QLoRA = 4-bit frozen base + normal LoRA adapters.
- 7B model weights: about 14 GB → about 3.5 GB.
- Slightly slower, quality close to LoRA, fits on one GPU.

## Related Questions
- [What is LoRA?](07-What-is-LoRA.md)
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is quantization?](09-What-is-quantization.md)
