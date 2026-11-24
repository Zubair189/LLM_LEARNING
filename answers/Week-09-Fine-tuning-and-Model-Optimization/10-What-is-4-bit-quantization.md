# What is 4-bit quantization?

> Week 09 – Fine-tuning and Model Optimization · Question 10

## In One Line
4-bit quantization stores each weight in just **4 bits**, so every weight must pick from only **16 possible values**, making the model about 4× smaller than normal 16-bit.

## Simple Explanation
4 bits can make 2 × 2 × 2 × 2 = 16 different patterns. So each weight is rounded to one of 16 levels.

That sounds very rough, but it works thanks to one trick:

1. Split the weights into small **blocks** (for example 64 weights each).
2. Give each block its own **scale** (one shared multiplier).
3. Store each weight as a 4-bit code (a level from 0 to 15) inside its block.

So each block has its own "ruler", fitted to its own numbers. With the scales included, it costs a little over 4 bits per weight.

**Memory, in words:** a 7B model needs 7 billion × 0.5 bytes ≈ 3.5 GB, plus a little for scales (roughly 4 GB in total), instead of 14 GB. A 70B model shrinks to about 35–40 GB.

**Popular 4-bit formats** (you'll see these names):
- **NF4:** used for QLoRA training. Its 16 levels are packed more closely near zero, where most weights are.
- **GPTQ** and **AWQ:** made for fast GPU serving. They use a small sample of data to choose the rounding carefully.
- **GGUF** (for example `Q4_K_M`): used by llama.cpp and Ollama to run models on laptops and CPUs.

**Quality:** 4-bit is the usual sweet spot. Big models handle it well; small models (3B and below) can lose more, so test them.

## Real-Life Example
Imagine describing every colour in a photo using a box of only 16 crayons. If each small patch of the photo gets its own box of 16 shades, the picture still looks very close to the original.

## Code Example
Round a block of weights to 16 levels.

```python
block = [0.10, -0.42, 0.05, 0.37, -0.20]     # one small block of weights

scale = max(abs(w) for w in block) / 7       # map the block onto levels -8..7
codes = [max(-8, min(7, round(w / scale))) for w in block]  # 16 possible values
back = [round(c * scale, 2) for c in codes]  # rebuild the weights

print(codes)  # [2, -7, 1, 6, -3]
print(back)   # [0.12, -0.42, 0.06, 0.36, -0.18]
```

**What happened:** every weight became one of 16 small whole numbers (-8 to 7). The rebuilt values are a bit rougher than with 8-bit, but still close.

## Watch Out
- 4-bit saves memory but costs a little quality. Always measure on your task.
- NF4 is mainly for training (QLoRA); GPTQ/AWQ/GGUF are mainly for running models.

## Remember
- 4 bits = 16 levels per weight, with a scale per small block.
- 7B model: ~14 GB → ~3.5–4 GB.
- Common formats: NF4, GPTQ, AWQ, GGUF.

## Related Questions
- [What is quantization?](09-What-is-quantization.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [What is LoRA?](07-What-is-LoRA.md)
