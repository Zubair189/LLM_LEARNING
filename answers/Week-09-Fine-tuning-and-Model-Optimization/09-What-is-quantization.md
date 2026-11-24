# What is quantization?

> Week 09 – Fine-tuning and Model Optimization · Question 9

## In One Line
Quantization means storing a model's numbers with **fewer bits** (less detail), so the model takes less memory and often runs faster.

## Simple Explanation
A model's weights are normally stored as 16-bit numbers, which is 2 bytes each. Quantization rounds them onto a smaller set of allowed values, so each one fits in 8 bits (1 byte) or even 4 bits (half a byte).

How it works:

1. Find the biggest weight (ignoring its sign) in a group of weights.
2. Pick a **scale**, so the biggest weight maps to the biggest small whole number (127 for 8-bit).
3. Divide every weight by the scale and **round** it to a whole number. Store those small numbers.
4. When the model needs a weight, multiply back by the scale. You get a value very close to the original.

The tiny difference is called the **quantization error**. Real tools use a separate scale for each small block of weights, so one unusually large number doesn't spoil the rest.

**Memory for a 7B model, in words:**
- 16-bit: 7 billion × 2 bytes ≈ 14 GB
- 8-bit: 7 billion × 1 byte ≈ 7 GB
- 4-bit: 7 billion × 0.5 bytes ≈ 3.5 GB (plus a little for the scales)

**Why it's faster:** to produce each new word, the computer must read all the weights from memory. Fewer bytes means less to read.

**Quality:** 8-bit is almost lossless. 4-bit loses a little. 3-bit and 2-bit start to hurt noticeably.

**Link to LLMs:** this is how people run big models on laptops and single GPUs, and it is the "Q" in QLoRA.

## Real-Life Example
It's like rounding prices to the nearest dollar. $4.97 becomes $5. Your shopping list gets shorter to write, and the total is still almost right.

## Code Example
Squeeze some weights into 8-bit whole numbers and back.

```python
weights = [0.12, -0.50, 0.33, 0.07]           # original numbers

scale = max(abs(w) for w in weights) / 127    # biggest weight -> 127
q = [round(w / scale) for w in weights]       # small whole numbers (8-bit)
back = [round(x * scale, 3) for x in q]       # turn back into decimals

print(q)      # [30, -127, 84, 18]
print(back)   # [0.118, -0.5, 0.331, 0.071]
```

**What happened:** each weight is now stored as a small whole number (1 byte instead of 2). When turned back, the values are very close to the originals, just slightly rounded.

## Watch Out
- Quantization makes a model **smaller**, not smarter. Always check quality after.
- Small models usually lose more quality from heavy quantization than big ones.

## Remember
- Quantization = fewer bits per number.
- 7B model: ~14 GB (16-bit) → ~7 GB (8-bit) → ~3.5 GB (4-bit).
- Small accuracy loss, big memory saving.

## Related Questions
- [What is 4-bit quantization?](10-What-is-4-bit-quantization.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
- [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
