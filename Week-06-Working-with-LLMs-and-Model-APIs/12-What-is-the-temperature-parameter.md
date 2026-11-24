# What is the temperature parameter?

> Week 06 – Working with LLMs and Model APIs · Question 12

## Short Answer
Temperature is a sampling parameter that controls how random an LLM's output is. It rescales the model's next-token probabilities before sampling: low temperature (near 0) makes the most likely token dominate, giving focused, repeatable answers; higher temperature flattens the distribution, giving more varied, creative — and more error-prone — text. Use low values for extraction, classification and code; higher values for brainstorming and creative writing.

## Detailed Explanation
At each step the model outputs a score (logit) for every token. Temperature *T* divides those logits before softmax:

```
 p_i = exp(logit_i / T) / Σ exp(logit_j / T)
```

- **T < 1** → sharper distribution (top token gets more probability).
- **T = 1** → the model's raw distribution.
- **T > 1** → flatter distribution (rare tokens become likelier).
- **T → 0** → effectively greedy decoding (always pick the top token).

Illustrative distribution for the next word after "The sky is":

| Token | T = 0.2 | T = 1.0 | T = 1.5 |
|---|---|---|---|
| blue | 0.99 | 0.70 | 0.55 |
| clear | 0.01 | 0.20 | 0.27 |
| falling | 0.00 | 0.10 | 0.18 |

Ranges differ by provider: Anthropic accepts 0.0–1.0 (default 1.0); OpenAI-style APIs typically 0–2; Ollama defaults around 0.8. Some newer reasoning-focused models restrict or ignore sampling parameters — check the model's docs.

| Use case | Suggested temperature |
|---|---|
| Data extraction, classification, JSON | 0 – 0.2 |
| Q&A, summarisation | 0.3 – 0.7 |
| Brainstorming, stories | 0.8 – 1.0+ |

Note: temperature 0 is *mostly* deterministic, not guaranteed — batching and floating-point effects on GPUs can still cause small differences.

## Example
**Analogy:** Temperature is like a "risk dial" for a jazz musician: at low settings they play the melody exactly as written; at high settings they improvise — sometimes brilliantly, sometimes off-key.

Pure-Python demo of the math:
```python
import math

def softmax_with_temp(logits, T):
    exps = [math.exp(l / T) for l in logits]
    s = sum(exps)
    return [round(e / s, 2) for e in exps]

logits = [2.0, 1.0, 0.5]   # blue, clear, falling
for T in (0.2, 1.0, 2.0):
    print(T, softmax_with_temp(logits, T))
# Expected:
# 0.2 [0.99, 0.01, 0.0]
# 1.0 [0.63, 0.23, 0.14]
# 2.0 [0.48, 0.29, 0.23]
```

With the API:
```python
import anthropic
client = anthropic.Anthropic()
for t in (0.0, 1.0):
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=30, temperature=t,
        messages=[{"role": "user", "content": "Suggest a name for a coffee shop."}])
    print(t, r.content[0].text)
# Expected: at 0.0 the same name on repeated runs; at 1.0 different names each run
```

## Common Mistakes / Misconceptions
- **"Temperature 0 makes the model correct."** It makes it consistent, not accurate.
- **Cranking temperature up for "smarter" answers.** It increases randomness and hallucination risk.
- **Tuning temperature and top_p together blindly.** Change one at a time (Anthropic recommends adjusting one, not both).
- **Assuming the same scale across providers.** 1.0 on one API is not the same "feel" as 1.0 on another.

## Key Takeaways
- Temperature scales logits before softmax to control randomness.
- Low = focused/repeatable; high = diverse/creative.
- Use near 0 for structured or factual tasks.
- Ranges and defaults vary by provider and model.

## Related Questions
- [What are top-k and top-p sampling?](13-What-are-top-k-and-top-p-sampling.md)
- [What is structured output?](15-What-is-structured-output.md)
- [What is NumPy?](../Week-05-Python-for-LLM-Engineering/10-What-is-NumPy.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
