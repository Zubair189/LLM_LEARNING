# What are top-k and top-p sampling?

> Week 06 – Working with LLMs and Model APIs · Question 13

## Short Answer
Top-k and top-p are sampling filters that limit which tokens an LLM may choose next. **Top-k** keeps only the *k* most likely tokens; **top-p** (nucleus sampling) keeps the smallest set of tokens whose cumulative probability reaches *p* (e.g. 0.9). Both cut off the long tail of unlikely, often nonsensical tokens, and then the model samples from what remains. Temperature reshapes probabilities; top-k/top-p decide which candidates are allowed.

## Detailed Explanation
Pipeline for choosing each token:

```
 logits ─► ÷ temperature ─► softmax ─► top-k filter ─► top-p filter ─► renormalise ─► sample
```

Illustrative next-token probabilities after "I drank a cup of":

| Token | Prob | Cumulative |
|---|---|---|
| coffee | 0.50 | 0.50 |
| tea | 0.30 | 0.80 |
| water | 0.12 | 0.92 |
| milk | 0.05 | 0.97 |
| sand | 0.03 | 1.00 |

- **top_k = 2** → {coffee, tea}, always exactly 2 candidates.
- **top_p = 0.9** → {coffee, tea, water} (cumulative reaches 0.92 ≥ 0.9).

Why top-p is often preferred: it **adapts** to the model's confidence. When the model is sure (one token at 0.95), the nucleus is 1 token; when uncertain, it widens. Top-k uses a fixed count regardless.

| | Top-k | Top-p |
|---|---|---|
| Rule | Fixed number of tokens | Cumulative probability threshold |
| Adaptive? | No | Yes |
| Typical values | 20–50 | 0.8–0.95 |
| Off value | k = vocab size (or unset) | p = 1.0 |

Provider support: Anthropic accepts `top_k` and `top_p` (advanced; usually only `temperature` is needed, and you should not tune `temperature` and `top_p` together); OpenAI exposes `top_p` only; Ollama and `transformers` support both. Some reasoning models restrict sampling parameters — check the docs.

## Example
**Analogy:** Choosing dinner. Top-k: "pick from the 3 most popular restaurants." Top-p: "pick from the restaurants that together account for 90% of where people go" — one place if it dominates, several if tastes are split.

```python
def top_k_filter(probs: dict, k: int) -> dict:
    kept = dict(sorted(probs.items(), key=lambda x: -x[1])[:k])
    s = sum(kept.values())
    return {t: round(p / s, 2) for t, p in kept.items()}

def top_p_filter(probs: dict, p: float) -> dict:
    kept, cum = {}, 0.0
    for t, pr in sorted(probs.items(), key=lambda x: -x[1]):
        kept[t] = pr
        cum += pr
        if cum >= p:
            break
    s = sum(kept.values())
    return {t: round(v / s, 2) for t, v in kept.items()}

probs = {"coffee": 0.50, "tea": 0.30, "water": 0.12, "milk": 0.05, "sand": 0.03}
print(top_k_filter(probs, 2))    # Expected: {'coffee': 0.62, 'tea': 0.38}
print(top_p_filter(probs, 0.9))  # Expected: {'coffee': 0.54, 'tea': 0.33, 'water': 0.13}
```

Setting them in APIs:
```python
import anthropic, requests
client = anthropic.Anthropic()
r = client.messages.create(model="claude-haiku-5-5", max_tokens=50, top_k=40,
    messages=[{"role": "user", "content": "Finish: I drank a cup of"}])
print(r.content[0].text)

# Ollama
requests.post("http://localhost:11434/api/chat", json={"model": "llama3.2", "stream": False,
    "messages": [{"role": "user", "content": "Finish: I drank a cup of"}],
    "options": {"top_k": 40, "top_p": 0.9}}, timeout=120)
```

## Common Mistakes / Misconceptions
- **Thinking top-p is a probability of being correct.** It is a cumulative cutoff on candidate tokens.
- **Setting very low top-p *and* low temperature** — output becomes repetitive and bland.
- **Tweaking every knob at once.** Start with temperature; touch top-k/top-p only if needed, one at a time.
- **Expecting identical knobs everywhere.** Not every API exposes top-k.

## Key Takeaways
- Top-k: keep a fixed number of most likely tokens.
- Top-p: keep the smallest set covering probability *p*; adapts to confidence.
- Both remove the unlikely tail, then the model samples among survivors.
- Temperature reshapes; top-k/top-p truncate. Usually adjust temperature first.

## Related Questions
- [What is the temperature parameter?](12-What-is-the-temperature-parameter.md)
- [What is response streaming?](14-What-is-response-streaming.md)
- [What is a dictionary?](../Week-05-Python-for-LLM-Engineering/04-What-is-a-dictionary.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
