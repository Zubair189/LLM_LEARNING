# What are top-k and top-p sampling?

> Week 06 – Working with LLMs and Model APIs · Question 13

## In One Line
Top-k and top-p are filters that **throw away unlikely next tokens**, so the model only picks from sensible choices.

## Simple Explanation
When an LLM writes, it gives every possible next **token** (small piece of text) a chance. Most of these chances are tiny and many of those tokens would be nonsense. **Sampling** means picking one token based on these chances.

Top-k and top-p cut the list down before the pick.

Example: after "I drank a cup of", the chances might be:
coffee 50%, tea 30%, water 12%, milk 5%, sand 3%.

1. **Top-k:** keep only the **k** most likely tokens.
   *top_k = 2* → keep {coffee, tea}. Always exactly 2.
2. **Top-p** (also called **nucleus sampling**): keep the most likely tokens until their chances **add up to p**.
   *top_p = 0.9* → coffee (50) + tea (30) + water (12) = 92% ≥ 90% → keep {coffee, tea, water}.

Then the model picks randomly among the tokens that are left.

**Why top-p is often preferred:** it adapts. If the model is very sure (one token at 95%), only one token is kept. If it is unsure, more are kept. Top-k always keeps the same number.

**How it differs from temperature:** temperature **reshapes** the chances; top-k and top-p **remove** the unlikely ones.

**Link to LLMs:** Anthropic accepts `top_k` and `top_p`, but usually you only need `temperature`. Ollama supports both too.

## Real-Life Example
Choosing a restaurant. Top-k: "pick from the 3 most popular places." Top-p: "pick from the places that together get 90% of all visitors": just one if it is hugely popular, several if tastes are split.

## Code Example
```python
probs = {"coffee": 0.50, "tea": 0.30, "water": 0.12, "milk": 0.05, "sand": 0.03}
ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)  # most likely first

def top_k(k):
    return [token for token, p in ranked[:k]]       # keep the first k tokens

def top_p(p):
    kept, total = [], 0.0
    for token, prob in ranked:                       # add tokens one by one...
        kept.append(token)
        total += prob
        if total >= p:                               # ...until their chances add up to p
            break
    return kept

print("top_k=2:  ", top_k(2))
print("top_p=0.9:", top_p(0.9))

# Output:
# top_k=2:   ['coffee', 'tea']
# top_p=0.9: ['coffee', 'tea', 'water']
```

**What happened:** both filters removed "sand" and "milk". The model would then choose only between the remaining sensible words.

## Watch Out
- Top-p is **not** "the chance the answer is correct". It is just a cut-off for which tokens may be picked.
- Don't change temperature, top-k and top-p all at once. Start with temperature and change one thing at a time.

## Remember
- **Top-k** = keep a **fixed number** of best tokens.
- **Top-p** = keep tokens until chances **add up to p**.
- Both remove **unlikely, odd** choices.

## Related Questions
- [What is the temperature parameter?](12-What-is-the-temperature-parameter.md)
- [What is response streaming?](14-What-is-response-streaming.md)
- [What is a dictionary?](../Week-05-Python-for-LLM-Engineering/04-What-is-a-dictionary.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
