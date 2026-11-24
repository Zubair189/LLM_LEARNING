# What is softmax?

> Week 03 – Mathematics and Neural Network Foundations · Question 10

## In One Line
Softmax turns a list of raw scores into **probabilities that are all positive and add up to 1**.

## Simple Explanation
A model often produces raw scores called **logits**. They can be any number, even negative, and they don't add up to anything special. Softmax converts them into proper probabilities.

**How it works, step by step, for scores `[2, 1, 0]`:**
1. **Raise e to each score** (e ≈ 2.718). This makes every number positive:
   e² ≈ 7.39, e¹ ≈ 2.72, e⁰ = 1.00
2. **Add them up:** 7.39 + 2.72 + 1.00 = 11.11
3. **Divide each by the total:**
   7.39 / 11.11 ≈ **0.67**, 2.72 / 11.11 ≈ **0.24**, 1.00 / 11.11 ≈ **0.09**

Now the numbers add up to 1, and the biggest score still gets the biggest share.

**Why use e (exponentials)?** It makes everything positive, and it **stretches the gaps**: a slightly higher score gets a clearly bigger share, but every option still gets something.

**Temperature:** before softmax, the scores can be divided by a number called the temperature.
- Low temperature (e.g. 0.5) → the top choice gets even more of the share (more predictable).
- High temperature (e.g. 2) → shares become more even (more random, "creative").

**Link to LLMs:** softmax is used in two key places: turning the final scores into **next-word probabilities**, and turning attention scores into **attention weights**.

## Real-Life Example
Think of an election where enthusiasm counts extra. A candidate who is a bit more popular wins a clearly bigger share of the votes, but every candidate still gets some votes, and all shares add up to 100%.

## Code Example
```python
import math

def softmax(scores):
    exps = [math.exp(s) for s in scores]   # step 1: make positive
    total = sum(exps)                      # step 2: add them up
    return [round(e / total, 2) for e in exps]   # step 3: divide

print(softmax([2, 1, 0]))                  # [0.67, 0.24, 0.09]

# Temperature: divide scores before softmax
print(softmax([2/0.5, 1/0.5, 0/0.5]))      # [0.87, 0.12, 0.02]  T=0.5, sharper
print(softmax([2/2, 1/2, 0/2]))            # [0.51, 0.31, 0.19]  T=2, flatter
```

**What happened:** softmax turned the scores into probabilities. A low temperature made the top choice dominate; a high temperature spread the chances out.

## Watch Out
- Softmax probabilities are not a promise of correctness. 0.95 does not mean "right 95% of the time".
- Real code subtracts the largest score first to avoid overflow with huge numbers. The result is the same.

## Remember
- Softmax: raw scores → probabilities that sum to 1.
- Steps: exponentiate, add up, divide.
- Used for next-word prediction and attention weights.

## Related Questions
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
