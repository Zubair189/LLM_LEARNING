# What is the temperature parameter?

> Week 06 – Working with LLMs and Model APIs · Question 12

## In One Line
Temperature is a setting that controls **how random or creative** the model's answers are.

## Simple Explanation
An LLM writes one **token** (a small piece of text) at a time. For each step, it gives every possible next token a score. The scores are turned into **probabilities** (chances), and one token is picked.

Temperature changes those chances before the pick:

1. **Low temperature (near 0):** the most likely token wins almost every time. Answers are focused and nearly the same on every run.
2. **Temperature 1:** the model's normal chances.
3. **High temperature:** less likely tokens get a bigger chance. Answers are more varied and creative, but also more likely to go wrong.

Rough guide:
- **0 – 0.2:** extracting data, classifying, JSON, code.
- **0.3 – 0.7:** questions and summaries.
- **0.8 – 1.0:** brainstorming and stories.

Ranges differ by provider. Anthropic accepts 0 to 1 (default 1). Some others go up to 2.

**Link to LLMs:** temperature is one of the most common settings in `client.messages.create(..., temperature=0.2)`.

## Real-Life Example
Think of a jazz musician with a "risk dial". Turned low, they play the song exactly as written. Turned high, they improvise: sometimes brilliant, sometimes off-key.

## Code Example
The maths behind temperature, in plain Python.

```python
import math

def probs_with_temperature(scores, T):
    exps = [math.exp(s / T) for s in scores]       # divide each score by T, then exp
    total = sum(exps)
    return [round(e / total, 2) for e in exps]     # turn into probabilities (sum = 1)

scores = [2.0, 1.0, 0.5]                           # model's scores for: blue, clear, falling
for T in (0.2, 1.0, 2.0):                          # low, normal, high temperature
    print(T, probs_with_temperature(scores, T))

# Output:
# 0.2 [0.99, 0.01, 0.0]
# 1.0 [0.63, 0.23, 0.14]
# 2.0 [0.48, 0.29, 0.23]
```

**What happened:** for the next word after "The sky is", low temperature gave "blue" a 99% chance. High temperature spread the chances out, so "falling" became much more likely.

## Watch Out
- Temperature 0 makes answers **consistent**, not **correct**. A wrong answer can be repeated every time.
- Change temperature **or** top-p, not both at once, so you know what made the difference.

## Remember
- Temperature = **randomness dial**.
- **Low** = focused and repeatable. **High** = creative and varied.
- Use **low** values for facts, JSON and code.

## Related Questions
- [What are top-k and top-p sampling?](13-What-are-top-k-and-top-p-sampling.md)
- [What is structured output?](15-What-is-structured-output.md)
- [What is NumPy?](../Week-05-Python-for-LLM-Engineering/10-What-is-NumPy.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
