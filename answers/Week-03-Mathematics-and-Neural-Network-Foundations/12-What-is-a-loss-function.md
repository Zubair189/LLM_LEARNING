# What is a loss function?

> Week 03 – Mathematics and Neural Network Foundations · Question 12

## In One Line
A loss function gives **one number that says how wrong the model is**. Lower is better, and training tries to make it as small as possible.

## Simple Explanation
After the model makes a prediction, we need to know how good it was. The loss function compares the **prediction** with the **correct answer** and returns a single score.

- Small loss → the prediction was close.
- Big loss → the prediction was far off.

Training is basically: "change the weights a little so the loss goes down, and repeat".

**Two common loss functions:**

1. **Mean Squared Error (MSE)** – for predicting numbers (like a house price). Take each error, square it (so negatives don't cancel out and big mistakes count extra), then average.
2. **Cross-entropy** – for picking from options (like the next word). It looks at the probability the model gave to the **correct** answer and computes `−log(probability)`.
   - Correct word got probability 0.9 → loss ≈ 0.11 (good).
   - Correct word got probability 0.01 → loss ≈ 4.6 (bad).

**Perplexity** is a friendlier version of the language-model loss: `e^loss`. Roughly, it means "how many words the model is unsure between". Lower is better.

**Link to LLMs:** LLMs are pre-trained with cross-entropy on next-word prediction. The training loss you see in logs is the average of these scores.

## Real-Life Example
A golf score. It sums up in one number how far off you were. You practise (train) to lower it, and the scoring rules shape how you play.

## Code Example
```python
import math

# MSE: predicting numbers
true = [3.0, -0.5, 2.0]
pred = [2.5, 0.0, 2.0]
mse = sum((t - p) ** 2 for t, p in zip(true, pred)) / len(true)
print(round(mse, 3))                   # 0.167

# Cross-entropy: probability given to the correct next word
for p in [0.9, 0.1, 0.01]:
    print(p, round(-math.log(p), 2))   # loss = -log(p)
# 0.9 0.11
# 0.1 2.3
# 0.01 4.61

print(round(math.exp(2.0), 2))         # 7.39  perplexity for a loss of 2.0
```

**What happened:** MSE measured how far the number predictions were. Cross-entropy gave a small loss when the correct word had high probability and a big loss when it had low probability.

## Watch Out
- Low training loss does not guarantee a helpful or truthful model. That is why LLMs get extra training (fine-tuning) afterwards.
- Watch the loss on **unseen** data too (validation loss), not only training loss, to catch memorising.

## Remember
- Loss = one "how wrong" number; training lowers it.
- MSE for numbers, cross-entropy for choices like the next word.
- Perplexity = e^loss.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is probability?](../Week-03-Mathematics-and-Neural-Network-Foundations/06-What-is-probability.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
