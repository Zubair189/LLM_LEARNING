# What is a loss function?

> Week 03 – Mathematics and Neural Network Foundations · Question 12

## Short Answer
A loss function measures how wrong a model's prediction is, compared with the correct answer, as a single number. Lower is better. Training means adjusting the weights to minimise this number. The loss is also what gradients are computed from, so the choice of loss defines what the model actually learns. LLMs are pre-trained with cross-entropy loss on next-token prediction.

## Detailed Explanation
**Common loss functions**
| Loss | Formula | Task |
|---|---|---|
| Mean Squared Error (MSE) | $\frac{1}{n}\sum (y - \hat{y})^2$ | regression |
| Mean Absolute Error | $\frac{1}{n}\sum \lvert y-\hat{y}\rvert$ | regression robust to outliers |
| Binary cross-entropy | $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | yes/no classification |
| Cross-entropy | $-\log \hat{p}_{\text{correct class}}$ | multi-class, **next-token prediction** |

**Cross-entropy for LLMs.** For each position, the model outputs a distribution over the vocabulary. The loss is the negative log-probability of the token that actually came next:
$$L = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$
- Correct token predicted with p = 0.9: loss = 0.105 (small).
- Correct token predicted with p = 0.01: loss = 4.6 (large).
- Random guessing over 50k tokens: loss = ln 50000 ≈ 10.8.

**Perplexity** = $e^{L}$: roughly "how many tokens the model is choosing between". Loss 2.0 corresponds to perplexity ≈ 7.4.

**Loss vs metric vs objective**
- *Loss:* differentiable, used for gradients (cross-entropy).
- *Metric:* what humans care about (accuracy, BLEU, pass@k, win rate). Often not differentiable.
- *Objective:* loss plus any regularisation terms (e.g. weight decay).

**Loss in later training stages:** supervised fine-tuning still uses cross-entropy (usually only on response tokens). Preference methods such as DPO use a different loss built from log-probability ratios of chosen and rejected answers.

## Example
**Analogy:** A golf score. It summarises how far off you were in one number. You practise (train) to lower it, and the scoring rule shapes your strategy.

```python
import numpy as np

# Regression: MSE
y_true = np.array([3.0, -0.5, 2.0])
y_pred = np.array([2.5,  0.0, 2.0])
print(np.mean((y_true - y_pred) ** 2))     # 0.1666...

# Next-token cross-entropy: model's probabilities for the correct token
p_correct = np.array([0.7, 0.1, 1 / 50000])
print((-np.log(p_correct)).round(3))       # [ 0.357  2.303 10.82 ]

loss = 2.0
print(round(np.exp(loss), 2))              # 7.39  perplexity
```

## Common Mistakes / Misconceptions
- **Optimising a loss that doesn't match your goal.** Low training loss doesn't guarantee helpful, truthful outputs, which is why LLMs get further post-training.
- **Comparing loss across different tokenizers.** Per-token loss depends on token granularity. Compare bits-per-byte instead.
- **Only watching training loss.** Track validation loss to detect overfitting.
- **Passing probabilities where logits are expected** (or vice versa) in library loss functions.

## Key Takeaways
- The loss is a single scalar measuring prediction error, and training minimises it.
- MSE is used for regression and cross-entropy for classification and language modelling.
- LLM pre-training loss = average negative log-probability of the true next token. Perplexity = $e^{\text{loss}}$.
- The loss is what backpropagation differentiates. It defines what "better" means.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is probability?](../Week-03-Mathematics-and-Neural-Network-Foundations/06-What-is-probability.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
