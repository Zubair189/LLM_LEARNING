# What is underfitting and the bias-variance tradeoff?

> Week 01 – Introduction to Artificial Intelligence · Bonus question (added)

## Short Answer
**Underfitting** happens when a model is too simple, or too poorly trained, to capture the real pattern in the data. It performs badly on *both* training and test data. The **bias-variance tradeoff** describes the tension between two sources of error. **Bias** is error from overly simple assumptions, and it causes underfitting. **Variance** is error from being too sensitive to the particular training sample, and it causes overfitting. The goal is the level of model complexity that minimizes the total error on unseen data.

## Detailed Explanation
**Decomposition of expected error (for squared loss):**
```
Expected test error = Bias²  +  Variance  +  Irreducible noise
                      (too     (too         (randomness in the data
                      simple)  wiggly)       that no model can remove)
```

**The classic U-curve (as model complexity grows):**
```
model complexity:  low ──────────────────────────► high
bias²:             HIGH    ↘    medium    ↘    low
variance:          low     ↗    medium    ↗    HIGH
total test error:  high    ↘   ★ MINIMUM  ↗    high
                   underfit    sweet spot      overfit
```

| | Underfitting (high bias) | Overfitting (high variance) |
|---|---|---|
| Training error | High | Very low |
| Test error | High | High |
| Train–test gap | Small | Large |
| Cause | Model too simple, too few features, too little training, too much regularization | Model too complex, too little data, trained too long |
| Fixes | Bigger model, better features, train longer, reduce regularization | More data, regularization, dropout, early stopping, simpler model |

**Intuition:**
- **Bias** asks: averaged over many possible training sets, how far is the model's prediction from the truth? A straight line fitted to a curve is *systematically* wrong.
- **Variance** asks: how much does the prediction change if you retrain on a different sample? A degree-15 polynomial changes wildly.

**A modern nuance for LLMs:** very large neural networks don't always follow the simple U-curve. With huge datasets and models you can see **"double descent"**: test error falls again after the point where the model can fit the training data perfectly. LLMs are heavily over-parameterized, yet they generalize well. This is because they are trained on enormous, diverse data, usually for about one pass (epoch), so they rarely see the same text twice. The bias-variance tradeoff is still the right first mental model, especially for small datasets and fine-tuning.

**Underfitting in LLM practice:**
- A **too-small model** for the task (a 1B model failing at multi-step reasoning).
- **Too few fine-tuning steps**, or a learning rate that is too low, so the model never picks up the target format.
- **LoRA rank too low** for a substantial behaviour change (Week 9).

## Example
**Analogy:** An archery target.
- **High bias, low variance:** all arrows land tightly together, but in the wrong spot. You are consistently wrong (underfit).
- **Low bias, high variance:** arrows are centred on the bullseye *on average*, but scattered everywhere (overfit).
- **Low bias, low variance:** a tight group on the bullseye (the goal).

**Python: measuring bias and variance by retraining on many samples:**
```python
import random, statistics
random.seed(1)

true_f = lambda x: x * x            # the real pattern is a curve
x0 = 3.0                            # point where we measure predictions

def sample(n=10):
    xs = [random.uniform(-4, 4) for _ in range(n)]
    return [(x, true_f(x) + random.gauss(0, 2)) for x in xs]

def constant_model(data):           # very simple -> high bias
    m = statistics.mean(y for _, y in data)
    return lambda x: m

def nearest_neighbour(data):        # very flexible -> high variance
    return lambda x: min(data, key=lambda p: abs(p[0] - x))[1]

for name, fit in [("constant (underfit)", constant_model), ("1-NN (overfit)", nearest_neighbour)]:
    preds = [fit(sample())(x0) for _ in range(2000)]
    bias = statistics.mean(preds) - true_f(x0)
    var = statistics.pvariance(preds)
    print(f"{name:20s} bias^2={bias**2:6.2f}  variance={var:6.2f}")
# constant (underfit)  bias^2= 12.90  variance=  2.76
# 1-NN (overfit)       bias^2=  0.02  variance= 12.39
```
The constant model has a large bias² and a small variance. 1-nearest-neighbour has a small bias² and a larger variance. Neither is ideal; something in between, such as a quadratic fit or k-NN with k≈3, would minimize the total.

## Common Mistakes / Misconceptions
- **"Low training error means a good model."** It could be overfitting. Also check the test error.
- **"High test error always means overfitting."** If the training error is also high, the problem is underfitting, and adding regularization will make it worse.
- **"Bias here means unfairness."** Statistical bias (systematic error) is different from social or algorithmic bias, though both matter.
- **"You can always drive both bias and variance to zero."** Irreducible noise sets a floor.

## Key Takeaways
- Underfitting means bad on training *and* test data; the model is too simple or undertrained.
- Bias = systematic error from simplistic assumptions; variance = sensitivity to the training sample.
- Total error = bias² + variance + noise; aim for the complexity sweet spot.
- Diagnose with train vs. validation error: both high → bias; big gap → variance.
- Huge models show "double descent", but the tradeoff still guides fine-tuning decisions.

## Related Questions
- [What is overfitting?](./12-What-is-overfitting.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is LoRA?](../Week-09-Fine-tuning-and-Model-Optimization/07-What-is-LoRA.md)
