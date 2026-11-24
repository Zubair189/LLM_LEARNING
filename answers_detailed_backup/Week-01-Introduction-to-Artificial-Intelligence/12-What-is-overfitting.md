# What is overfitting?

> Week 01 – Introduction to Artificial Intelligence · Question 12

## Short Answer
Overfitting happens when a model learns the training data *too* well. It memorizes noise and quirks instead of the general pattern, so it performs very well on training data but poorly on new, unseen data. The tell-tale sign is a large gap between training and validation/test performance. Common fixes are more or better data, a simpler model, regularization (L2, dropout), early stopping, and data augmentation.

## Detailed Explanation
**Visual intuition (fitting points with a curve):**
```
 Underfit (too simple)     Good fit                Overfit (too complex)
   ·   ·  ____·            ·  _·_  ·               ·\  /·\    /·
 ___·___··                 ·_/   \_·                · \/   \/\_/ ·
 ·                        ·        ·              ·
 misses the trend         captures the trend      passes through every point,
                                                  wild between them
```

**Learning curves, the classic diagnostic:**
```
loss
 │\
 │ \__________ validation loss starts rising  ← overfitting begins here
 │  \      __/
 │   \___/
 │    \________ training loss keeps falling
 └──────────────────────────► training epochs
            ▲ best point: stop here (early stopping)
```

**Why it happens:**
- The model is too complex for the amount of data (too many parameters).
- Training runs too long.
- The data is noisy or mislabelled, and the model fits the noise.
- Data leakage or duplicates make memorization pay off.

**How to fix it:**
| Technique | Idea |
|---|---|
| More / more diverse data | Harder to memorize, easier to generalize |
| Data augmentation | Flip or crop images, paraphrase text |
| Simpler model | Fewer layers or parameters, lower polynomial degree |
| Regularization (L1/L2, weight decay) | Penalize large weights |
| Dropout | Randomly switch off neurons during training |
| Early stopping | Stop when the validation loss stops improving |
| Cross-validation | A more reliable estimate of generalization |

**Overfitting in LLM work:**
- **Fine-tuning on a few hundred examples for too many epochs** makes the model parrot the training answers and lose general ability. This is related to [catastrophic forgetting](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md).
- **Prompt overfitting:** tweaking a prompt until it passes your five test cases, then watching it fail in production. Use a larger eval set.
- **Benchmark overfitting:** models tuned to leaderboards rather than real tasks.

## Example
**Analogy:** A student memorizes the exact answers to last year's exam. They score 100% on that paper and fail this year's, because the questions changed. They learned the *answers*, not the *subject*.

**Python: polynomial degree 1 vs. "memorize every point":**
```python
import random
random.seed(0)

true_f = lambda x: 2 * x + 1
train = [(x, true_f(x) + random.gauss(0, 1)) for x in range(0, 10)]
test  = [(x + 0.5, true_f(x + 0.5) + random.gauss(0, 1)) for x in range(0, 10)]

# Model A: least-squares straight line (simple)
n = len(train)
mx = sum(x for x, _ in train) / n
my = sum(y for _, y in train) / n
w = sum((x - mx) * (y - my) for x, y in train) / sum((x - mx) ** 2 for x, _ in train)
b = my - w * mx
line = lambda x: w * x + b

# Model B: Lagrange polynomial through all 10 points (degree 9 -> memorizes)
def lagrange(x):
    total = 0.0
    for i, (xi, yi) in enumerate(train):
        term = yi
        for j, (xj, _) in enumerate(train):
            if i != j:
                term *= (x - xj) / (xi - xj)
        total += term
    return total

mse = lambda f, ds: sum((f(x) - y) ** 2 for x, y in ds) / len(ds)
print(f"line : train {mse(line, train):.2f}  test {mse(line, test):.2f}")
print(f"poly9: train {mse(lagrange, train):.2f}  test {mse(lagrange, test):.2f}")
# line : train 0.55  test 1.85
# poly9: train 0.00  test 138.46
```
The degree-9 polynomial has a training error of 0.00 (perfect memorization), but its test error is far larger than the straight line's. That is overfitting.

## Common Mistakes / Misconceptions
- **"Higher training accuracy is always better."** Not if the validation accuracy falls.
- **"Overfitting only happens with big models."** A small model on tiny, noisy data overfits too.
- **Diagnosing overfitting without a held-out set.** You can't see the gap without one.
- **Mixing it up with underfitting.** Underfitting is poor performance on *both* training and test data (see the bonus question).

## Key Takeaways
- Overfitting means great on training data, poor on new data: the model memorized noise.
- Detect it with the train/validation gap and learning curves.
- Fix it with more data, simpler models, regularization, dropout and early stopping.
- It shows up in LLM fine-tuning and prompt engineering too, so always evaluate on held-out examples.

## Related Questions
- [What is underfitting and the bias-variance tradeoff?](./13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is catastrophic forgetting?](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
