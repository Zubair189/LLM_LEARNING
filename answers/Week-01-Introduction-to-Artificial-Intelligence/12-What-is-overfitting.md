# What is overfitting?

> Week 01 – Introduction to Artificial Intelligence · Question 12

## In One Line
Overfitting is when a model **memorises** its training examples instead of learning the general pattern, so it does great in practice but badly on new data.

## Simple Explanation
Real data always has some **noise**: small random errors and odd cases. A good model learns the main pattern and ignores the noise. An overfitted model learns the noise too.

How to spot it: compare scores.

- **Training score:** very high.
- **Test score (new data):** much lower.

A big gap between the two is the warning sign.

Why it happens:

1. The model is **too complex** for the amount of data.
2. It **trained for too long** on the same examples.
3. The data is **small or noisy**.

How to fix it:

- Get **more and more varied data**.
- Use a **simpler model**.
- **Stop training early**, when the test-style (validation) score stops improving.
- Use **regularization**: tricks that stop the model from relying too much on any one detail. **Dropout**, which randomly switches off some neurons during training, is one example.

**Link to LLMs:** fine-tuning a model on a few hundred examples for too long can make it parrot those answers and forget general skills. Tweaking a prompt until it passes only your five test cases is a kind of overfitting too.

## Real-Life Example
A student memorises the exact answers to last year's exam. They score 100% on that paper, then fail this year's because the questions changed. They learned the **answers**, not the **subject**.

## Code Example
The real pattern is roughly "y = 2 × x", with a little noise.

```python
train = {1: 2.3, 2: 3.8, 3: 6.4, 4: 7.7, 5: 10.2}   # x -> y (noisy)
test = {6: 12.1, 7: 13.8}                           # new, unseen x values

def memorizer(x):                  # remembers every training answer exactly
    return train.get(x, 0)         # never seen x? it has no idea, says 0
def simple_line(x):                # learned the general pattern
    return 2 * x

def error(model, data):            # average distance from the true y
    return round(sum(abs(model(x) - y) for x, y in data.items()) / len(data), 2)

print("memorizer   train:", error(memorizer, train), " test:", error(memorizer, test))
print("simple line train:", error(simple_line, train), " test:", error(simple_line, test))
# memorizer   train: 0.0  test: 12.95
# simple line train: 0.28  test: 0.15
```

**What happened:** the memoriser was perfect on training data (error 0) but useless on new data. The simple line made small mistakes in training but worked well on new data. That is overfitting vs. good learning.

## Watch Out
- **A higher training score is not always better.** Check the test score too.
- **Don't mix it up with underfitting**, where the model does badly on **both** training and test data.

## Remember
- Overfitting = great on training data, poor on new data.
- Spot it with a big **train vs. test gap**.
- Fix it with more data, simpler models or early stopping.

## Related Questions
- [What is underfitting and the bias-variance tradeoff?](./13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is catastrophic forgetting?](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
