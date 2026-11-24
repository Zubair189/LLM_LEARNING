# What is underfitting and the bias-variance tradeoff?

> Week 01 – Introduction to Artificial Intelligence · Bonus question (added)

## In One Line
Underfitting is when a model is **too simple** to learn the pattern, and the bias-variance tradeoff is the balance between being **too simple** and **too complex**.

## Simple Explanation
**Underfitting** means the model has not learned enough. It does badly on training data **and** on new data. Usually the model is too simple, or it did not train long enough.

Two kinds of mistakes pull in opposite directions:

- **Bias:** errors from being **too simple**. The model makes the same kind of mistake every time, because its idea of the world is too basic. High bias → underfitting.
- **Variance:** errors from being **too sensitive**. The model changes a lot depending on which examples it saw, because it learned the noise. High variance → overfitting.

The **tradeoff**: making a model more complex lowers bias but raises variance. The goal is the **sweet spot** in the middle, where errors on new data are smallest. Some error from random noise in the data can never be removed.

How to tell which problem you have:

1. Training **and** test errors both high → **underfitting**. Use a bigger model, better inputs, or train longer.
2. Training error low but test error high → **overfitting**. Use more data, a simpler model, or stop earlier.

**Link to LLMs:** a model that is too small for a task, or fine-tuned for too few steps, can underfit. Huge LLMs trained on enormous data do not always follow the simple picture, but it is still the right place to start.

## Real-Life Example
Think of archery. High bias: arrows land close together but in the wrong spot. High variance: arrows are scattered all over the target. The goal is a tight group on the bullseye.

## Code Example
Three models on data where the pattern is roughly "y = 2 × x".

```python
train = {1: 2.3, 2: 3.8, 3: 6.4, 4: 7.7, 5: 10.2}   # x -> y (noisy)
test = {6: 12.1, 7: 13.8}                           # new x values
average = sum(train.values()) / len(train)          # 6.08

models = {
    "always average (underfit)": lambda x: average,         # ignores x: too simple
    "y = 2x (good fit)": lambda x: 2 * x,                   # the real pattern
    "memorizer (overfit)": lambda x: train.get(x, 0),       # copies training data
}
def error(model, data):            # average distance from the true y
    return round(sum(abs(model(x) - y) for x, y in data.items()) / len(data), 2)

for name, model in models.items():
    print(f"{name:26s} train {error(model, train):5}  test {error(model, test):5}")
# always average (underfit)  train  2.42  test  6.87
# y = 2x (good fit)          train  0.28  test  0.15
# memorizer (overfit)        train   0.0  test 12.95
```

**What happened:** the underfit model was bad everywhere. The overfit model was perfect in training but terrible on new data. The middle model did well on both.

## Watch Out
- **High test error does not always mean overfitting.** If training error is high too, the model is underfitting, and making it simpler will make things worse.
- **"Bias" here is not about fairness.** It means a consistent, built-in mistake.

## Remember
- Underfitting = bad on training **and** test data.
- Bias = too simple; variance = too sensitive.
- Aim for the sweet spot in between.

## Related Questions
- [What is overfitting?](./12-What-is-overfitting.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is LoRA?](../Week-09-Fine-tuning-and-Model-Optimization/07-What-is-LoRA.md)
