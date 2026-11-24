# What is a learning rate?

> Week 03 – Mathematics and Neural Network Foundations · Bonus question (added)

## In One Line
The learning rate is **how big a step** the model takes each time it updates its weights.

## Simple Explanation
In gradient descent, each update looks like this:

```
new_weight = old_weight − learning_rate × gradient
```

The gradient says **which direction** to move. The learning rate says **how far** to move.

- **Too small:** tiny steps. Training is very slow and may never finish in time.
- **Just right:** the loss goes down steadily.
- **Too big:** the steps jump right over the lowest point. The loss bounces around or even shoots up, sometimes to `NaN` ("not a number", meaning the numbers broke).

It is usually the **most important setting** to get right when training.

**Learning-rate schedules:** big models don't keep one fixed learning rate. They usually:
1. **Warm up:** start very small and slowly increase for the first steps, to keep things stable.
2. **Decay:** then slowly lower it, so the model can settle into a good spot.

**Typical values:** training a Transformer from scratch with AdamW often uses around 0.0001–0.0006. **Fine-tuning** (extra training on top of an already-trained model) uses much smaller values, otherwise the model can forget what it already knew.

**Link to LLMs:** almost every LLM is trained with warm-up followed by a slow decay of the learning rate.

## Real-Life Example
Walking downhill in fog. Baby steps are safe but take all day. Giant leaps make you jump over the valley onto the other slope, maybe even higher than where you started. You want steady, medium-sized steps.

## Code Example
Minimise `(w − 3)²` from `w = 0` for 10 steps with different learning rates. The goal is w = 3.

```python
def train(lr, steps=10):
    w = 0.0
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)    # gradient of (w-3)^2 is 2(w-3)
    return round(w, 2)

for lr in [0.01, 0.1, 0.9, 1.1]:
    print(lr, train(lr))
# 0.01 0.55     too small: barely moved
# 0.1 2.68      good: close to 3
# 0.9 2.68      overshoots back and forth, still gets there
# 1.1 -15.58    too big: blew up
```

**What happened:** the same problem gave very different results depending only on the learning rate. Too small was slow, and too big made `w` fly away from the answer.

## Watch Out
- Don't reuse a pre-training learning rate for fine-tuning; it is usually far too high.
- If the loss suddenly becomes `NaN`, a too-high learning rate is the most common cause.

## Remember
- Learning rate = step size.
- Too small = slow; too big = unstable.
- LLMs use warm-up, then decay.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
- [What is a scalar?](../Week-03-Mathematics-and-Neural-Network-Foundations/01-What-is-a-scalar.md)
- [What is catastrophic forgetting?](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
