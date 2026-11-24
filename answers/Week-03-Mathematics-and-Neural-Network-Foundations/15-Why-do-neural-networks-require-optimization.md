# Why do neural networks require optimization?

> Week 03 – Mathematics and Neural Network Foundations · Question 15

## In One Line
A network **starts with random weights**, and there is no formula that gives the right ones, so we must **search for good weights step by step**. That search is optimization.

## Simple Explanation
**Optimization** means "adjust things until a score is as good as possible". For a neural network, the score is the **loss** (how wrong it is), and we adjust the **weights**.

Why can't we just calculate the right weights directly?

1. **They start random.** A brand-new network gives nonsense answers.
2. **There is no shortcut formula.** For a very simple model (a straight line) there is a formula. But deep networks have bends (activation functions) and millions or billions of weights. No formula can solve that in one go.
3. **So we search.** We make small improvements over and over, guided by the gradient, until the loss is low.

**What you need:** a **loss function** (what "good" means), **gradients** from backpropagation (which way to move), an **optimizer** such as AdamW (how to step), and settings like the learning rate.

**The real goal is not just a low training loss.** We want the model to do well on **new** data it has never seen (this is called **generalisation**). A model that only memorises its training data is not useful.

**Link to LLMs:** pre-training an LLM is a huge optimization run, often weeks on thousands of GPUs. Fine-tuning and alignment later are optimization too.

## Real-Life Example
Tuning an old radio with thousands of knobs. You can't calculate the perfect settings in advance, so you turn the knobs a little, listen whether the static got better, and keep adjusting. Optimization is that careful nudging, guided by gradients instead of ears.

## Code Example
Learn `w` and `b` for the line `y = 3x + 2`, starting from zero.

```python
xs = [0.0, 1.0, 2.0, 3.0]
ys = [2.0, 5.0, 8.0, 11.0]          # data made with y = 3x + 2
w, b, lr = 0.0, 0.0, 0.05           # random-ish start, learning rate

for step in range(501):
    errs = [(w * x + b) - y for x, y in zip(xs, ys)]     # prediction errors
    if step in (0, 50, 500):
        print(step, round(sum(e * e for e in errs) / 4, 4))   # loss (MSE)
    w -= lr * 2 * sum(e * x for e, x in zip(errs, xs)) / 4   # nudge w
    b -= lr * 2 * sum(errs) / 4                              # nudge b
# 0 53.5
# 50 0.0042
# 500 0.0
print(round(w, 2), round(b, 2))     # 3.0 2.0
```

**What happened:** starting from a useless model (loss 53.5), many small nudges found the true values `w = 3`, `b = 2`.

## Watch Out
- The lowest training loss is not always the best model; it may just be memorising. Check performance on unseen data.
- We don't need the one "perfect" set of weights. Many different settings work well; we just need a good one.

Note: Question 13 explains *how* gradient descent works; this question is about *why* we need optimization at all.

## Remember
- Networks start random; optimization makes them useful.
- No formula exists for deep networks, so we search step by step.
- The true goal is doing well on new data.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
