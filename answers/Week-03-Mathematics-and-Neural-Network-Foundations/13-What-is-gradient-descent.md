# What is gradient descent?

> Week 03 – Mathematics and Neural Network Foundations · Question 13

## In One Line
Gradient descent is how a model learns: it **checks which way makes the loss go up, then takes a small step the other way**, again and again.

## Simple Explanation
Imagine the loss (how wrong the model is) as a hilly landscape. The model's weights decide where you are standing. You want to reach the **lowest valley**, where the loss is smallest.

The **gradient** is the slope under your feet. It points in the direction where the loss goes **up** fastest. So we step in the **opposite** direction.

The steps:
1. Look at the current loss.
2. Compute the gradient (backpropagation does this).
3. Move each weight a small step opposite to its gradient.
4. Repeat many times.

The update rule, written once:

```
new_weight = old_weight − learning_rate × gradient
```

- **gradient:** the slope (which way and how steeply the loss changes).
- **learning_rate:** how big a step to take, e.g. 0.1.

Steep slope → bigger step. Near the bottom, the slope gets flat → steps get smaller.

In practice, models look at a small chunk of data at a time (a **mini-batch**) to estimate the gradient. LLMs use an improved version called **Adam/AdamW**, which picks a suitable step size for each weight automatically.

**Link to LLMs:** every LLM is trained with gradient descent (usually AdamW), repeated millions of times.

## Real-Life Example
You are on a foggy mountain and want to get down to the valley. You can't see far, but you can feel the slope under your feet. You take a step downhill, feel again, and repeat. Your step size is the learning rate.

## Code Example
Find the `w` that makes the loss `(w − 3)²` as small as possible. (The answer is w = 3.)

```python
w = 0.0                          # start somewhere
lr = 0.1                         # learning rate (step size)

for step in range(1, 6):
    grad = 2 * (w - 3)           # slope of (w - 3)^2
    w = w - lr * grad            # step the opposite way
    print(step, round(w, 4), round((w - 3) ** 2, 4))   # step, w, loss
# 1 0.6 5.76
# 2 1.08 3.6864
# 3 1.464 2.3593
# 4 1.7712 1.5099
# 5 2.017 0.9664
```

**What happened:** each step moved `w` closer to 3 and the loss dropped every time. With more steps it would reach almost exactly 3.

## Watch Out
- Gradient descent is not backpropagation. **Backpropagation computes** the gradients; **gradient descent uses** them to update the weights.
- If the learning rate is too big, the steps overshoot and the loss can explode.

## Remember
- Step **opposite** to the gradient (downhill).
- Step size = learning rate.
- LLMs use a variant called AdamW.

## Related Questions
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
