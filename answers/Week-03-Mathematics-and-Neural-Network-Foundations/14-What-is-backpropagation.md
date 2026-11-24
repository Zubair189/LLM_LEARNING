# What is backpropagation?

> Week 03 – Mathematics and Neural Network Foundations · Question 14

## In One Line
Backpropagation works **backwards from the loss** to figure out how much each weight was to blame, so we know how to change every weight.

## Simple Explanation
After the forward pass, we have a loss (how wrong the model was). Now we need to answer, for **every weight**: "if I nudge this weight up a little, does the loss go up or down, and by how much?" That answer is the weight's **gradient**.

A model can have billions of weights. Checking each one separately would take forever. Backpropagation gets **all** the gradients in one backward sweep.

**How it works:**
1. **Forward pass:** compute the prediction and the loss, and remember the in-between values.
2. **Start at the end:** see how the loss changes when the prediction changes.
3. **Go back one layer at a time:** at each step, multiply by how much that layer affected the next one. This "multiply the effects along the chain" idea is called the **chain rule** from calculus.
4. **Hand the gradients to the optimiser** (gradient descent / Adam), which actually updates the weights.

In tools like PyTorch you just call `loss.backward()` and it does all of this automatically.

**Link to LLMs:** every training step of an LLM runs a forward pass, then backpropagation, then a weight update.

## Real-Life Example
A company misses its sales target. Management traces the shortfall backwards: from the region, to each team, to each salesperson. Everyone gets a share of the blame based on how much they affected the result. That share is the gradient.

## Code Example
One neuron: prediction = `w × x + b`, loss = `(prediction − target)²`.

```python
x, target = 2.0, 3.0
w, b = 0.5, 0.0

# Forward pass
pred = w * x + b                 # 1.0
loss = (pred - target) ** 2      # 4.0

# Backward pass (chain rule)
d_pred = 2 * (pred - target)     # how loss changes with pred: -4.0
d_w = d_pred * x                 # how loss changes with w:   -8.0
d_b = d_pred * 1                 # how loss changes with b:   -4.0
print(d_w, d_b)                  # -8.0 -4.0

# Use the gradients (one gradient-descent step)
w, b = w - 0.1 * d_w, b - 0.1 * d_b
print(round(w, 2), round(b, 2))                # 1.3 0.4
print(round((w * x + b - target) ** 2, 4))     # 0.0  new loss
```

**What happened:** backpropagation found that increasing `w` and `b` would reduce the loss (negative gradients). One update step brought the loss from 4.0 down to 0.

## Watch Out
- Backpropagation does **not** update weights by itself. It only computes gradients; the optimiser does the learning.
- In very deep networks, gradients can shrink to almost nothing (vanishing) or blow up (exploding). Tricks like ReLU and residual connections help.

## Remember
- Backprop = the chain rule, applied backwards.
- It finds every weight's gradient in one sweep.
- Training loop: forward → loss → backward → update.

## Related Questions
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is ReLU?](../Week-03-Mathematics-and-Neural-Network-Foundations/09-What-is-ReLU.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
