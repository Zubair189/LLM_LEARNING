# What is backpropagation?

> Week 03 – Mathematics and Neural Network Foundations · Question 14

## Short Answer
Backpropagation is the algorithm that efficiently computes the gradient of the loss with respect to every weight in a neural network. It applies the chain rule of calculus backward from the loss, layer by layer, reusing intermediate results so that all gradients cost roughly one extra forward pass. Those gradients are then used by gradient descent (or Adam) to update the weights.

## Detailed Explanation
**The chain rule.** If $L$ depends on $w$ through intermediate values, $w \to z \to \hat{y} \to L$, then
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**The algorithm**
1. **Forward pass:** compute outputs and cache intermediate values ($z$, activations).
2. **Compute the loss.**
3. **Backward pass:** start with $\partial L/\partial L = 1$. Move backward through each operation, multiplying by its local derivative, which gives the gradient for each layer's outputs, weights and inputs.
4. Hand the gradients to the optimiser.

```
Forward:   x ──W1──► z1 ──ReLU──► h ──w2──► ŷ ──► L
Backward:  dW1 ◄──── dz1 ◄──────── dh ◄──── dŷ ◄── 1
```

**Gradients for a 2-layer net** ($\hat{y} = \text{ReLU}(xW_1)\,w_2$, $L = \tfrac12(\hat y - t)^2$):
- $d\hat{y} = \hat{y} - t$
- $dw_2 = d\hat{y}\cdot h$
- $dh = d\hat{y}\cdot w_2$
- $dz_1 = dh \odot \mathbb{1}[z_1 > 0]$ (ReLU passes gradient only where it was active)
- $dW_1 = x^T dz_1$

**Why it matters**
- **Efficiency:** computing each of billions of gradients separately would be impossibly slow. Backprop gets all of them in one backward sweep (reverse-mode automatic differentiation).
- **Automation:** PyTorch and JAX build a computation graph during the forward pass and run backprop automatically with `loss.backward()`.
- **Gradient problems:** multiplying many local derivatives can shrink gradients (vanishing) or blow them up (exploding). Residual connections, normalisation, good initialisation and gradient clipping keep deep Transformers trainable.

## Example
**Analogy:** A company misses its sales target. Management traces the shortfall backward: how much was due to the regional manager, then each team, then each salesperson. Everyone gets a share of the blame proportional to their influence. That share is the gradient.

**Single neuron by hand:** $x=2$, target $y=3$, $w=0.5$, $b=0$, $\hat y = wx+b$, $L=(\hat y - y)^2$.
- Forward: $\hat y = 1$, $L = (1-3)^2 = 4$
- Backward: $\partial L/\partial \hat y = 2(\hat y - y) = -4$; $\partial L/\partial w = -4 \cdot x = -8$; $\partial L/\partial b = -4$

**Two-layer network in numpy:**
```python
import numpy as np

x, t = np.array([1.0, 2.0]), 1.0
W1 = np.array([[0.1,  0.3],
               [0.2, -0.4]])
w2 = np.array([0.5, -0.5])

# forward
z1 = x @ W1                 # [ 0.5 -0.5]
h = np.maximum(0, z1)       # [0.5 0. ]
yhat = h @ w2               # 0.25
L = 0.5 * (yhat - t) ** 2   # 0.28125

# backward (chain rule)
dy = yhat - t               # -0.75
dw2 = dy * h                # [-0.375 -0.   ]
dh = dy * w2
dz1 = dh * (z1 > 0)         # ReLU gate
dW1 = np.outer(x, dz1)
print(dW1)
# [[-0.375  0.   ]
#  [-0.75   0.   ]]

# gradient check on W1[0,0] with finite differences
eps = 1e-6
W1p = W1.copy(); W1p[0, 0] += eps
Lp = 0.5 * (np.maximum(0, x @ W1p) @ w2 - t) ** 2
print(round((Lp - L) / eps, 4))   # -0.375  matches dW1[0,0]
```

## Common Mistakes / Misconceptions
- **"Backprop is the learning algorithm."** It only computes gradients. The optimiser (SGD or Adam) does the learning.
- **"Backprop is how the brain learns."** It was inspired by neuroscience but is not biologically realistic.
- **Forgetting that it needs the cached forward activations.** This is why training uses much more memory than inference (techniques like gradient checkpointing trade compute for memory).
- **Not verifying hand-written gradients.** Use a finite-difference gradient check.

## Key Takeaways
- Backprop = the chain rule applied backward through the computation graph.
- It computes all parameter gradients in roughly the cost of one more forward pass.
- Frameworks automate it (`loss.backward()`), but understanding it helps you debug vanishing and exploding gradients.
- Training loop: forward, then loss, then backward (backprop), then optimiser step.

## Related Questions
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is ReLU?](../Week-03-Mathematics-and-Neural-Network-Foundations/09-What-is-ReLU.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
