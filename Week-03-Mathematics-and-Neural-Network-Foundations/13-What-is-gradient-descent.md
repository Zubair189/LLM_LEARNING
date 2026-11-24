# What is gradient descent?

> Week 03 – Mathematics and Neural Network Foundations · Question 13

## Short Answer
Gradient descent is an iterative algorithm for minimising a loss function. At each step, compute the gradient (the direction in which the loss increases fastest) and move the parameters a small step in the opposite direction: $\theta \leftarrow \theta - \eta \nabla_\theta L$. Repeating this walks the parameters "downhill" toward low loss. Every neural network, including LLMs, is trained with a variant of it, most commonly mini-batch stochastic gradient descent with the Adam/AdamW optimiser.

## Detailed Explanation
**Update rule**
$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)$$
- $\theta$: all model parameters (weights and biases)
- $\nabla_\theta L$: the gradient, a vector of partial derivatives, one per parameter, computed by backpropagation
- $\eta$: the **learning rate** (step size)

**Intuition:** the gradient points uphill, so we step downhill. Where the slope is steep the gradient is large and so is the step. Near a minimum, the slope and the steps shrink.

```
loss
 |\                 /
 | \               /
 |  \   ●→        /      ● = current parameters; step opposite to slope
 |   \     ●→    /
 |    \_____●___/        minimum
 +-----------------------> θ
```

**Variants by how much data each step uses**
| Variant | Data per step | Trade-off |
|---|---|---|
| Batch GD | entire dataset | exact gradient but slow, impossible for LLM-scale data |
| Stochastic GD (SGD) | 1 example | fast, very noisy |
| **Mini-batch GD** | e.g. 32 to millions of tokens | the standard: GPU-efficient, and the noise is manageable |

**Improved optimisers**
- **Momentum:** accumulates a running average of gradients to roll through small bumps and dampen zig-zags.
- **Adam / AdamW:** per-parameter adaptive step sizes from running averages of the gradient and squared gradient. AdamW (with decoupled weight decay) is the de facto standard for training Transformers. Newer optimisers (e.g. Muon) are being adopted for some large runs.
- **Learning-rate schedules:** warm-up followed by cosine or linear decay are typical for LLMs.

**Challenges:** local minima and saddle points (less of a problem in very high dimensions than intuition suggests), a poorly chosen learning rate, and noisy gradients.

## Example
**Analogy:** You are on a foggy mountain and want to reach the valley. You can't see far, but you can feel the slope under your feet. You take a step downhill, feel again, and repeat. The step length is your learning rate.

**Worked example:** minimise $L(w) = (w-3)^2$, gradient $L'(w) = 2(w-3)$, start $w=0$, $\eta = 0.1$.
- Step 1: grad = 2(0−3) = −6 → w = 0 − 0.1·(−6) = 0.6
- Step 2: grad = 2(0.6−3) = −4.8 → w = 0.6 + 0.48 = 1.08

```python
w, lr = 0.0, 0.1
for step in range(1, 6):
    grad = 2 * (w - 3)          # dL/dw
    w -= lr * grad              # gradient descent update
    print(step, round(w, 4), round((w - 3) ** 2, 4))
# 1 0.6 5.76
# 2 1.08 3.6864
# 3 1.464 2.3593
# 4 1.7712 1.5099
# 5 2.017 0.9664     -> converging toward w = 3, loss -> 0
```

## Common Mistakes / Misconceptions
- **Confusing gradient descent with backpropagation.** Backprop *computes* the gradients. Gradient descent *uses* them to update the weights.
- **Thinking it always finds the global minimum.** For neural networks it finds a good-enough low-loss region, and that is usually sufficient.
- **Setting the learning rate carelessly.** Too high diverges, too low barely moves.
- **Forgetting to zero gradients between steps** in PyTorch (`optimizer.zero_grad()`), which silently accumulates them.

## Key Takeaways
- Gradient descent: step parameters opposite to the gradient, scaled by the learning rate.
- Mini-batch SGD with AdamW plus warm-up and decay schedules is the standard recipe for LLMs.
- Backprop supplies the gradient, and the optimiser applies the update.
- The learning rate is the most important hyperparameter to tune.

## Related Questions
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
