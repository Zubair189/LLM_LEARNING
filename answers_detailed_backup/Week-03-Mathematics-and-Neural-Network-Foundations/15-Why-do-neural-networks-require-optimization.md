# Why do neural networks require optimization?

> Week 03 – Mathematics and Neural Network Foundations · Question 15

## Short Answer
A neural network starts with random weights, so its predictions are initially meaningless. There is no closed-form formula that gives the best weights for a deep, non-linear model with millions or billions of parameters. Instead, we must search for good weights numerically: optimisation repeatedly adjusts the weights to reduce the loss on training data. Optimisation is how a network learns, turning a random function into a useful one.

## Detailed Explanation
**Learning = optimisation.** Training solves
$$\theta^* = \arg\min_\theta \; \frac{1}{N}\sum_{i=1}^N L\big(f_\theta(x_i),\, y_i\big)$$
i.e. find the parameters $\theta$ that make the model's predictions match the data.

**Why there is no shortcut**
| Model | Can we solve for weights directly? |
|---|---|
| Linear regression | Yes, the normal equation $\theta = (X^TX)^{-1}X^Ty$ (though GD is still used at scale) |
| Deep network with non-linear activations | No. The loss surface is non-convex and high-dimensional, so it must be solved iteratively |
| LLM (billions of parameters, trillions of tokens) | No. Even storing $X^TX$ would be impossible |

**What optimisation involves**
1. A **loss function** defining "good".
2. **Gradients** from backpropagation, giving the direction of improvement.
3. An **optimiser** (SGD, momentum, Adam/AdamW) deciding the size and direction of each step.
4. **Hyperparameters:** learning rate and its schedule, batch size, weight decay, number of steps.

**Difficulties optimisation must handle**
- Non-convex landscapes with saddle points and flat regions.
- Vanishing or exploding gradients in deep networks, addressed with residual connections, normalisation, careful initialisation and gradient clipping.
- Noisy gradients from mini-batches.
- Huge scale: LLM pre-training runs for weeks on thousands of GPUs, so optimiser choice and learning-rate schedules have large cost implications.

**Optimisation is not the whole story.** The goal is *generalisation*: good performance on unseen data, not just low training loss. Regularisation (weight decay, dropout), validation monitoring and enough data keep optimisation from simply memorising.

**Beyond pre-training:** fine-tuning (SFT, LoRA) and preference optimisation (RLHF, DPO) are also optimisation problems with different data and losses.

## Example
**Analogy:** Tuning a radio with thousands of knobs to get a clear signal. You can't calculate the right settings in advance, so you nudge the knobs, listen for whether the static decreased, and keep nudging. Optimisation is that systematic nudging, guided by gradients instead of ears.

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1))
y = 3 * X[:, 0] + 2 + 0.1 * rng.normal(size=100)   # true w=3, b=2

w, b, lr = 0.0, 0.0, 0.1                            # start "untrained"
for epoch in range(100):
    err = (w * X[:, 0] + b) - y
    w -= lr * 2 * np.mean(err * X[:, 0])            # gradient of MSE wrt w
    b -= lr * 2 * np.mean(err)                      # gradient of MSE wrt b
    if epoch in (0, 10, 99):
        print(epoch, round(np.mean(err ** 2), 4))
# 0 13.3807     random start: large error
# 10 0.1475
# 99 0.0091     close to the noise floor
print(round(w, 3), round(b, 3))   # 3.006 1.994  -> recovered the true parameters
```

## Common Mistakes / Misconceptions
- **"Lowest training loss = best model."** That can mean overfitting. Judge by validation performance.
- **"Optimisation finds the single perfect solution."** Many different weight settings work about equally well. We need a good one, not the global optimum.
- **Treating optimiser settings as unimportant.** A bad learning rate or schedule can waste an entire training run.
- **Thinking only pre-training needs optimisation.** Fine-tuning and alignment are optimisation too.

> **Note on overlap:** [Q13 – What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md) explains *how* the most common optimiser works. This question is about *why* any optimisation is needed in the first place.

## Key Takeaways
- Networks start random. Optimisation turns them into useful models by minimising a loss.
- Deep non-linear models have no closed-form solution, so iterative gradient-based search is required.
- Loss + gradients (backprop) + optimiser + hyperparameters = the optimisation recipe.
- The real goal is generalisation, not just low training loss.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
