# What is ReLU?

> Week 03 – Mathematics and Neural Network Foundations · Question 9

## Short Answer
ReLU (Rectified Linear Unit) is the activation function $f(z) = \max(0, z)$: it outputs the input if it is positive and 0 otherwise. It is cheap to compute, and its gradient is exactly 1 for positive inputs, so it largely avoided the vanishing-gradient problem of sigmoid and tanh. That made deep networks practical to train from around 2012 onward. Modern LLMs mostly use smooth relatives (GELU, SwiGLU), but ReLU is still the reference point.

## Detailed Explanation
**Definition and derivative**
$$\text{ReLU}(z) = \max(0, z) \qquad \frac{d}{dz}\text{ReLU}(z) = \begin{cases}1 & z>0\\ 0 & z<0\end{cases}$$
(At $z=0$ the derivative is undefined. Libraries simply use 0.)

```
 out
  |        /
  |       /
  |      /
  |_____/_______ z
        0
```

**Why ReLU became the default**
| Advantage | Why it matters |
|---|---|
| No saturation for $z>0$ | Gradients don't shrink as they pass back through many layers |
| Very cheap | A comparison, no exponentials |
| Sparsity | Many neurons output exactly 0, which gives efficient, somewhat disentangled representations |
| Simple | Easy to analyse and to initialise for (He initialisation) |

**Drawbacks**
- **Dying ReLU:** if a neuron's input is always negative (e.g. after a large update pushes its bias very negative), its output and gradient are always 0 and it never recovers.
- **Not zero-centred:** outputs are never negative.
- **Kink at 0:** not smooth, which can matter slightly for optimisation.

**Variants**
- **Leaky ReLU:** $\max(0.01z, z)$, a small slope for negatives that prevents dying.
- **GELU:** $z\,\Phi(z)$, a smooth version that weights inputs by how likely they are to be positive. Used in BERT and GPT-2/3.
- **SiLU/Swish** $z\sigma(z)$ and **SwiGLU** (gated): used in Llama-family and most current open LLMs' feed-forward layers.

## Example
**Analogy:** A one-way valve. Water pushing forward (positive) flows through unchanged. Water pushing backward (negative) is fully blocked.

**Worked example:** inputs [−3, −1, 0, 2, 5] give ReLU outputs [0, 0, 0, 2, 5] and gradients [0, 0, 0, 1, 1].

```python
import numpy as np

z = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
relu = np.maximum(0, z)
grad = (z > 0).astype(float)
leaky = np.where(z > 0, z, 0.01 * z)

print(relu)    # [0. 0. 0. 2. 5.]
print(grad)    # [0. 0. 0. 1. 1.]
print(leaky)   # [-0.03 -0.01  0.    2.    5.  ]
```

## Common Mistakes / Misconceptions
- **"ReLU is linear, so it adds no power."** It is *piecewise* linear. The kink at 0 is enough non-linearity, and many ReLUs combined can approximate complex curves.
- **Applying ReLU to the output layer for classification.** Use softmax (multi-class) or sigmoid (binary) there.
- **Ignoring dead neurons.** Very high learning rates can kill many ReLUs. Monitor the fraction of zero activations.
- **Assuming GPT-style models use ReLU.** Most use GELU or SwiGLU.

## Key Takeaways
- ReLU(z) = max(0, z). Gradient is 1 for positive inputs and 0 for negative inputs.
- It fixed much of the vanishing-gradient problem and made deep networks trainable.
- Watch for "dying ReLU". Leaky ReLU and GELU are common fixes.
- Transformers today mostly use the smooth or gated descendants (GELU, SwiGLU).

## Related Questions
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
