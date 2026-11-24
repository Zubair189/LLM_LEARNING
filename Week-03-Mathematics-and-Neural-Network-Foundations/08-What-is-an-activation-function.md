# What is an activation function?

> Week 03 – Mathematics and Neural Network Foundations · Question 8

## Short Answer
An activation function is a non-linear function applied to a neuron's weighted sum ($a = f(\mathbf{w}\cdot\mathbf{x} + b)$). Without it, any stack of layers would collapse into a single linear transformation and could only learn straight-line relationships. Common choices are ReLU, sigmoid and tanh. Modern LLMs mostly use smooth variants such as GELU and SwiGLU inside their feed-forward blocks.

## Detailed Explanation
**Why non-linearity is essential:** two linear layers compose into one, $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$. Inserting a non-linear $f$ between them, $W_2 f(W_1\mathbf{x})$, lets the network bend and fold its input space, so it can approximate essentially any function (the universal approximation theorem).

**Common activation functions**
| Name | Formula | Output range | Used in |
|---|---|---|---|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | (0, 1) | binary outputs, gates (LSTM) |
| Tanh | $\tanh(z)$ | (−1, 1) | RNNs, LSTM |
| ReLU | $\max(0,z)$ | [0, ∞) | CNNs, MLPs, the classic default |
| GELU | $z\,\Phi(z)$ | ≈[−0.17, ∞) | BERT, GPT-2/3 |
| SiLU / Swish | $z\,\sigma(z)$ | ≈[−0.28, ∞) | building block of SwiGLU |
| SwiGLU | $\text{SiLU}(xW)\odot(xV)$ | — | Llama, Mistral, Qwen and most modern open LLMs |
| Softmax | $e^{z_i}/\sum e^{z_j}$ | (0,1), sums to 1 | output layer, attention weights |

**Properties that matter**
- **Gradient flow:** sigmoid and tanh *saturate* (flat at the extremes), so their gradients approach 0 and deep networks stop learning (vanishing gradients). ReLU's gradient is exactly 1 for positive inputs.
- **Cost:** ReLU is just a comparison. GELU and SwiGLU cost more but tend to train better in Transformers.
- **Smoothness:** GELU and SiLU are smooth near 0, which tends to help optimisation in very deep networks.

**Where activations sit in a Transformer:** in the position-wise feed-forward network (FFN) of each block, $\text{FFN}(x) = f(xW_1)W_2$, or the gated SwiGLU form. Attention uses softmax to normalise its scores.

## Example
**Analogy:** A neuron is like a dimmer switch with a rule. ReLU says "if the signal is negative, the light stays off; otherwise pass it through as-is." Sigmoid squashes any signal into a 0–100% brightness.

```python
import numpy as np

z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
gelu = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))  # tanh approx.

print(relu)               # [0.  0.  0.  0.5 2. ]
print(sigmoid.round(3))   # [0.119 0.378 0.5   0.622 0.881]
print(tanh.round(3))      # [-0.964 -0.462  0.     0.462  0.964]
print(gelu.round(3))      # [-0.045 -0.154  0.     0.346  1.955]
```

## Common Mistakes / Misconceptions
- **Thinking activations are optional.** Without them, depth adds nothing: the whole network is one linear map.
- **Using sigmoid in hidden layers of deep nets.** It saturates and causes vanishing gradients. Prefer ReLU-family functions.
- **Calling softmax a hidden-layer activation.** It normalises a whole vector into a distribution and is used for outputs and attention, not as a per-neuron hidden activation.
- **Assuming ReLU is what LLMs use.** Most current LLMs use GELU or SwiGLU.

## Key Takeaways
- An activation function adds non-linearity after each weighted sum.
- Non-linearity is what makes deep networks more powerful than linear models.
- ReLU fixed many vanishing-gradient problems. GELU and SwiGLU are the Transformer-era defaults.
- Choice of activation affects gradient flow, speed and final quality.

## Related Questions
- [What is ReLU?](../Week-03-Mathematics-and-Neural-Network-Foundations/09-What-is-ReLU.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
