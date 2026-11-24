# What is forward propagation?

> Week 03 – Mathematics and Neural Network Foundations · Question 11

## Short Answer
Forward propagation (the forward pass) is the process of feeding an input through the network layer by layer to produce an output: at each layer, compute a weighted sum $z = xW + b$, apply an activation function, and pass the result on. During training, the forward pass produces a prediction that is compared with the target to compute the loss. At inference, the forward pass *is* the model: for an LLM, one forward pass yields the next-token probabilities.

## Detailed Explanation
**For a 2-layer network:**
$$\mathbf{z}_1 = \mathbf{x}W_1 + \mathbf{b}_1,\quad \mathbf{h} = \text{ReLU}(\mathbf{z}_1),\quad \hat{y} = \mathbf{h}W_2 + \mathbf{b}_2,\quad L = \text{loss}(\hat{y}, y)$$

```
x ──► [ xW1+b1 ] ──► ReLU ──► [ hW2+b2 ] ──► ŷ ──► loss(ŷ, y)
      layer 1               layer 2
```

**Key points**
- Information flows strictly in one direction, input to output (hence "feed-forward").
- Intermediate values ($z_1$, $h$) are **cached**, because backpropagation needs them to compute gradients.
- With batching, $x$ becomes a matrix $X$ of shape `(batch, features)` and the same equations apply.

**Training vs inference**
| | Training | Inference |
|---|---|---|
| Forward pass | yes | yes |
| Store activations | yes (for backprop, a big memory cost) | no (except the KV cache in LLMs) |
| Backward pass | yes | no |
| Dropout | active | disabled |

**Forward pass of a decoder-only LLM**
1. Token IDs → embedding lookup (plus positional information, e.g. RoPE).
2. Repeat N times: (masked) self-attention + residual → feed-forward MLP + residual, with normalisation (typically pre-norm RMSNorm).
3. Final norm → project to vocabulary logits → softmax gives P(next token).

Generating text runs this forward pass once per new token (with a KV cache to avoid redundant work).

## Example
**Analogy:** An assembly line. Raw materials (input) pass through stations (layers). Each station transforms the product a bit, and the finished item (prediction) comes out the end. Quality control (the loss) inspects it afterward.

**Worked numeric example:** $x=[1,2]$.
- $z_1 = xW_1 + b_1 = [0.9, -0.1, -0.9]$
- ReLU gives $h = [0.9, 0, 0]$
- $\hat{y} = hW_2 + b_2 = 0.9\cdot1.0 + 0.2 = 1.1$

```python
import numpy as np

x  = np.array([1.0, 2.0])
W1 = np.array([[0.5, -1.0,  0.3],
               [0.2,  0.4, -0.6]]);  b1 = np.array([0.0, 0.1, 0.0])
W2 = np.array([[1.0], [-0.5], [2.0]]); b2 = np.array([0.2])

z1 = x @ W1 + b1          # layer 1 pre-activation
h  = np.maximum(0, z1)    # ReLU
y  = h @ W2 + b2          # layer 2 (output)
print(z1, h, y)           # [ 0.9 -0.1 -0.9] [0.9 0.  0. ] [1.1]

target = 1.0
print((y - target) ** 2)  # [0.01]  squared-error loss
```

## Common Mistakes / Misconceptions
- **Thinking the forward pass updates weights.** It only computes outputs. Weights change in the optimiser step after backpropagation.
- **Forgetting train/eval mode.** Dropout (and batch-norm statistics) behave differently. Call `model.eval()` for inference in PyTorch.
- **Assuming an LLM generates a whole reply in one forward pass.** Each new token needs another forward pass (autoregressive decoding).
- **Ignoring memory.** Activations stored during training often take more memory than the weights.

## Key Takeaways
- The forward pass maps input to output through repeated "linear transform, then activation" steps.
- It produces predictions, and during training also the loss and cached activations for backprop.
- In LLMs, one forward pass gives one next-token distribution.
- Forward pass, then backward pass, then optimiser step is one training iteration.

## Related Questions
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
