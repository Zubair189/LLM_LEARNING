# What is an activation function?

> Week 03 – Mathematics and Neural Network Foundations · Question 8

## In One Line
An activation function is a small **"bend"** applied to each neuron's output, so the network can learn curved, complex patterns instead of only straight lines.

## Simple Explanation
A neuron first computes a **weighted sum**: multiply each input by a weight and add them up. That part is a straight-line (linear) calculation.

Then the activation function takes that sum and **changes it in a non-straight way**. For example, ReLU turns every negative number into 0.

**Why do we need this bend?**
1. Stacking straight-line steps only ever gives another straight line. Ten linear layers are no smarter than one.
2. Adding a bend after each layer lets the network combine many small bends into **any curved shape** it needs.
3. That is what lets networks recognise faces, understand language, and so on.

**Common activation functions:**
- **ReLU:** negative → 0, positive stays the same. Simple and fast.
- **Sigmoid:** squashes any number into the range 0 to 1. Useful for yes/no answers.
- **Tanh:** squashes any number into the range −1 to 1.
- **GELU / SwiGLU:** smooth, curved cousins of ReLU. Most modern LLMs use these.

Sigmoid and tanh go almost flat for big numbers, which can make deep networks learn very slowly (the **vanishing gradient** problem). ReLU-style functions avoid much of this.

**Link to LLMs:** inside every Transformer block there is a feed-forward part with an activation function (usually GELU or SwiGLU).

## Real-Life Example
Think of a light with a dimmer and a rule. ReLU says: "if the signal is negative, keep the light off; otherwise, pass it through as it is". Sigmoid says: "turn any signal into a brightness between 0% and 100%".

## Code Example
```python
import math

inputs = [-2.0, 0.0, 2.0]                # some neuron sums

relu = [max(0.0, z) for z in inputs]      # negative -> 0
sigmoid = [round(1 / (1 + math.exp(-z)), 3) for z in inputs]  # squash to 0..1
tanh = [round(math.tanh(z), 3) for z in inputs]               # squash to -1..1

print(relu)       # [0.0, 0.0, 2.0]
print(sigmoid)    # [0.119, 0.5, 0.881]
print(tanh)       # [-0.964, 0.0, 0.964]
```

**What happened:** the same three inputs were bent in three different ways. ReLU cut off the negative, sigmoid squeezed into 0–1, tanh squeezed into −1 to 1.

## Watch Out
- Activation functions are **not optional**. Without them, a deep network is just one linear layer.
- Softmax is not a normal per-neuron activation. It turns a whole list into probabilities and is used at the output and in attention.

## Remember
- Activation = a non-linear "bend" after each weighted sum.
- It is what makes deep networks powerful.
- ReLU is the classic; GELU and SwiGLU are used in today's LLMs.

## Related Questions
- [What is ReLU?](../Week-03-Mathematics-and-Neural-Network-Foundations/09-What-is-ReLU.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
