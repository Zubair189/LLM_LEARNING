# What is a neural network?

> Week 01 – Introduction to Artificial Intelligence · Question 8

## Short Answer
A neural network is a machine learning model made of layers of simple connected units called **neurons**. Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear **activation function**. Stacking layers lets the network represent very complex functions. During training, the weights are adjusted (by backpropagation and gradient descent) to reduce prediction error. The idea is loosely inspired by the brain, but mathematically it is a chain of matrix multiplications and non-linearities.

## Detailed Explanation
**Structure:**
```
 Input layer        Hidden layer(s)        Output layer
  x1 ──┐          ┌──► h1 ──┐
       ├──(W1,b1)─┼──► h2 ──┼──(W2,b2)──► ŷ
  x2 ──┘          └──► h3 ──┘
```
- **Input layer:** receives the raw features (pixels, numbers, token embeddings).
- **Hidden layers:** transform the data step by step. More layers make the network "deeper".
- **Output layer:** produces the prediction: a class probability, a number, or (in an LLM) a probability for every token in the vocabulary.

**What happens in one layer:** `output = activation(W · x + b)`
- `W` is the weight matrix (how strongly each input affects each neuron).
- `b` is the bias vector (a shift).
- The activation (ReLU, sigmoid, GELU, ...) adds **non-linearity**. Without it, any number of stacked layers would collapse into a single linear function.

**How it learns (detailed in Week 3):**
1. **Forward pass:** compute predictions.
2. **Loss:** measure the error.
3. **Backpropagation:** compute how much each weight contributed to the error.
4. **Gradient descent:** nudge the weights to reduce the error. Repeat millions of times.

**Universal approximation:** a network with enough hidden neurons can, in theory, approximate almost any continuous function. In practice, depth and good architecture matter a lot.

**Common types:** feed-forward (MLP), convolutional (CNN), recurrent (RNN/LSTM), and Transformer (attention). LLMs are very large Transformer networks.

> **Note on overlap:** Q9 ([neurons, weights and biases](./09-What-are-neurons-weights-and-biases.md)) zooms into a single neuron. This question is about the whole network. Week 3 ([forward propagation](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md), [backpropagation](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)) covers the maths.

## Example
**Analogy:** A company decision chain. Junior analysts (first layer) each look at raw numbers and give an opinion. Managers (hidden layer) weigh the analysts' opinions, trusting some more than others (weights). The CEO (output) weighs the managers' views and decides. After every wrong decision, everyone adjusts how much they trust each source (training).

**Python: a forward pass through a 2-3-1 network (fixed weights):**
```python
def relu(z):
    return max(0.0, z)

def layer(inputs, weights, biases, act):
    return [act(sum(w * x for w, x in zip(w_row, inputs)) + b)
            for w_row, b in zip(weights, biases)]

x = [1.0, 2.0]                                   # 2 input features
W1 = [[0.5, -0.2], [0.3, 0.8], [-0.6, 0.1]]      # 3 hidden neurons
b1 = [0.0, 0.1, 0.2]
W2 = [[1.0, 0.5, -1.0]]                          # 1 output neuron
b2 = [0.05]

h = layer(x, W1, b1, relu)
y = layer(h, W2, b2, lambda z: z)                # linear output
print([round(v, 2) for v in h])  # [0.1, 2.0, 0.0]
print(round(y[0], 2))            # 1.15
```

## Common Mistakes / Misconceptions
- **"Neural networks work like the brain."** The inspiration is loose. Real neurons are far more complex, and brains don't learn by backpropagation as far as we know.
- **Forgetting the activation function.** Without non-linearity, a deep network is just one linear model.
- **"Bigger is always better."** Larger networks need more data and are more likely to overfit if the data is limited.
- **Treating the network as an explainable rulebook.** Its knowledge is spread across millions or billions of weights.

## Key Takeaways
- A neural network is layers of neurons computing `activation(W·x + b)`.
- Non-linear activations are what let it model complex patterns.
- It learns by adjusting weights to minimize loss (backprop + gradient descent).
- LLMs are giant Transformer neural networks with billions of weights.

## Related Questions
- [What are neurons, weights and biases?](./09-What-are-neurons-weights-and-biases.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
