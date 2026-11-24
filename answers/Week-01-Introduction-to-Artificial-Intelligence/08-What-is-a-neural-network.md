# What is a neural network?

> Week 01 – Introduction to Artificial Intelligence · Question 8

## In One Line
A neural network is a model made of **layers of small connected units (neurons)** that pass numbers forward to turn an input into a prediction.

## Simple Explanation
A neural network has three parts:

1. **Input layer:** takes in the data, such as numbers or pixels.
2. **Hidden layers:** transform the data step by step.
3. **Output layer:** gives the answer, for example "cat: 90%".

Each **neuron** does a tiny job. It multiplies each input by a **weight** (how important that input is), adds them up, adds a **bias** (a small adjustment), and passes the result through an **activation function**. The activation function bends the result, for example "turn any negative number into 0". Without it, stacking many layers would be no better than one.

How the network learns:

1. **Forward pass:** make a prediction.
2. **Loss:** measure how wrong it was.
3. **Backpropagation:** work out which weights caused the error.
4. **Gradient descent:** nudge those weights to reduce the error. Repeat many times.

The idea was loosely inspired by the brain, but really it is just a lot of multiplying and adding.

**Link to LLMs:** an LLM is a giant neural network (a Transformer) with billions of weights.

Question 9 zooms in on a single neuron. This question is about the whole network.

## Real-Life Example
Think of a company. Junior staff each look at raw numbers and give an opinion. Managers weigh those opinions, trusting some people more than others (weights). The boss weighs the managers' views and decides. After each mistake, everyone adjusts how much they trust each person (training).

## Code Example
One forward pass through a tiny network: 2 inputs → 3 hidden neurons → 1 output.

```python
def relu(z):                                  # activation: negatives become 0
    return max(0.0, z)

def neuron(inputs, weights, bias):            # weighted sum + bias, then activation
    return relu(sum(w * x for w, x in zip(weights, inputs)) + bias)

x = [1.0, 2.0]                                # 2 input numbers
h1 = neuron(x, [0.5, -0.2], 0.0)              # hidden neuron 1
h2 = neuron(x, [0.3, 0.8], 0.1)               # hidden neuron 2
h3 = neuron(x, [-0.6, 0.1], 0.2)              # hidden neuron 3
hidden = [h1, h2, h3]

# output neuron: combine the hidden values (no activation here)
output = sum(w * h for w, h in zip([1.0, 0.5, -1.0], hidden)) + 0.05

print([round(h, 2) for h in hidden])          # [0.1, 2.0, 0.0]
print(round(output, 2))                       # 1.15
```

**What happened:** the numbers flowed forward layer by layer to give one prediction. Training would change the weights until the predictions become good.

## Watch Out
- **It is not a real brain.** The brain idea is only a loose inspiration.
- **Bigger is not always better.** Large networks need lots of data, or they may just memorise.

## Remember
- Layers of neurons: input → hidden → output.
- Each neuron: weighted sum + bias → activation.
- It learns by adjusting weights to reduce errors.

## Related Questions
- [What are neurons, weights and biases?](./09-What-are-neurons-weights-and-biases.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
