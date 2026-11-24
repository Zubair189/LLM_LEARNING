# What are neurons, weights and biases?

> Week 01 – Introduction to Artificial Intelligence · Question 9

## In One Line
A **neuron** is a tiny calculator that adds up its inputs, each multiplied by a **weight**, then adds a **bias** to make a decision.

## Simple Explanation
A neuron does four small steps:

1. Take some inputs (numbers).
2. Multiply each input by its **weight**.
3. Add everything up, then add the **bias**.
4. Pass the total through an **activation function**, a simple rule such as "output 1 if the total is above 0, otherwise 0".

In short: `output = activation(w1·x1 + w2·x2 + ... + bias)`

**Weights** say how much each input matters:
- A big positive weight pushes the answer **up**.
- A negative weight pushes it **down**.
- A weight near 0 means the input is mostly **ignored**.

The **bias** is like a starting mood. It makes the neuron easier or harder to "switch on", even before looking at the inputs.

Weights and biases together are called the model's **parameters**. They start as random numbers, and training slowly adjusts them until the model gives good answers.

**Link to LLMs:** an LLM has **billions** of these parameters. When people say "a 7B model", they mean about 7 billion weights and biases. Week 2's question on model parameters covers this at LLM scale.

## Real-Life Example
Deciding whether to go for a run. Inputs: is it sunny, do you have time, are you tired. Weights: sunny +2, time +3, tired −4. Bias −2 is your general laziness. Add it all up: if the total is above 0, you go running.

## Code Example
That running decision, as a neuron.

```python
def neuron(inputs, weights, bias):
    total = sum(w * x for w, x in zip(weights, inputs)) + bias  # weighted sum + bias
    return 1 if total > 0 else 0          # activation: 1 = run, 0 = stay home

weights = [2, 3, -4]   # sunny, have_time, tired
bias = -2              # general laziness

print(neuron([1, 1, 0], weights, bias))   # 1  (2 + 3 - 2 = 3)  -> run
print(neuron([1, 1, 1], weights, bias))   # 0  (2 + 3 - 4 - 2 = -1) -> stay home
print(neuron([0, 1, 0], weights, bias))   # 1  (3 - 2 = 1)      -> run
```

**What happened:** being tired (weight −4) was enough to cancel out a sunny day with free time. Here we chose the numbers ourselves. In real training, the computer **learns** them from examples.

## Watch Out
- **No single weight holds a fact.** Knowledge is spread across many weights working together.
- **Parameters vs. hyperparameters:** parameters (weights, biases) are learned. Hyperparameters (like the number of layers) are chosen by you.

## Remember
- Neuron = weighted sum + bias → activation.
- Weights = importance of each input; bias = starting push.
- Weights and biases are the **parameters** that training learns.

## Related Questions
- [What is a neural network?](./08-What-is-a-neural-network.md)
- [What are model parameters?](../Week-02-Introduction-to-LLMs-and-Generative-AI/10-What-are-model-parameters.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
