# What is Deep Learning?

> Week 01 – Introduction to Artificial Intelligence · Question 7

## In One Line
Deep learning is machine learning that uses **neural networks with many layers**, which learn useful patterns straight from raw data like pictures, sound or text.

## Simple Explanation
A **neural network** is a model made of many small maths units stacked in **layers**. "Deep" simply means it has **many** layers.

In older machine learning, people had to choose the useful clues (called **features**) by hand. For example: "count the number of exclamation marks in an email". In deep learning, the layers find those clues **by themselves**.

Each layer builds on the one before it. For a photo:

1. Early layers spot simple things like edges and colours.
2. Middle layers combine them into shapes like eyes or wheels.
3. Deeper layers recognise whole objects like "face" or "car".

Deep learning took off around 2012 because of three things:

- **Lots of data** from the internet.
- **Fast chips (GPUs)** that do huge amounts of maths quickly.
- **Better methods**, including the **Transformer** design in 2017.

**Link to LLMs:** every LLM is a deep learning model, a very large Transformer network with many layers.

## Real-Life Example
Think of a car factory line. The first station sorts raw parts, the next builds small pieces, and later stations put together the finished car. No single station builds the whole car. Each layer in a deep network is one station.

## Code Example
XOR is "true when exactly one input is 1". One layer cannot solve it, but two layers can. Here the weights are set by hand to show the idea.

```python
def step(z):                          # a very simple "activation"
    return 1 if z > 0 else 0          # fire (1) or not (0)

def xor_network(x1, x2):
    h1 = step(x1 + x2 - 0.5)          # layer 1, unit 1: "at least one is 1" (OR)
    h2 = step(x1 + x2 - 1.5)          # layer 1, unit 2: "both are 1" (AND)
    return step(h1 - h2 - 0.5)        # layer 2: "OR but not AND" = XOR

for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(x1, x2, "->", xor_network(x1, x2))

# Output:
# 0 0 -> 0
# 0 1 -> 1
# 1 0 -> 1
# 1 1 -> 0
```

**What happened:** the first layer built two simple ideas (OR, AND), and the second layer combined them into a harder idea (XOR). That stacking is the heart of "depth". In real deep learning, training finds these weights automatically.

## Watch Out
- **"Deep" does not mean "understands deeply".** It only means "many layers".
- **Deep learning is not always best.** On small tables of data, simpler ML methods often win.

## Remember
- Deep learning = neural networks with **many layers**.
- Layers learn features **automatically**, from simple to complex.
- LLMs are deep learning models.

## Related Questions
- [What is a neural network?](./08-What-is-a-neural-network.md)
- [What are neurons, weights and biases?](./09-What-are-neurons-weights-and-biases.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
