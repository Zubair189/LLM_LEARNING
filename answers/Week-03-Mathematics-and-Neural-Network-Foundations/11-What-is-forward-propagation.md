# What is forward propagation?

> Week 03 – Mathematics and Neural Network Foundations · Question 11

## In One Line
Forward propagation (the **forward pass**) means sending the input **through the network, layer by layer, from start to end** to get a prediction.

## Simple Explanation
A neural network is a chain of layers. In the forward pass, data only moves **forwards**:

1. **Input:** start with the numbers you feed in (e.g. `[1, 2]`).
2. **Weighted sum:** each neuron in the first layer multiplies the inputs by its **weights** (learned importance numbers), adds them up, and adds a **bias** (a small extra number).
3. **Activation:** apply a bend such as ReLU (negatives become 0).
4. **Repeat** for the next layer, using the previous layer's outputs as its inputs.
5. **Output:** the last layer gives the prediction.

During **training**, we then compare the prediction with the correct answer using a **loss function** (a "how wrong" score). The forward pass also saves its in-between results, because the next step (backpropagation) needs them.

During **use** (called **inference**), the forward pass is all there is: input in, prediction out.

The forward pass **does not change any weights**. It only calculates.

**Link to LLMs:** one forward pass of an LLM gives the probabilities for **one** next word. To write a 100-word reply, it runs the forward pass about 100 times, adding one word each time.

## Real-Life Example
A factory assembly line. Raw materials go in one end, each station changes the product a little, and the finished item comes out the other end. A quality inspector (the loss) checks it at the end.

## Code Example
```python
x = [1.0, 2.0]                          # input

# Layer 1: two neurons (weighted sum + bias)
z1 = 0.5 * x[0] + 0.2 * x[1] + 0.0     # 0.9
z2 = -1.0 * x[0] + 0.4 * x[1] + 0.1    # -0.1
h1, h2 = max(0.0, z1), max(0.0, z2)    # ReLU -> 0.9 and 0.0

# Layer 2: one output neuron
y = 1.0 * h1 - 0.5 * h2 + 0.2          # prediction
print(round(y, 2))                      # 1.1

target = 1.0                            # the correct answer
print(round((y - target) ** 2, 2))      # 0.01  (squared-error loss)
```

**What happened:** the input flowed through layer 1, ReLU switched off the negative neuron, layer 2 made the prediction 1.1, and the loss measured that it was a little off.

## Watch Out
- The forward pass only **computes**. Weights change later, after backpropagation and the optimiser step.
- An LLM does not write a whole answer in one pass. It needs one forward pass per new token.

## Remember
- Forward pass = input → layers → output.
- Each layer: weighted sum, then activation.
- Training = forward pass, loss, backward pass, update.

## Related Questions
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [Why do neural networks use matrices?](../Week-03-Mathematics-and-Neural-Network-Foundations/04-Why-do-neural-networks-use-matrices.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
