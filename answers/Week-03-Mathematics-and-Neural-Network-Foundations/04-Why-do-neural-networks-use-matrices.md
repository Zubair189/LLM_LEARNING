# Why do neural networks use matrices?

> Week 03 – Mathematics and Neural Network Foundations · Question 4

## In One Line
Because one matrix multiplication can do **all the neuron sums for all the inputs at once**, and computers (especially GPUs) are extremely fast at it.

## Simple Explanation
Let's build up step by step.

1. **One neuron** takes some inputs, multiplies each by a **weight** (a number that says how important that input is), and adds them up. Example: inputs `[1, 2]`, weights `[0.1, 0.5]` → 1×0.1 + 2×0.5 = **1.1**.
2. **One layer** has many neurons. If we put each neuron's weights as a column in a grid, we get a **matrix**. Multiplying the input by this matrix gives every neuron's answer in one go.
3. **Many inputs (a batch):** we can also stack many examples as rows of another matrix. Now one multiplication handles every example × every neuron together.

So instead of writing loops inside loops, a whole layer becomes one line: `output = inputs @ weights + bias`.

**Why this matters:**
- **Speed:** GPUs (graphics cards) have special hardware for matrix multiplication. It can be thousands of times faster than a plain Python loop.
- **Simplicity:** the same short code works for 10 neurons or 10,000.
- **Learning:** the maths for training (computing how to change weights) is neat and well known for matrices.

**One important catch:** stacking matrix multiplications alone just gives another matrix multiplication. That is why networks put an **activation function** (a small bend, like ReLU) between layers. Without it, a deep network would act like one simple layer.

**Link to LLMs:** most of an LLM's work, including attention and its feed-forward layers, is matrix multiplication. That is why LLMs run on GPUs.

## Real-Life Example
A teacher has 30 students and 4 test scores each. Instead of calculating each student's final grade one by one, she uses a spreadsheet formula that does the whole class at once. A matrix multiplication is that "whole class at once" formula.

## Code Example
```python
X = [[1.0, 2.0],              # example 1 (2 inputs)
     [3.0, 0.0]]              # example 2
W = [[0.1, 0.2, 0.3],         # weights: 2 inputs -> 3 neurons
     [0.5, 0.6, 0.7]]

# every example x every neuron, in one go
H = [[round(sum(x[k] * W[k][j] for k in range(2)), 2)
      for j in range(3)] for x in X]
print(H)    # [[1.1, 1.4, 1.7], [0.3, 0.6, 0.9]]
```

**What happened:** 2 examples × 3 neurons = 6 weighted sums, all from one matrix multiplication (in NumPy: `X @ W`).

## Watch Out
- Matrices are not just neat notation. They are the **reason** deep learning is fast enough to be practical.
- Without activation functions between them, many layers collapse into one.

## Remember
- One neuron = a weighted sum; one layer = a matrix.
- A batch of inputs is also a matrix, so one multiply does it all.
- GPUs are built for matrix multiplication.

## Related Questions
- [What is a matrix?](../Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is a neural network?](../Week-01-Introduction-to-Artificial-Intelligence/08-What-is-a-neural-network.md)
