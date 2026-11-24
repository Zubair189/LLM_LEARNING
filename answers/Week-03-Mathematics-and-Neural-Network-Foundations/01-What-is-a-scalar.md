# What is a scalar?

> Week 03 – Mathematics and Neural Network Foundations · Question 1

## In One Line
A scalar is just **one single number**, like `7`, `-2` or `0.5`.

## Simple Explanation
In maths for AI, we group numbers by how they are arranged:

1. **Scalar:** one number on its own, e.g. `7`.
2. **Vector:** a list of numbers, e.g. `[1, 2, 3]`.
3. **Matrix:** a grid (table) of numbers with rows and columns.

A scalar is the smallest piece. Vectors and matrices are simply many scalars put together in an organised way.

A scalar has a size (how big it is) but **no direction**. It is just "how much".

**Where you meet scalars in AI:**
- **Loss:** one number that says how wrong the model is, e.g. `2.31`.
- **Learning rate:** one number that says how big each training step is, e.g. `0.01`.
- **Temperature:** one number that controls how random a chatbot's answers are, e.g. `0.7`.
- **Weights:** every single weight inside a model is a scalar. A "7B" model has about 7 billion of them.

When you multiply a list of numbers by a scalar, **every number in the list gets multiplied**. This is how a learning rate shrinks a whole set of updates in one go.

**Link to LLMs:** settings like temperature and the training loss you see in logs are all scalars.

## Real-Life Example
The temperature on a thermometer, say 21 °C, is a scalar: one number, no direction. Wind is not a scalar, because it has a speed **and** a direction (north, south…). That makes wind a vector.

## Code Example
```python
lr = 0.01                     # a scalar: the learning rate
loss = 2.31                   # a scalar: how wrong the model is

grad = [1.0, 2.0, 3.0]        # a vector (list of numbers)

# Multiply every number in the list by the scalar
step = [lr * g for g in grad]
print(step)                   # [0.01, 0.02, 0.03]

doubled = [2 * g for g in grad]
print(doubled)                # [2.0, 4.0, 6.0]
```

**What happened:** one scalar (`lr` or `2`) changed every number in the list at once. That is how scalars "scale" things, which is where the name comes from.

## Watch Out
- A list with one number, like `[3.5]`, is technically a tiny vector, not a scalar. Some code treats them differently.
- "Scalar" does not mean "unimportant". One scalar, the learning rate, can make or break training.

## Remember
- Scalar = **one number**.
- Loss, learning rate, temperature and each weight are scalars.
- Multiplying by a scalar changes every number in a list.

## Related Questions
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is a matrix?](../Week-03-Mathematics-and-Neural-Network-Foundations/03-What-is-a-matrix.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [What is a learning rate? (bonus)](../Week-03-Mathematics-and-Neural-Network-Foundations/16-BONUS-What-is-a-learning-rate.md)
