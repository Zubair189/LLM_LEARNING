# What is ReLU?

> Week 03 – Mathematics and Neural Network Foundations · Question 9

## In One Line
ReLU is the simplest activation function: **if the number is negative, output 0; otherwise, output the number unchanged**.

## Simple Explanation
ReLU stands for **Rectified Linear Unit**. The rule is:

```
ReLU(z) = max(0, z)
```

In words: "take the bigger of 0 and z". So:
- ReLU(−3) = 0
- ReLU(0) = 0
- ReLU(5) = 5

**Why did ReLU become so popular?**
1. **Very cheap:** it is just a comparison, no complicated maths.
2. **Helps deep networks learn:** for positive inputs its slope is always 1, so the learning signal (the **gradient**, which tells each weight how to change) passes back through many layers without shrinking. Older functions like sigmoid shrank this signal, which made deep networks hard to train.
3. **Many zeros:** lots of neurons output exactly 0, which keeps things simple and efficient.

**Its main weakness, "dying ReLU":** if a neuron's input is always negative, it always outputs 0 and its slope is 0, so it stops learning forever. **Leaky ReLU** fixes this by letting a tiny bit of negative through (e.g. 0.01 × z).

**Link to LLMs:** ReLU made deep learning practical from around 2012. Today's LLMs mostly use smooth relatives such as **GELU** and **SwiGLU**, but they follow the same idea.

## Real-Life Example
A one-way valve on a water pipe. Water pushing forward (positive) flows through unchanged. Water pushing backward (negative) is completely blocked.

## Code Example
```python
inputs = [-3.0, -1.0, 0.0, 2.0, 5.0]

relu = [max(0.0, z) for z in inputs]             # negatives become 0
slope = [1 if z > 0 else 0 for z in inputs]      # learning signal passes only where z > 0
leaky = [z if z > 0 else 0.01 * z for z in inputs]   # small leak for negatives

print(relu)    # [0.0, 0.0, 0.0, 2.0, 5.0]
print(slope)   # [0, 0, 0, 1, 1]
print(leaky)   # [-0.03, -0.01, 0.0, 2.0, 5.0]
```

**What happened:** ReLU zeroed the negatives and kept the positives. The slope list shows that only positive inputs pass a learning signal back. Leaky ReLU keeps a tiny bit of the negatives.

## Watch Out
- "ReLU is almost a straight line, so it adds nothing." Not true: the bend at 0 is enough, and many ReLUs together can build complex curves.
- Don't use ReLU on the final output for classification. Use softmax or sigmoid there.

## Remember
- ReLU(z) = max(0, z).
- Cheap, and it keeps the learning signal strong for positive inputs.
- Watch out for "dead" neurons; Leaky ReLU and GELU help.

## Related Questions
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is forward propagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/11-What-is-forward-propagation.md)
