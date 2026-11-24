# What are model parameters?

> Week 02 – Introduction to LLMs and Generative AI · Question 10

## In One Line
Parameters are the **learned numbers inside a model** (its weights and biases) that store everything it has learned.

## Simple Explanation
A neural network is full of numbers. During training, these numbers are adjusted again and again until the model gives good answers. After training, they are fixed. These numbers are the **parameters**.

- An LLM's size is described by how many parameters it has. "8B" means about **8 billion** parameters, and "70B" about 70 billion.
- More parameters usually means a more capable model, but also a **bigger and more expensive** one.

**How much memory does a model need?**

Each parameter is stored as a number that takes up some space:

- **16-bit** (the usual): 2 bytes per parameter. So an 8B model needs about **16 GB**.
- **8-bit**: 1 byte, about 8 GB.
- **4-bit**: half a byte, about 4 GB.

Storing numbers with fewer bits is called **quantization** (Week 9). The model also needs some extra memory while running.

**Don't mix these up:**
- **Parameters:** learned by training (the weights).
- **Hyperparameters:** choices made by engineers before training, like the number of layers.
- **Settings like `temperature`:** knobs you set when calling the model. They don't change the weights, even though API docs often call them "parameters" too.

**Bigger isn't everything.** Good training data and good fine-tuning matter just as much as size.

Week 1's question on neurons, weights and biases shows what one weight does. This question is about parameters at LLM scale.

## Real-Life Example
Think of a pianist's skill. It lives in millions of tiny muscle and brain adjustments built up over years. No single one "knows" the song, but together they play it. Parameters work the same way.

## Code Example
Estimate the memory needed just to hold an 8-billion-parameter model.

```python
params = 8_000_000_000                       # an "8B" model

for name, bytes_per_param in [("16-bit", 2), ("8-bit", 1), ("4-bit", 0.5)]:
    gb = params * bytes_per_param / 1e9      # total bytes -> gigabytes
    print(f"{name}: about {gb:.0f} GB")

# Output:
# 16-bit: about 16 GB
# 8-bit: about 8 GB
# 4-bit: about 4 GB
```

**What happened:** memory ≈ number of parameters × bytes per parameter. Using fewer bits per number shrinks the model, which is how large models can run on smaller computers.

## Watch Out
- **Parameters are not tokens.** "70B parameters" is model size; "15 trillion tokens" is how much text it trained on.
- **More parameters doesn't always win.** A well-trained smaller model can beat an older, bigger one.

## Remember
- Parameters = the learned weights and biases.
- "8B" = about 8 billion parameters.
- Memory ≈ parameters × bytes per parameter.

## Related Questions
- [What are neurons, weights and biases?](../Week-01-Introduction-to-Artificial-Intelligence/09-What-are-neurons-weights-and-biases.md)
- [What is the difference between an open-weight model and a proprietary model?](./13-What-is-the-difference-between-an-open-weight-model-and-a-proprietary-model.md)
- [What is quantization?](../Week-09-Fine-tuning-and-Model-Optimization/09-What-is-quantization.md)
- [What is the temperature parameter?](../Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
