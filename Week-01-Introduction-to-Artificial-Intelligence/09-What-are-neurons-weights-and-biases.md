# What are neurons, weights and biases?

> Week 01 – Introduction to Artificial Intelligence · Question 9

## Short Answer
A **neuron** is the basic computing unit of a neural network. It takes several inputs, multiplies each by a **weight**, adds them up, adds a **bias**, and passes the total through an activation function. **Weights** control how strongly each input influences the neuron. The **bias** is a learnable offset that lets the neuron activate even when its inputs are zero, or makes it harder to activate. Weights and biases together are the model's **parameters**, the numbers learned during training. An LLM has billions of them.

## Detailed Explanation
**A single neuron:**
```
x1 ──(w1)──┐
x2 ──(w2)──┼──► Σ (w·x) + b ──► activation f(·) ──► output
x3 ──(w3)──┘          ▲
                      bias b
```
Formula: `output = f(w1·x1 + w2·x2 + ... + wn·xn + b)`

| Component | Role | Intuition |
|---|---|---|
| Input `x` | Data coming in | The evidence |
| Weight `w` | Importance and direction of each input | Positive means it pushes up; negative means it pushes down; near 0 means it is ignored |
| Bias `b` | Shifts the activation threshold | How easily the neuron "fires" |
| Activation `f` | Adds non-linearity | Decides how to squash or gate the signal |

**Why bias matters:** without `b`, the decision line `w·x = 0` must pass through the origin. Bias works like the intercept in `y = mx + c`: it lets the line move.

**Counting parameters:** a fully connected layer from `n` inputs to `m` neurons has `n·m` weights and `m` biases.
- Example: 768 → 3072 gives 768×3072 + 3072 = 2,362,368 parameters.
- Stack many such layers (plus attention) and you reach the billions of parameters in an LLM.

**Where they come from:** weights start as small random numbers. Training (gradient descent) updates every weight and bias a little at each step to reduce the loss. After training they are frozen and saved. Those saved files are what "model weights" or "open-weight model" refer to.

> **Note on overlap:** Week 2's [What are model parameters?](../Week-02-Introduction-to-LLMs-and-Generative-AI/10-What-are-model-parameters.md) covers the same numbers at LLM scale: model size, memory and the "7B vs 70B" meaning. This question covers the mechanics of a single neuron.

## Example
**Analogy:** Deciding whether to go for a run. Inputs: is it sunny (1/0), do you have time (1/0), are you tired (1/0). Weights: sunny +2, time +3, tired −4. Bias −2 represents your general laziness. Add it all up. If the result is above 0, you run.

**Python: that neuron, coded:**
```python
def neuron(inputs, weights, bias):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    return 1 if z > 0 else 0          # step activation

weights = [2, 3, -4]   # sunny, have_time, tired
bias = -2

print(neuron([1, 1, 0], weights, bias))  # 1  (z = 2+3-2 = 3)  -> run
print(neuron([1, 1, 1], weights, bias))  # 0  (z = 3-4 = -1)   -> stay home
print(neuron([0, 1, 0], weights, bias))  # 1  (z = 3-2 = 1)    -> run

# Parameter count of a dense layer
n_in, n_out = 768, 3072
print(n_in * n_out + n_out)  # 2362368
```
Training would *learn* the values 2, 3, −4 and −2 from examples instead of us choosing them.

## Common Mistakes / Misconceptions
- **"Weights store facts like a database."** Knowledge is spread across many weights. No single weight means "Paris is the capital of France".
- **Forgetting the biases when counting parameters.** There are far fewer of them than weights, but they count.
- **"Larger weight means more important feature."** Only if the inputs are on comparable scales. Feature scaling matters.
- **Mixing up parameters and hyperparameters.** Parameters (weights and biases) are learned. Hyperparameters (learning rate, number of layers) are chosen by you.

## Key Takeaways
- Neuron: weighted sum of inputs + bias → activation.
- Weights set the influence of each input; the bias shifts the threshold.
- Weights and biases are the learnable parameters; training tunes them.
- A dense layer has `n_in × n_out + n_out` parameters; LLMs have billions.

## Related Questions
- [What is a neural network?](./08-What-is-a-neural-network.md)
- [What are model parameters?](../Week-02-Introduction-to-LLMs-and-Generative-AI/10-What-are-model-parameters.md)
- [What is a dot product?](../Week-03-Mathematics-and-Neural-Network-Foundations/05-What-is-a-dot-product.md)
- [What is an activation function?](../Week-03-Mathematics-and-Neural-Network-Foundations/08-What-is-an-activation-function.md)
