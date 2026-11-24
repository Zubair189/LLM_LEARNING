# What is a probability distribution?

> Week 03 – Mathematics and Neural Network Foundations · Question 7

## Short Answer
A probability distribution assigns a probability to every possible outcome of a random variable, so that all the probabilities are non-negative and sum (or integrate) to 1. Discrete distributions cover countable outcomes, like a die roll or the next token. Continuous distributions cover real-valued outcomes, like height. At each step, an LLM outputs a categorical distribution over its entire vocabulary, and decoding means choosing a token from it.

## Detailed Explanation
**Two main types**
| Type | Described by | Example | Constraint |
|---|---|---|---|
| Discrete | Probability mass function (PMF), $P(X=x)$ | die roll, next token | $\sum_x P(x) = 1$ |
| Continuous | Probability density function (PDF), $p(x)$ | height, weight initialisation | $\int p(x)\,dx = 1$ |

**Distributions you will meet in LLM work**
- **Categorical:** $K$ outcomes with probabilities $p_1..p_K$. This is the next-token distribution, produced by softmax over the logits.
- **Bernoulli:** a single yes/no outcome. Dropout masks are Bernoulli samples.
- **Gaussian (Normal) $\mathcal{N}(\mu, \sigma^2)$:** the bell curve. Weights are typically initialised from it. Noise in diffusion models is Gaussian.
- **Uniform:** all outcomes equally likely. This is the "know nothing" baseline. A model guessing uniformly over 50k tokens has loss $\ln 50000 \approx 10.8$.

**Summary statistics**
- Mean / expected value: $\mathbb{E}[X] = \sum x\,P(x)$
- Variance: how spread out the values are.
- **Entropy:** $H = -\sum p \log p$, i.e. how uncertain the distribution is. A peaked next-token distribution has low entropy (confident). A flat one has high entropy.

**Decoding = choosing from the distribution**
```
logits --softmax--> distribution --(greedy | temperature | top-k | top-p)--> token
```
- *Greedy:* take the argmax.
- *Temperature:* sharpen (T<1) or flatten (T>1) the distribution before sampling.
- *Top-k / top-p:* cut off the unlikely tail, renormalise, then sample.

## Example
**Analogy:** A loaded die is a distribution, e.g. [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]. Rolling it many times reproduces those frequencies. An LLM's next-token distribution is a loaded die with ~100k faces whose loading changes with every context.

```python
import numpy as np

vocab = ["cat", "dog", "the", "sat"]
p = np.array([0.5, 0.3, 0.15, 0.05])     # categorical distribution
print(p.sum())                           # 1.0  (valid distribution)

rng = np.random.default_rng(0)
samples = rng.choice(vocab, size=10_000, p=p)    # "sampling" decoding
print({w: round(float((samples == w).mean()), 3) for w in vocab})
# {'cat': 0.499, 'dog': 0.299, 'the': 0.152, 'sat': 0.05}

entropy = -(p * np.log(p)).sum()
print(round(entropy, 3))                 # 1.142 nats (max for 4 outcomes is ln4 = 1.386)
```

## Common Mistakes / Misconceptions
- **Reading a PDF value as a probability.** For continuous variables, $p(x)$ can exceed 1. Only areas under the curve are probabilities.
- **Forgetting to renormalise** after top-k / top-p truncation. The remaining probabilities must again sum to 1.
- **Thinking temperature 0 is "sampling".** It is effectively greedy (argmax) decoding.
- **Confusing logits with probabilities.** Logits are unnormalised scores that can be negative. Softmax turns them into a distribution.

## Key Takeaways
- A distribution lists every outcome's probability, and the probabilities sum to 1.
- LLMs output a categorical distribution over the vocabulary at every step.
- Gaussian distributions matter for weight initialisation, and Bernoulli distributions for dropout.
- Entropy measures uncertainty. Decoding strategies reshape or truncate the distribution before picking a token.

## Related Questions
- [What is probability?](../Week-03-Mathematics-and-Neural-Network-Foundations/06-What-is-probability.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
