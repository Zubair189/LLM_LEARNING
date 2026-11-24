# What is probability?

> Week 03 – Mathematics and Neural Network Foundations · Question 6

## Short Answer
Probability is a number between 0 and 1 that measures how likely an event is: 0 means impossible and 1 means certain. You can read it as a long-run frequency ("a fair die shows 6 about 1/6 of the time") or as a degree of belief. LLMs are probabilistic models: at each step they output a probability for every possible next token and then pick (sample) one.

## Detailed Explanation
**Basic rules**
- $0 \le P(A) \le 1$
- The probabilities of all possible outcomes sum to 1.
- Complement: $P(\text{not }A) = 1 - P(A)$
- For **independent** events: $P(A \text{ and } B) = P(A)\,P(B)$
- **Conditional probability:** $P(A \mid B) = \dfrac{P(A \text{ and } B)}{P(B)}$, the probability of $A$ *given* that $B$ happened.

**Why conditional probability is the heart of LLMs.** A language model estimates
$$P(\text{next token} \mid \text{all previous tokens})$$
and the probability of a whole sentence comes from the **chain rule**:
$$P(w_1, w_2, w_3) = P(w_1)\,P(w_2\mid w_1)\,P(w_3\mid w_1,w_2)$$

| Concept | Where it shows up in LLMs |
|---|---|
| Conditional probability | next-token prediction |
| Chain rule | scoring or generating full sequences |
| Log-probability | multiplying many small probabilities underflows, so models add $\log P$ instead |
| Sampling | temperature, top-k and top-p choose tokens from the predicted probabilities |
| Likelihood | training maximises the probability the model assigns to real text |

**Two interpretations**
- *Frequentist:* the proportion of times an event happens over many trials.
- *Bayesian:* a degree of belief, updated as evidence arrives.

LLM output probabilities are best read as the model's learned beliefs. They are not guaranteed to be calibrated to real-world truth.

## Example
**Analogy:** A weather forecast of "70% chance of rain" means that on days like this, it rained about 7 out of 10 times. An LLM saying P("Paris" | "The capital of France is") = 0.92 is the same kind of statement about text.

**Chain-rule example:** P("the") = 0.05, P("cat" | "the") = 0.01, P("sat" | "the cat") = 0.2 → P("the cat sat") = 0.05 × 0.01 × 0.2 = 0.0001. That is tiny, which is why we use logs: log P = log 0.05 + log 0.01 + log 0.2 ≈ −9.21.

```python
import numpy as np

rng = np.random.default_rng(42)
rolls = rng.integers(1, 7, size=100_000)   # simulate a fair die
print((rolls == 6).mean())                 # 0.16586  (about 1/6 = 0.1667)

p = [0.05, 0.01, 0.2]
print(np.prod(p), np.sum(np.log(p)))       # ~1e-04  -9.2103...
```

## Common Mistakes / Misconceptions
- **Treating a high probability as "true".** An LLM can assign high probability to a fluent but false statement (hallucination).
- **Confusing $P(A\mid B)$ with $P(B\mid A)$.** P(wet ground | rain) is high, but P(rain | wet ground) may not be (sprinklers exist).
- **Assuming independence.** Words in a sentence are highly dependent. That dependence is exactly what an LLM models.
- **Multiplying raw probabilities over long sequences.** The product underflows to 0. Use log-probabilities.

## Key Takeaways
- Probability quantifies uncertainty on a 0–1 scale, and all outcomes sum to 1.
- LLMs model the conditional probability of the next token given the context.
- The chain rule turns next-token probabilities into whole-sequence probabilities.
- Work in log space for numerical stability. Cross-entropy loss is a negative log-probability.

## Related Questions
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
- [What is a loss function?](../Week-03-Mathematics-and-Neural-Network-Foundations/12-What-is-a-loss-function.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
