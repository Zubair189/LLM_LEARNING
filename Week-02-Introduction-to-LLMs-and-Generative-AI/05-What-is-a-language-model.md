# What is a language model?

> Week 02 – Introduction to LLMs and Generative AI · Question 5

## Short Answer
A language model is a model that assigns **probabilities to sequences of words or tokens**. Most commonly, it predicts the probability of the **next token given the previous ones**: `P(next | context)`. That one ability lets it score how natural a sentence is, autocomplete text, and generate text by repeatedly sampling the next token. Language models range from simple n-gram counters (used in old phone keyboards) to neural Transformer LLMs with billions of parameters.

## Detailed Explanation
**The core idea, the chain rule of probability:**
```
P("the cat sat") = P("the") × P("cat" | "the") × P("sat" | "the cat")
```
A language model estimates each of these conditional probabilities. Generation simply means sampling one token at a time from `P(next | everything so far)`.

**The evolution of language models:**
| Generation | Approach | Context it can use | Limitation |
|---|---|---|---|
| N-gram (1980s–2000s) | Count word sequences in a corpus | Last n−1 words (e.g. 2) | No long-range context; unseen sequences get zero probability |
| Neural LM / RNN / LSTM (2003–2017) | Neural network over word embeddings | Theoretically unlimited, practically short | Slow (sequential); forgets long context |
| Transformer LM (2017 →) | Self-attention over all tokens | Thousands to millions of tokens (the context window) | Compute grows with context length |
| Large LM (2020 →) | Transformer at massive scale + post-training | Same | Hallucination, cost |

**Two training styles:**
- **Causal / autoregressive LM** (GPT, Claude, Llama): predicts the next token. Used for generation.
- **Masked LM** (BERT): fills in blanked-out words using both sides of the context. Used for understanding and embeddings, not free generation.

**How good is a language model? Perplexity.** Perplexity is roughly "how many choices the model is confused between on average". Lower is better. It is computed as `exp(average negative log-probability of the actual next tokens)`.

> **Note on overlap:** this is the *general concept*. [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md) covers the large-scale, instruction-tuned Transformer systems you use through APIs. Every LLM is a language model, but a 10-line bigram counter is also a language model.

## Example
**Analogy:** Phone keyboard suggestions. After you type "See you", the keyboard offers "soon", "tomorrow" or "later". It has a language model that knows which words usually follow.

**Python: a bigram language model that scores and predicts:**
```python
import math
from collections import Counter, defaultdict

corpus = "i like tea . i like coffee . you like tea . i drink tea .".split()
bigrams = defaultdict(Counter)
for a, b in zip(corpus, corpus[1:]):
    bigrams[a][b] += 1

def p_next(word, nxt):
    total = sum(bigrams[word].values())
    return bigrams[word][nxt] / total if total else 0.0

print(round(p_next("like", "tea"), 2))    # 0.67  (tea after like: 2 of 3)
print(round(p_next("i", "like"), 2))      # 0.67
print(bigrams["like"].most_common(1))     # [('tea', 2)]  -> most likely next word

def sentence_prob(words):
    return math.prod(p_next(a, b) for a, b in zip(words, words[1:]))

print(round(sentence_prob("i like tea".split()), 3))     # 0.444
print(round(sentence_prob("i drink coffee".split()), 3)) # 0.0  <- never seen "drink coffee"
```
The last line shows the classic n-gram weakness: a perfectly reasonable sentence gets probability 0 because that exact pair never appeared. Neural LMs fix this by learning that "tea" and "coffee" are similar.

## Common Mistakes / Misconceptions
- **"Language model = chatbot."** A language model is the underlying probability engine. A chatbot is one application.
- **"High probability means true."** Language models model *likely text*, not *facts*. This is the root of hallucination.
- **Mixing up masked LMs (BERT) with generative LMs (GPT).** They are trained differently and used for different things.
- **"Language models only work on English or natural language."** They work on any token sequence: code, other languages, even DNA.

## Key Takeaways
- A language model estimates `P(next token | previous tokens)`, and from that the probability of whole sequences.
- Generation = repeatedly sample the next token and append it.
- The progression was n-gram → RNN → Transformer → large, instruction-tuned LLMs.
- Perplexity measures LM quality; lower is better.

## Related Questions
- [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
- [What is a token?](./06-What-is-a-token.md)
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
