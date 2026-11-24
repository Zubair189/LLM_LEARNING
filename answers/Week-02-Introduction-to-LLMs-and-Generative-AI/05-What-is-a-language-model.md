# What is a language model?

> Week 02 – Introduction to LLMs and Generative AI · Question 5

## In One Line
A language model is a model that **guesses how likely each next word is**, given the words before it.

## Simple Explanation
At its heart, a language model answers one question: **"What word probably comes next?"**

After "I like", it might say: "tea" 67%, "coffee" 33%. These percentages are called **probabilities**: numbers from 0 to 1 (or 0% to 100%) that show how likely something is.

With this one skill, a language model can:

1. **Autocomplete** text (like phone keyboard suggestions).
2. **Score** sentences: "I like tea" is more natural than "tea like I".
3. **Generate** text: pick a next word, add it, and repeat.

Language models have grown over time:

- **Counting models (n-grams):** count which words follow which in a text. Simple, but they only look at the last word or two.
- **Neural network models:** learn smarter patterns and can see longer context.
- **Transformers and LLMs:** huge neural models that can look at thousands of words at once.

Question 2 covers large language models. This question is the **general idea**: every LLM is a language model, but a tiny word-counting program is a language model too.

## Real-Life Example
Your phone keyboard. After you type "See you", it suggests "soon", "tomorrow" or "later". It has a small language model that knows which words usually come next.

## Code Example
A tiny counting language model.

```python
from collections import Counter, defaultdict

text = "i like tea . i like coffee . you like tea . i drink tea .".split()

follows = defaultdict(Counter)            # word -> counts of the words after it
for word, nxt in zip(text, text[1:]):
    follows[word][nxt] += 1

def p_next(word, nxt):                    # chance that nxt comes after word
    total = sum(follows[word].values())
    return follows[word][nxt] / total

print(round(p_next("like", "tea"), 2))    # 0.67  ("tea" came after "like" 2 of 3 times)
print(round(p_next("like", "coffee"), 2)) # 0.33
print(follows["like"].most_common(1))     # [('tea', 2)]  <- best guess for next word
print(p_next("drink", "coffee"))          # 0.0   <- never seen, so "impossible"
```

**What happened:** the model learned next-word chances by counting. The last line shows its weakness: "drink coffee" is perfectly normal, but it scores 0 because it never appeared. Neural models fix this by learning that "tea" and "coffee" are similar.

## Watch Out
- **Likely is not the same as true.** A language model predicts what text *usually* looks like, not what is factually correct.
- **A language model is not a chatbot.** It is the engine; a chatbot is one thing you can build with it.

## Remember
- A language model predicts the **next word** (token).
- Generating text = predict, add, repeat.
- It grew from simple counting to giant Transformers (LLMs).

## Related Questions
- [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
- [What is a token?](./06-What-is-a-token.md)
- [What is a probability distribution?](../Week-03-Mathematics-and-Neural-Network-Foundations/07-What-is-a-probability-distribution.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
