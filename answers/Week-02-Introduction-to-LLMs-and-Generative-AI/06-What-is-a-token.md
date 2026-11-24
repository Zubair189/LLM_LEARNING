# What is a token?

> Week 02 – Introduction to LLMs and Generative AI · Question 6

## In One Line
A token is a **small piece of text**, often a whole word or part of a word, that an LLM reads and writes.

## Simple Explanation
LLMs don't read letters or whole sentences. They read **tokens**.

For example, "Tokenization isn't hard!" might be split like this (each model splits a bit differently):

> `Token` · `ization` · ` isn` · `'t` · ` hard` · `!`

A few things to notice:

- **Common words** are usually **one** token.
- **Rare or long words** are split into **several** pieces.
- The **space** before a word is often part of the token.
- Each token is turned into a number (an **ID**) before the model sees it.

**Why tokens matter to you:**

1. **Cost:** APIs charge per token, for both your input and the model's output.
2. **Limits:** everything you send plus the reply must fit in the **context window** (the model's maximum number of tokens).
3. **Speed:** replies are produced one token at a time, so longer answers take longer.

**Rule of thumb for English:** 1 token ≈ 4 characters ≈ ¾ of a word. So 100 tokens ≈ 75 words. Code and other languages often use more tokens.

Question 7 explains **how** text is split, and Question 8 explains **why** pieces are used instead of whole words.

## Real-Life Example
Tokens are like Lego bricks for text. Common shapes come as one big brick. Unusual shapes are built from several small bricks. The model only ever handles bricks, never whole words.

## Code Example
Estimate tokens and cost with the "4 characters per token" rule.

```python
def estimate_tokens(text):
    return max(1, round(len(text) / 4))      # about 4 characters per token

prompt = "Summarize this customer review in one sentence: " + "great " * 200
input_tokens = estimate_tokens(prompt)
output_tokens = 50                           # expected reply length

price_in, price_out = 3.00, 15.00            # example $ per 1 million tokens (check real prices)
cost = input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out

print(input_tokens, "input tokens")          # 312 input tokens
print(f"cost: ${cost:.6f}")                  # cost: $0.001686
```

**What happened:** we turned text length into an approximate token count, then into money. For exact counts, use the provider's token counter, such as `client.messages.count_tokens(...)` in the Anthropic SDK.

## Watch Out
- **1 token is not 1 word.** On average it is about ¾ of a word.
- **Different models count differently.** Each model family has its own way of splitting text.

## Remember
- A token = a word or piece of a word.
- Cost, limits and speed are all measured in tokens.
- English: about 4 characters per token.

## Related Questions
- [How does tokenization work?](./07-How-does-tokenization-work.md)
- [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
- [What is a context window?](./09-What-is-a-context-window.md)
- [What is token consumption?](../Week-11-Production-LLM-Applications/11-What-is-token-consumption.md)
