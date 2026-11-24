# What is a Large Language Model (LLM)?

> Week 02 – Introduction to LLMs and Generative AI · Question 2

## In One Line
An LLM is a very big neural network that has read a huge amount of text and learned to **predict the next piece of text**, which lets it write, answer and chat.

## Simple Explanation
Let's break down the name:

- **Large:** it has billions of **parameters** (the learned numbers inside it) and was trained on trillions of words.
- **Language:** it works with human language, and also code.
- **Model:** a learned program that takes text in and predicts what comes next.

How it writes an answer:

1. Your text is cut into small pieces called **tokens**.
2. The model scores every possible next token, for example " Paris" 92%, " Lyon" 1%.
3. It picks one, adds it to the text, and repeats until the answer is done.

How it is made, in three stages:

1. **Pretraining:** read a huge amount of text and practise guessing the next token. This gives knowledge.
2. **Fine-tuning and feedback:** learn to follow instructions and be helpful, like a good assistant.
3. **Use (inference):** you send it prompts through an app or an API.

Limits to know:
- It has a **knowledge cutoff**: it does not know about events after its training data ends.
- It can **hallucinate**, meaning it says false things confidently.
- It has **no memory between calls**. The app resends the whole chat each time.

Question 5 explains the general idea of a "language model". This question is about the large, modern ones.

## Real-Life Example
Imagine someone who has read most of the public internet and has amazing autocomplete instincts. Ask them anything and they continue the text in the most likely way. They are usually right, sometimes confidently wrong, and know nothing that happened after they stopped reading.

## Code Example
Calling an LLM with the Anthropic SDK (needs an API key, so the output below is only an example).

```python
import anthropic

client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY from your environment
resp = client.messages.create(
    model="claude-sonnet-5-5",               # which model to use
    max_tokens=150,                          # longest reply allowed, in tokens
    messages=[{"role": "user", "content": "In one sentence, what is an LLM?"}],
)
print(resp.content[0].text)                  # the model's reply
print(resp.usage.input_tokens, resp.usage.output_tokens)   # tokens used (you pay per token)

# e.g. An LLM is a neural network trained on huge amounts of text to predict and generate language.
# e.g. 18 25
```

**What happened:** we sent one message and got back generated text, plus a count of tokens used, which is how API pricing works.

## Watch Out
- **It doesn't search the internet by default.** It answers from what it learned, unless you give it tools.
- **"Just autocomplete" undersells it.** At this scale, predicting the next token leads to real writing and problem-solving skills, though still imperfect.

## Remember
- LLM = huge neural network that predicts the next token.
- Pretraining gives knowledge; fine-tuning makes it a helpful assistant.
- Limits: knowledge cutoff, hallucinations, no memory between calls.

## Related Questions
- [What is a language model?](./05-What-is-a-language-model.md)
- [What is a token?](./06-What-is-a-token.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](./15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
