# What is inference?

> Week 02 – Introduction to LLMs and Generative AI · Question 11

## In One Line
Inference means **using a trained model** to get an answer for new input. No learning happens.

## Simple Explanation
After training, the model's numbers (weights) are **frozen**. Every time you send a prompt and get a reply, that is inference.

For an LLM, inference works like this:

1. **Read the prompt.** The model processes your whole prompt in one go. This step mainly decides how long you wait for the **first** word to appear.
2. **Write the reply, one token at a time.** For each step, it scores every possible next token, picks one, adds it, and repeats.
3. **Stop** when it produces an "end" token or reaches the `max_tokens` limit.

How does it pick each token?
- **Greedy:** always take the most likely one.
- **Sampling:** add a little randomness, controlled by settings like **temperature** (Week 6).

What engineers care about during inference:
- **Speed:** time to the first token, and tokens per second.
- **Cost:** price per million tokens, or the cost of running your own GPUs.

Inference can run through a **hosted API** (the provider runs the model) or **on your own machine** with tools like Ollama.

Question 14 compares training and inference side by side. This question explains how inference itself works.

## Real-Life Example
Training is a chef spending years learning to cook. Inference is the chef cooking your order tonight, using those skills quickly without learning anything new.

## Code Example
A toy model writes a reply one word at a time, always picking the most likely next word.

```python
NEXT = {                                   # toy "model": next-word chances
    "The": {"sky": 0.7, "cat": 0.3},
    "sky": {"is": 1.0},
    "is": {"blue": 0.8, "green": 0.2},
    "blue": {"<end>": 1.0},
}

words, max_tokens = ["The"], 10            # the prompt, and a length limit
for _ in range(max_tokens):
    probs = NEXT[words[-1]]                # ask the model about the next word
    best = max(probs, key=probs.get)       # greedy: take the most likely
    if best == "<end>":                    # model says "I'm done"
        break
    words.append(best)                     # add it and go again

print(" ".join(words))
# Output:
# The sky is blue
```

**What happened:** the reply grew one word at a time, and the model's numbers never changed. Real LLMs do exactly this loop, just with a huge neural network.

## Watch Out
- **The model does not learn from your prompts.** Its weights stay the same. Examples in a prompt only help for that one request.
- **Inference is not "cheap" at scale.** With millions of users, running the model can cost more than training it.

## Remember
- Inference = using a frozen model to get answers.
- LLMs reply one token at a time.
- Engineers optimise speed and cost.

## Related Questions
- [What is the difference between training and inference?](./14-What-is-the-difference-between-training-and-inference.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [What are top-k and top-p sampling?](../Week-06-Working-with-LLMs-and-Model-APIs/13-What-are-top-k-and-top-p-sampling.md)
